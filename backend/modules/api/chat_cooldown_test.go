package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"code-proxy/modules/account"
	"code-proxy/modules/database"
	"code-proxy/modules/provider"
)

// flakyProvider fails the first attempt with a transient upstream status and
// succeeds afterwards, mimicking Anthropic having a bad moment on a large
// request.
type flakyProvider struct {
	mu         sync.Mutex
	failStatus int
	failures   int
	calls      int
	sawNilAcct bool
}

func (p *flakyProvider) Name() string             { return "flaky" }
func (p *flakyProvider) Models() []provider.Model { return nil }
func (p *flakyProvider) Category() string         { return "api" }
func (p *flakyProvider) IsAvailable() bool        { return true }

func (p *flakyProvider) Execute(_ context.Context, req *provider.Request) (<-chan provider.Event, error) {
	p.mu.Lock()
	p.calls++
	call := p.calls
	if req.Account == nil {
		p.sawNilAcct = true
	}
	failures := p.failures
	status := p.failStatus
	p.mu.Unlock()

	// Mirror the real provider: a nil account is reported as an auth problem.
	if req.Account == nil {
		return nil, &provider.UpstreamError{
			StatusCode: http.StatusInternalServerError,
			Body:       "Anthropic API requires a configured account",
		}
	}

	if call <= failures {
		return nil, &provider.UpstreamError{
			StatusCode: status,
			Body:       `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
		}
	}

	events := make(chan provider.Event, 2)
	events <- provider.Event{Type: "text", Text: "analysed"}
	events <- provider.Event{Type: "done"}
	close(events)
	return events, nil
}

func newChatTestDB(t *testing.T) *database.DB {
	t.Helper()
	db, err := database.Open(filepath.Join(t.TempDir(), "data.db"))
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func postChat(t *testing.T, registry *provider.Registry, manager *account.Manager, db *database.DB) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"claude-sonnet-5","messages":[{"role":"user","content":"hello"}],"stream":false}`),
	)
	handleChat(registry, manager, "claude-sonnet-5", db).ServeHTTP(recorder, request)
	return recorder
}

// The reported bug: a single transient upstream failure used to cool down the
// only account, so the retry inside the same request found nothing selectable
// and reported "Anthropic API requires a configured account" — an
// authentication error for a perfectly authenticated user.
func TestTransientUpstreamFailureIsNotReportedAsMissingAccount(t *testing.T) {
	for _, status := range []int{500, 503, 529} {
		db := newChatTestDB(t)
		if _, err := db.CreateAccountFull("anthropic-api", "Claude", "api_key", "", "", "sk-test", nil, nil); err != nil {
			t.Fatalf("create account: %v", err)
		}

		fake := &flakyProvider{failStatus: status, failures: 1}
		registry := provider.NewRegistry()
		registry.Register("anthropic-api", fake)

		recorder := postChat(t, registry, account.NewManager(db), db)

		if body := recorder.Body.String(); strings.Contains(body, "requires a configured account") {
			t.Fatalf("status %d: response falsely blamed account configuration: %s", status, body)
		}
		fake.mu.Lock()
		sawNil := fake.sawNilAcct
		fake.mu.Unlock()
		if sawNil {
			t.Fatalf("status %d: retry dispatched with a nil account", status)
		}
		if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), "analysed") {
			t.Fatalf("status %d: response = %d %s, want a successful retry", status, recorder.Code, recorder.Body.String())
		}
	}
}

// When every attempt fails, the client must learn what actually went wrong
// rather than a message about account configuration.
func TestExhaustedRetriesReportTheRealUpstreamError(t *testing.T) {
	db := newChatTestDB(t)
	if _, err := db.CreateAccountFull("anthropic-api", "Claude", "api_key", "", "", "sk-test", nil, nil); err != nil {
		t.Fatalf("create account: %v", err)
	}

	fake := &flakyProvider{failStatus: 529, failures: maxRetries}
	registry := provider.NewRegistry()
	registry.Register("anthropic-api", fake)

	recorder := postChat(t, registry, account.NewManager(db), db)

	body := recorder.Body.String()
	if strings.Contains(body, "requires a configured account") {
		t.Fatalf("response falsely blamed account configuration: %s", body)
	}
	if !strings.Contains(body, "overloaded_error") {
		t.Fatalf("response = %s, want the real upstream error preserved", body)
	}
	if recorder.Code != 529 {
		t.Errorf("status = %d, want the upstream 529 preserved", recorder.Code)
	}
}

// An account that is genuinely rate limited should surface as a wait, with a
// Retry-After the client can act on — not as an auth failure.
func TestRateLimitedAccountReportsRetryAfter(t *testing.T) {
	db := newChatTestDB(t)
	if _, err := db.CreateAccountFull("anthropic-api", "Claude", "api_key", "", "", "sk-test", nil, nil); err != nil {
		t.Fatalf("create account: %v", err)
	}

	fake := &flakyProvider{failStatus: http.StatusTooManyRequests, failures: maxRetries}
	registry := provider.NewRegistry()
	registry.Register("anthropic-api", fake)

	recorder := postChat(t, registry, account.NewManager(db), db)

	if body := recorder.Body.String(); strings.Contains(body, "requires a configured account") {
		t.Fatalf("response falsely blamed account configuration: %s", body)
	}
	if recorder.Code != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429 preserved", recorder.Code)
	}
}

// With no accounts at all the old message is still the right one.
func TestMissingAccountStillReportsMissingAccount(t *testing.T) {
	db := newChatTestDB(t)

	fake := &flakyProvider{failStatus: 529}
	registry := provider.NewRegistry()
	registry.Register("anthropic-api", fake)

	recorder := postChat(t, registry, account.NewManager(db), db)

	if !strings.Contains(recorder.Body.String(), "requires a configured account") {
		t.Fatalf("response = %s, want the genuine missing-account error", recorder.Body.String())
	}
}
