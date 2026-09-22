package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"code-proxy/modules/account"
	"code-proxy/modules/database"
	"code-proxy/modules/provider"
)

// scriptedProvider replays a fixed event sequence, letting a test stage a
// stream that dies partway through.
type scriptedProvider struct {
	events []provider.Event
}

func (p *scriptedProvider) Name() string             { return "scripted" }
func (p *scriptedProvider) Models() []provider.Model { return nil }
func (p *scriptedProvider) Category() string         { return "api" }
func (p *scriptedProvider) IsAvailable() bool        { return true }

func (p *scriptedProvider) Execute(_ context.Context, _ *provider.Request) (<-chan provider.Event, error) {
	events := make(chan provider.Event, len(p.events)+1)
	for _, event := range p.events {
		events <- event
	}
	close(events)
	return events, nil
}

func postScripted(t *testing.T, db *database.DB, manager *account.Manager, stream bool, events []provider.Event) *httptest.ResponseRecorder {
	t.Helper()
	registry := provider.NewRegistry()
	registry.Register("anthropic-api", &scriptedProvider{events: events})

	body := `{"model":"claude-sonnet-5","messages":[{"role":"user","content":"hi"}],"stream":` +
		map[bool]string{true: "true", false: "false"}[stream] + `}`
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	handleChat(registry, manager, "claude-sonnet-5", db).ServeHTTP(recorder, request)
	return recorder
}

func withAccount(t *testing.T) (*database.DB, *account.Manager, string) {
	t.Helper()
	db := newChatTestDB(t)
	created, err := db.CreateAccountFull("anthropic-api", "Claude", "api_key", "", "", "sk-test", nil, nil)
	if err != nil {
		t.Fatalf("create account: %v", err)
	}
	return db, account.NewManager(db), created.ID
}

// A stream that stops mid-message used to be dressed up with a synthesized
// finish_reason "stop", so a truncated analysis looked like a complete one.
func TestTruncatedStreamIsReportedRatherThanFinishedCleanly(t *testing.T) {
	db, manager, accountID := withAccount(t)

	recorder := postScripted(t, db, manager, true, []provider.Event{
		{Type: "text", Text: "partial answer"},
		// No done event, no finish_reason: the upstream connection dropped.
	})

	body := recorder.Body.String()
	if strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("truncated stream was reported as a clean stop: %s", body)
	}
	if !strings.Contains(body, "upstream_stream_error") {
		t.Errorf("truncated stream did not surface an error frame: %s", body)
	}
	if !strings.Contains(body, "partial answer") {
		t.Errorf("content received before the truncation was dropped: %s", body)
	}

	stored, err := db.GetAccount(accountID)
	if err != nil {
		t.Fatalf("reload account: %v", err)
	}
	if stored.LastError == "" {
		t.Error("a truncated stream was recorded as a clean success")
	}
}

// A mid-stream error event must not count as a success either.
func TestMidStreamErrorIsNotRecordedAsSuccess(t *testing.T) {
	db, manager, accountID := withAccount(t)

	recorder := postScripted(t, db, manager, true, []provider.Event{
		{Type: "text", Text: "started"},
		{Type: "error", Text: "Overloaded"},
	})

	if body := recorder.Body.String(); !strings.Contains(body, "Overloaded") {
		t.Errorf("stream error was not surfaced: %s", body)
	}

	stored, err := db.GetAccount(accountID)
	if err != nil {
		t.Fatalf("reload account: %v", err)
	}
	if !strings.Contains(stored.LastError, "Overloaded") {
		t.Errorf("LastError = %q, want the stream failure recorded", stored.LastError)
	}
	// Upstream overload is not the account's fault, so it must stay selectable.
	if selected, err := manager.Select("anthropic-api", "claude-sonnet-5"); err != nil || selected == nil {
		t.Errorf("account became unselectable after an upstream stream error (err=%v)", err)
	}
}

// A stream whose upstream already signalled completion in-band is clean, even
// though no done event follows. This guards against over-correcting.
func TestStreamWithInBandFinishReasonIsClean(t *testing.T) {
	db, manager, accountID := withAccount(t)

	recorder := postScripted(t, db, manager, true, []provider.Event{
		{Type: "sse_chunk", JSON: `{"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}`},
	})

	body := recorder.Body.String()
	if strings.Contains(body, "upstream_stream_error") {
		t.Errorf("a cleanly finished stream was flagged as truncated: %s", body)
	}
	if !strings.Contains(body, "[DONE]") {
		t.Errorf("stream was not terminated with [DONE]: %s", body)
	}

	stored, err := db.GetAccount(accountID)
	if err != nil {
		t.Fatalf("reload account: %v", err)
	}
	if stored.LastError != "" {
		t.Errorf("LastError = %q, want a clean stream to record nothing", stored.LastError)
	}
}

// The non-streaming path dropped error events entirely, returning HTTP 200
// with whatever partial text had arrived.
func TestNonStreamingErrorReturnsAnErrorStatus(t *testing.T) {
	db, manager, _ := withAccount(t)

	recorder := postScripted(t, db, manager, false, []provider.Event{
		{Type: "text", Text: "partial"},
		{Type: "error", Text: "Overloaded"},
	})

	if recorder.Code == http.StatusOK {
		t.Errorf("non-streaming upstream error returned 200: %s", recorder.Body.String())
	}
	if recorder.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", recorder.Code)
	}
	if body := recorder.Body.String(); !strings.Contains(body, "Overloaded") {
		t.Errorf("error body = %s, want the upstream message", body)
	}
}

// A normal streamed completion still works end to end.
func TestHealthyStreamStillSucceeds(t *testing.T) {
	db, manager, accountID := withAccount(t)

	recorder := postScripted(t, db, manager, true, []provider.Event{
		{Type: "text", Text: "all good"},
		{Type: "done"},
	})

	body := recorder.Body.String()
	if !strings.Contains(body, "all good") || !strings.Contains(body, "[DONE]") {
		t.Errorf("healthy stream body = %s", body)
	}
	if strings.Contains(body, "upstream_stream_error") {
		t.Errorf("healthy stream reported an error: %s", body)
	}

	stored, err := db.GetAccount(accountID)
	if err != nil {
		t.Fatalf("reload account: %v", err)
	}
	if stored.LastError != "" {
		t.Errorf("LastError = %q, want empty after a healthy stream", stored.LastError)
	}
}
