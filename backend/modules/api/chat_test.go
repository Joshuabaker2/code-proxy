package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"code-proxy/modules/account"
	"code-proxy/modules/database"
	"code-proxy/modules/provider"
)

type authRecoveryProvider struct {
	mu     sync.Mutex
	tokens []string
}

func (p *authRecoveryProvider) Name() string             { return "auth-recovery" }
func (p *authRecoveryProvider) Models() []provider.Model { return nil }
func (p *authRecoveryProvider) Category() string         { return "api" }
func (p *authRecoveryProvider) IsAvailable() bool        { return true }

func (p *authRecoveryProvider) Execute(_ context.Context, req *provider.Request) (<-chan provider.Event, error) {
	p.mu.Lock()
	p.tokens = append(p.tokens, req.Account.AccessToken)
	p.mu.Unlock()
	if req.Account.AccessToken == "access-old" {
		return nil, &provider.UpstreamError{
			StatusCode: http.StatusUnauthorized,
			Body:       `{"type":"error","error":{"type":"authentication_error","message":"OAuth access token has been revoked."}}`,
		}
	}
	events := make(chan provider.Event, 2)
	events <- provider.Event{Type: "text", Text: "recovered"}
	events <- provider.Event{Type: "done"}
	close(events)
	return events, nil
}

func TestChunkHasFinishReason(t *testing.T) {
	if chunkHasFinishReason(`{"choices":[{"delta":{},"finish_reason":null}]}`) {
		t.Fatal("null finish reason was treated as final")
	}
	if !chunkHasFinishReason(`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`) {
		t.Fatal("tool_calls finish reason was not detected")
	}
}

func TestParseModelAndEffort(t *testing.T) {
	tests := []struct {
		model      string
		wantModel  string
		wantEffort string
	}{
		{"cc/claude-opus-5:low", "cc/claude-opus-5", "low"},
		{"cc/claude-opus-5:med", "cc/claude-opus-5", "medium"},
		{"cc/claude-opus-5:xhigh", "cc/claude-opus-5", "xhigh"},
		{"cc/claude-opus-5:max", "cc/claude-opus-5", "max"},
		{"cc/claude-opus-5:unknown", "cc/claude-opus-5:unknown", ""},
	}

	for _, test := range tests {
		gotModel, gotEffort := parseModelAndEffort(test.model)
		if gotModel != test.wantModel || gotEffort != test.wantEffort {
			t.Errorf("parseModelAndEffort(%q) = (%q, %q), want (%q, %q)",
				test.model, gotModel, gotEffort, test.wantModel, test.wantEffort)
		}
	}
}

func TestNormalizeZedReasoningEffort(t *testing.T) {
	tests := map[string]string{
		"none":    "",
		"minimal": "low",
		"low":     "low",
		"medium":  "medium",
		"high":    "high",
		"xhigh":   "xhigh",
		"max":     "max",
		"unknown": "",
	}
	for input, want := range tests {
		if got := normalizeEffort(input); got != want {
			t.Errorf("normalizeEffort(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestProviderErrorStatusPreservesUpstreamStatus(t *testing.T) {
	upstream := &provider.UpstreamError{
		StatusCode: http.StatusTooManyRequests,
		Body:       `{"type":"error","error":{"type":"rate_limit_error"}}`,
	}
	if got := providerErrorStatus(upstream); got != http.StatusTooManyRequests {
		t.Fatalf("providerErrorStatus() = %d, want %d", got, http.StatusTooManyRequests)
	}
	if got := providerErrorStatus(errors.New("network failed")); got != http.StatusInternalServerError {
		t.Fatalf("providerErrorStatus() = %d, want %d", got, http.StatusInternalServerError)
	}
}

func TestExtractResponseUsageIncludesCacheBreakdown(t *testing.T) {
	json := `{
		"choices":[],
		"usage":{
			"prompt_tokens":155,
			"completion_tokens":20,
			"total_tokens":175,
			"prompt_tokens_details":{
				"cached_tokens":120,
				"cache_creation_tokens":25
			}
		}
	}`
	usage, ok := extractResponseUsage(json)
	if !ok {
		t.Fatal("usage was not extracted")
	}
	if usage.InputTokens != 155 ||
		usage.OutputTokens != 20 ||
		usage.CacheCreationInputTokens != 25 ||
		usage.CacheReadInputTokens != 120 ||
		!usage.Actual {
		t.Fatalf("unexpected usage: %#v", usage)
	}
}

func TestStreamResponseForwardsProviderErrorWithoutSuccessfulStop(t *testing.T) {
	recorder := httptest.NewRecorder()
	events := make(chan provider.Event, 1)
	events <- provider.Event{
		Type: "error",
		Text: "Anthropic overloaded_error: Overloaded (request_id: req_error)",
	}
	close(events)

	streamResponse(recorder, events, "claude-opus-5", "cc/claude-opus-5")

	body := recorder.Body.String()
	if !strings.Contains(body, `"error"`) ||
		!strings.Contains(body, "overloaded_error") ||
		!strings.Contains(body, "req_error") {
		t.Fatalf("streamed error details were not forwarded: %s", body)
	}
	for _, forbidden := range []string{`(no response)`, `"finish_reason":"stop"`, `[DONE]`} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("stream error was disguised as success (%q): %s", forbidden, body)
		}
	}
}

func TestChatRecoversFromUpstream401WithRotatedOAuthToken(t *testing.T) {
	db, err := database.Open(filepath.Join(t.TempDir(), "data.db"))
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	expiresAt := time.Now().Add(4 * time.Hour)
	created, err := db.CreateAccountFull(
		"anthropic-api",
		"Claude",
		"oauth",
		"access-old",
		"refresh-old",
		"",
		&expiresAt,
		nil,
	)
	if err != nil {
		t.Fatalf("create account: %v", err)
	}

	manager := account.NewManager(db)
	refreshCalls := 0
	manager.SetTokenRefresher(func(stored *provider.Account) (account.RefreshedTokens, error) {
		refreshCalls++
		return account.RefreshedTokens{
			AccessToken:  "access-new",
			RefreshToken: "refresh-new",
			ExpiresAt:    time.Now().Add(8 * time.Hour),
		}, nil
	})
	fakeProvider := &authRecoveryProvider{}
	registry := provider.NewRegistry()
	registry.Register("anthropic-api", fakeProvider)
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"claude-sonnet-5","messages":[{"role":"user","content":"hello"}],"stream":false}`),
	)

	handleChat(registry, manager, "claude-sonnet-5", db).ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), "recovered") {
		t.Fatalf("chat response = %d %s, want recovered 200", recorder.Code, recorder.Body.String())
	}
	if refreshCalls != 1 {
		t.Fatalf("refresh calls = %d, want 1", refreshCalls)
	}
	fakeProvider.mu.Lock()
	gotTokens := append([]string(nil), fakeProvider.tokens...)
	fakeProvider.mu.Unlock()
	if strings.Join(gotTokens, ",") != "access-old,access-new" {
		t.Fatalf("provider tokens = %v, want old then new", gotTokens)
	}
	stored, err := db.GetAccount(created.ID)
	if err != nil {
		t.Fatalf("get refreshed account: %v", err)
	}
	if stored.AccessToken != "access-new" || stored.RefreshToken != "refresh-new" {
		t.Fatalf("stored tokens were not rotated: %#v", stored)
	}
	if stored.CooldownUntil != nil {
		t.Fatalf("successful auth recovery left a cooldown: %v", stored.CooldownUntil)
	}
}
