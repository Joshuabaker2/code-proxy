package api

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"code-proxy/modules/provider"
)

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
