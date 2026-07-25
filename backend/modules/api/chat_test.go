package api

import "testing"

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
