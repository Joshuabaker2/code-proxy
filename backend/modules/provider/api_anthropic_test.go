package provider

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPrepareClaudeCodeOAuthBody(t *testing.T) {
	input := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"hello"}],
		"system":"Keep edits focused.",
		"metadata":{"trace":"kept"},
		"stream":true
	}`)

	body, sessionID, err := prepareClaudeCodeOAuthBody(input, "account-123")
	if err != nil {
		t.Fatal(err)
	}
	if sessionID == "" {
		t.Fatal("expected a session ID")
	}

	var got struct {
		System []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"system"`
		Metadata map[string]any `json:"metadata"`
	}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.System) != 3 {
		t.Fatalf("expected billing, SDK, and client system blocks; got %d", len(got.System))
	}
	if !strings.HasPrefix(got.System[0].Text, "x-anthropic-billing-header:") {
		t.Fatalf("missing billing header: %q", got.System[0].Text)
	}
	if got.System[2].Text != "Keep edits focused." {
		t.Fatalf("client system prompt was not preserved: %q", got.System[2].Text)
	}
	if got.Metadata["trace"] != "kept" {
		t.Fatalf("existing metadata was not preserved: %#v", got.Metadata)
	}

	rawUserID, ok := got.Metadata["user_id"].(string)
	if !ok {
		t.Fatalf("metadata.user_id is not a JSON string: %#v", got.Metadata["user_id"])
	}
	var userID map[string]string
	if err := json.Unmarshal([]byte(rawUserID), &userID); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"device_id", "account_uuid", "session_id"} {
		if userID[key] == "" {
			t.Fatalf("missing metadata user ID field %q", key)
		}
	}
	if userID["session_id"] != sessionID {
		t.Fatalf("body session ID %q does not match header session ID %q", userID["session_id"], sessionID)
	}
}

func TestApplyAnthropicEffort(t *testing.T) {
	input := []byte(`{
		"model":"claude-opus-5",
		"output_config":{"format":{"type":"json_schema"}}
	}`)
	body, err := applyAnthropicEffort(input, "xhigh")
	if err != nil {
		t.Fatal(err)
	}

	var got struct {
		OutputConfig struct {
			Effort string         `json:"effort"`
			Format map[string]any `json:"format"`
		} `json:"output_config"`
	}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if got.OutputConfig.Effort != "xhigh" {
		t.Fatalf("effort was not forwarded: %#v", got.OutputConfig)
	}
	if got.OutputConfig.Format["type"] != "json_schema" {
		t.Fatalf("existing output_config was not preserved: %#v", got.OutputConfig)
	}
}

func TestApplyAnthropicEffortRejectsUnknownLevel(t *testing.T) {
	if _, err := applyAnthropicEffort([]byte(`{}`), "ultra"); err == nil {
		t.Fatal("expected an unsupported effort error")
	}
}

func TestStableUUID(t *testing.T) {
	first := stableUUID("account", "same")
	if first != stableUUID("account", "same") {
		t.Fatal("stable UUID changed for the same input")
	}
	if first == stableUUID("device", "same") {
		t.Fatal("stable UUID namespaces collided")
	}
}

func TestTranslateAnthropicToolCallToOpenAI(t *testing.T) {
	input := []byte(`{
		"id":"msg_123",
		"type":"message",
		"role":"assistant",
		"model":"claude-sonnet-4-6",
		"stop_reason":"tool_use",
		"content":[{
			"type":"tool_use",
			"id":"toolu_123",
			"name":"create_file",
			"input":{"path":"hello.txt","content":"hello from zed"}
		}],
		"usage":{"input_tokens":10,"output_tokens":20}
	}`)

	output, err := TranslateAnthropicResponseToOpenAI(input)
	if err != nil {
		t.Fatal(err)
	}

	var got struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				ToolCalls []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(output, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Choices) != 1 || got.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("unexpected choices: %#v", got.Choices)
	}
	calls := got.Choices[0].Message.ToolCalls
	if len(calls) != 1 || calls[0].Function.Name != "create_file" {
		t.Fatalf("tool call was not preserved: %#v", calls)
	}
	if !strings.Contains(calls[0].Function.Arguments, "hello.txt") {
		t.Fatalf("tool arguments were not preserved: %q", calls[0].Function.Arguments)
	}
}

func TestNormalizeAnthropicOpenAIChunk(t *testing.T) {
	indexes := make(map[int]int)
	first, chatID := normalizeAnthropicOpenAIChunk(
		[]byte(`{"id":"msg_123","choices":[{"delta":{"role":"assistant"}}]}`),
		"",
		indexes,
	)
	second, secondID := normalizeAnthropicOpenAIChunk(
		[]byte(`{"id":"generated","choices":[{"delta":{"tool_calls":[{"index":2,"id":"toolu_1"}]}}]}`),
		chatID,
		indexes,
	)

	if secondID != "msg_123" {
		t.Fatalf("stream ID changed: %q", secondID)
	}
	if !strings.Contains(string(first), `"id":"msg_123"`) ||
		!strings.Contains(string(second), `"id":"msg_123"`) {
		t.Fatalf("stream chunks do not share one ID: %s / %s", first, second)
	}
	if !strings.Contains(string(second), `"index":0`) {
		t.Fatalf("first OpenAI tool index was not normalized to zero: %s", second)
	}
}

func TestParseClaudeOAuthModels(t *testing.T) {
	body := []byte(`{
		"data": [
			{"id":"claude-opus-5","display_name":"Claude Opus 5"},
			{"id":"claude-fable-5","display_name":"Claude Fable 5"},
			{"id":"not-claude","display_name":"Other"}
		]
	}`)
	models, err := parseClaudeOAuthModels(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 {
		t.Fatalf("expected two Claude models, got %#v", models)
	}
	if models[0].ID != "cc/claude-opus-5" || models[1].ID != "cc/claude-fable-5" {
		t.Fatalf("model IDs were not routed through OAuth: %#v", models)
	}
}
