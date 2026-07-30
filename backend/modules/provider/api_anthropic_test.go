package provider

import (
	"encoding/json"
	"io"
	"net/http"
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

func TestTranslateOpenAIToAnthropicEnablesAutomaticPromptCaching(t *testing.T) {
	input := []byte(`{
		"model":"claude-opus-5",
		"messages":[{"role":"user","content":"hello"}],
		"stream":true
	}`)

	body, _, err := TranslateOpenAIToAnthropic(input, "claude-opus-5")
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		CacheControl struct {
			Type string `json:"type"`
		} `json:"cache_control"`
		Thinking struct {
			Type    string `json:"type"`
			Display string `json:"display"`
		} `json:"thinking"`
	}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if got.CacheControl.Type != "ephemeral" {
		t.Fatalf("automatic prompt caching was not enabled: %s", body)
	}
	if got.Thinking.Type != "adaptive" || got.Thinking.Display != "summarized" {
		t.Fatalf("adaptive summarized thinking was not enabled: %s", body)
	}
}

func TestTranslateOpenAIToAnthropicDoesNotEnableAdaptiveThinkingForHaiku45(t *testing.T) {
	input := []byte(`{
		"model":"claude-haiku-4-5-20251001",
		"messages":[{"role":"user","content":"hello"}],
		"stream":true
	}`)

	body, _, err := TranslateOpenAIToAnthropic(input, "claude-haiku-4-5-20251001")
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]json.RawMessage
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if got["thinking"] != nil {
		t.Fatalf("Haiku 4.5 does not support adaptive thinking: %s", body)
	}
}

func TestTranslateOpenAIToAnthropicPreservesImageContent(t *testing.T) {
	input := []byte(`{
		"model":"claude-opus-5",
		"messages":[{
			"role":"user",
			"content":[
				{"type":"text","text":"What is in this image?"},
				{
					"type":"image_url",
					"image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}
				}
			]
		}],
		"stream":true
	}`)

	body, _, err := TranslateOpenAIToAnthropic(input, "claude-opus-5")
	if err != nil {
		t.Fatal(err)
	}

	var request struct {
		Messages []struct {
			Content []struct {
				Type   string `json:"type"`
				Text   string `json:"text,omitempty"`
				Source *struct {
					Type      string `json:"type"`
					MediaType string `json:"media_type"`
					Data      string `json:"data"`
				} `json:"source,omitempty"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 1 || len(request.Messages[0].Content) != 2 {
		t.Fatalf("image content was dropped or flattened: %s", body)
	}
	image := request.Messages[0].Content[1]
	if image.Type != "image" || image.Source == nil ||
		image.Source.Type != "base64" ||
		image.Source.MediaType != "image/png" ||
		image.Source.Data != "iVBORw0KGgoAAAANSUhEUg==" {
		t.Fatalf("image was not translated to an Anthropic image block: %#v", image)
	}
}

func TestTranslateOpenAIToAnthropicPreservesToolResultImageContent(t *testing.T) {
	input := []byte(`{
		"model":"claude-opus-5",
		"messages":[
			{"role":"user","content":"Render the map"},
			{
				"role":"assistant",
				"content":null,
				"tool_calls":[{
					"id":"toolu_read_image",
					"type":"function",
					"function":{"name":"read_file","arguments":"{\"path\":\"map.png\"}"}
				}]
			},
			{
				"role":"tool",
				"tool_call_id":"toolu_read_image",
				"content":[
					{"type":"text","text":"Rendered map.png"},
					{
						"type":"image_url",
						"image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}
					}
				]
			}
		],
		"stream":true
	}`)

	body, _, err := TranslateOpenAIToAnthropic(input, "claude-opus-5")
	if err != nil {
		t.Fatal(err)
	}

	var request struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 3 {
		t.Fatalf("tool result message was not preserved: %s", body)
	}
	var content []struct {
		Type    string `json:"type"`
		Content []struct {
			Type   string `json:"type"`
			Text   string `json:"text,omitempty"`
			Source *struct {
				Type      string `json:"type"`
				MediaType string `json:"media_type"`
				Data      string `json:"data"`
			} `json:"source,omitempty"`
		} `json:"content,omitempty"`
	}
	if err := json.Unmarshal(request.Messages[2].Content, &content); err != nil {
		t.Fatal(err)
	}
	if len(content) != 1 {
		t.Fatalf("tool result message was not preserved: %s", body)
	}
	toolResult := content[0]
	if toolResult.Type != "tool_result" || len(toolResult.Content) != 2 {
		t.Fatalf("tool result image content was dropped or flattened: %s", body)
	}
	image := toolResult.Content[1]
	if image.Type != "image" || image.Source == nil ||
		image.Source.Type != "base64" ||
		image.Source.MediaType != "image/png" ||
		image.Source.Data != "iVBORw0KGgoAAAANSUhEUg==" {
		t.Fatalf("tool result image was not translated to an Anthropic image block: %#v", image)
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
		"content":[
			{
				"type":"thinking",
				"thinking":"I should create the requested file.",
				"signature":"signed-thinking"
			},
			{
				"type":"tool_use",
				"id":"toolu_123",
				"name":"create_file",
				"input":{"path":"hello.txt","content":"hello from zed"}
			}
		],
		"usage":{
			"input_tokens":10,
			"cache_creation_input_tokens":15,
			"cache_read_input_tokens":30,
			"output_tokens":20
		}
	}`)

	output, err := TranslateAnthropicResponseToOpenAI(input)
	if err != nil {
		t.Fatal(err)
	}

	var got struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				ReasoningContent string `json:"reasoning_content"`
				ToolCalls        []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
			PromptDetails    struct {
				CachedTokens        int `json:"cached_tokens"`
				CacheCreationTokens int `json:"cache_creation_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
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
	if got.Choices[0].Message.ReasoningContent != "I should create the requested file." {
		t.Fatalf("thinking summary was not preserved: %#v", got.Choices[0].Message)
	}
	if !strings.Contains(calls[0].Function.Arguments, "hello.txt") {
		t.Fatalf("tool arguments were not preserved: %q", calls[0].Function.Arguments)
	}
	if got.Usage.PromptTokens != 55 || got.Usage.CompletionTokens != 20 ||
		got.Usage.TotalTokens != 75 ||
		got.Usage.PromptDetails.CachedTokens != 30 ||
		got.Usage.PromptDetails.CacheCreationTokens != 15 {
		t.Fatalf("usage did not include cached input tokens: %#v", got.Usage)
	}
}

func TestAnthropicStreamResponseEmitsFinalUsageChunk(t *testing.T) {
	body := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_123","model":"claude-opus-5","usage":{"input_tokens":10,"cache_creation_input_tokens":15,"cache_read_input_tokens":30,"output_tokens":1}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	resp := &http.Response{Body: io.NopCloser(strings.NewReader(body))}
	events := make(chan Event, 10)

	NewAnthropicAPI().streamResponse(resp, events, "claude-opus-5")

	var usageChunk struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []any  `json:"choices"`
		Usage   *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
			PromptDetails    struct {
				CachedTokens        int `json:"cached_tokens"`
				CacheCreationTokens int `json:"cache_creation_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	var eventTypes []string
	for event := range events {
		eventTypes = append(eventTypes, event.Type)
		if event.Type != "sse_chunk" {
			continue
		}
		var candidate struct {
			Usage json.RawMessage `json:"usage"`
		}
		if json.Unmarshal([]byte(event.JSON), &candidate) == nil && candidate.Usage != nil {
			if err := json.Unmarshal([]byte(event.JSON), &usageChunk); err != nil {
				t.Fatal(err)
			}
		}
	}

	if usageChunk.Usage == nil {
		t.Fatalf("stream did not emit usage; events were %v", eventTypes)
	}
	if usageChunk.ID != "msg_123" || usageChunk.Model != "claude-opus-5" {
		t.Fatalf("usage chunk lost stream identity: %#v", usageChunk)
	}
	if len(usageChunk.Choices) != 0 {
		t.Fatalf("usage chunk must have no choices: %#v", usageChunk.Choices)
	}
	if usageChunk.Usage.PromptTokens != 55 ||
		usageChunk.Usage.CompletionTokens != 20 ||
		usageChunk.Usage.TotalTokens != 75 ||
		usageChunk.Usage.PromptDetails.CachedTokens != 30 ||
		usageChunk.Usage.PromptDetails.CacheCreationTokens != 15 {
		t.Fatalf("unexpected streamed usage: %#v", usageChunk.Usage)
	}
	if len(eventTypes) == 0 || eventTypes[len(eventTypes)-1] != "done" ||
		eventTypes[len(eventTypes)-2] != "sse_chunk" {
		t.Fatalf("usage must be emitted immediately before done: %v", eventTypes)
	}
}

func TestAnthropicStreamResponseSurfacesErrorEvent(t *testing.T) {
	body := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_error","model":"claude-opus-5","usage":{"input_tokens":173387,"output_tokens":0}}}`,
		``,
		`event: error`,
		`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"},"request_id":"req_error"}`,
		``,
	}, "\n")
	resp := &http.Response{Body: io.NopCloser(strings.NewReader(body))}
	events := make(chan Event, 10)

	NewAnthropicAPI().streamResponse(resp, events, "claude-opus-5")

	var eventTypes []string
	var errorText string
	for event := range events {
		eventTypes = append(eventTypes, event.Type)
		if event.Type == "error" {
			errorText = event.Text
		}
	}

	if len(eventTypes) != 3 ||
		eventTypes[0] != "sse_chunk" ||
		eventTypes[1] != "sse_chunk" ||
		eventTypes[2] != "error" {
		t.Fatalf("expected role, usage, then error without done; got %v", eventTypes)
	}
	if !strings.Contains(errorText, "overloaded_error") ||
		!strings.Contains(errorText, "Overloaded") ||
		!strings.Contains(errorText, "req_error") {
		t.Fatalf("upstream error details were not preserved: %q", errorText)
	}
}

func TestAnthropicStreamTranslatesThinkingAndReplaysSignedBlockForToolResult(t *testing.T) {
	body := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_thinking","model":"claude-opus-5","usage":{"input_tokens":12,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"I should "}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"inspect first."}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"opaque-signature"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_read","name":"read_file","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"main.go\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":18}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	resp := &http.Response{Body: io.NopCloser(strings.NewReader(body))}
	events := make(chan Event, 20)
	api := NewAnthropicAPI()

	api.streamResponse(resp, events, "claude-opus-5")

	var reasoning strings.Builder
	for event := range events {
		if event.Type != "sse_chunk" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(event.JSON), &chunk) == nil && len(chunk.Choices) > 0 {
			reasoning.WriteString(chunk.Choices[0].Delta.ReasoningContent)
		}
		if strings.Contains(event.JSON, "opaque-signature") {
			t.Fatalf("opaque thinking signature leaked into the client stream: %s", event.JSON)
		}
	}
	if reasoning.String() != "I should inspect first." {
		t.Fatalf("unexpected streamed reasoning: %q", reasoning.String())
	}

	openAIRequest := []byte(`{
		"model":"cc/claude-opus-5",
		"stream":true,
		"messages":[
			{"role":"user","content":"Inspect main.go"},
			{
				"role":"assistant",
				"content":null,
				"reasoning_content":"I should inspect first.",
				"tool_calls":[{
					"id":"toolu_read",
					"type":"function",
					"function":{"name":"read_file","arguments":"{\"path\":\"main.go\"}"}
				}]
			},
			{"role":"tool","tool_call_id":"toolu_read","content":"package main"}
		]
	}`)
	translated, _, err := translateOpenAIToAnthropic(
		openAIRequest,
		"claude-opus-5",
		api.replayThinkingBlocks,
	)
	if err != nil {
		t.Fatal(err)
	}
	var request struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(translated, &request); err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) < 2 {
		t.Fatalf("unexpected translated assistant tool message: %s", translated)
	}
	var assistantContent []map[string]any
	if err := json.Unmarshal(request.Messages[1].Content, &assistantContent); err != nil {
		t.Fatal(err)
	}
	if len(assistantContent) != 2 {
		t.Fatalf("unexpected translated assistant tool message: %s", translated)
	}
	thinking := assistantContent[0]
	if thinking["type"] != "thinking" ||
		thinking["thinking"] != "I should inspect first." ||
		thinking["signature"] != "opaque-signature" {
		t.Fatalf("signed thinking block was not replayed exactly: %#v", thinking)
	}
	if assistantContent[1]["type"] != "tool_use" {
		t.Fatalf("tool use did not follow thinking block: %#v", assistantContent)
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
			{
				"id":"claude-opus-5",
				"display_name":"Claude Opus 5",
				"max_input_tokens":1000000,
				"max_tokens":128000
			},
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
	if models[0].MaxInputTokens != 1_000_000 || models[0].MaxOutputTokens != 128_000 {
		t.Fatalf("discovered token limits were not preserved: %#v", models[0])
	}
	if models[1].MaxInputTokens != 1_000_000 || models[1].MaxOutputTokens != 128_000 {
		t.Fatalf("fallback token limits were not applied: %#v", models[1])
	}
}
