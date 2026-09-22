package provider

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"
)

// --- OpenAI → Anthropic Messages API ---

// TranslateOpenAIToAnthropic translates an OpenAI chat/completions request to the Claude Messages API
func TranslateOpenAIToAnthropic(body []byte, model string) ([]byte, string, error) {
	return translateOpenAIToAnthropic(body, model, nil)
}

func translateOpenAIToAnthropic(
	body []byte,
	model string,
	resolveThinking func(json.RawMessage, string) []map[string]any,
) ([]byte, string, error) {
	var openAI struct {
		Model       string            `json:"model"`
		Messages    []json.RawMessage `json:"messages"`
		Stream      bool              `json:"stream"`
		MaxTokens   int               `json:"max_tokens,omitempty"`
		Temperature *float64          `json:"temperature,omitempty"`
		TopP        *float64          `json:"top_p,omitempty"`
		Tools       json.RawMessage   `json:"tools,omitempty"`
		ToolChoice  json.RawMessage   `json:"tool_choice,omitempty"`
		Stop        json.RawMessage   `json:"stop,omitempty"`
	}
	if err := json.Unmarshal(body, &openAI); err != nil {
		return nil, "", fmt.Errorf("parse OpenAI request: %w", err)
	}

	// Build the Anthropic request
	claude := map[string]any{
		"model":         mapModelToAnthropic(model),
		"stream":        openAI.Stream,
		"cache_control": map[string]string{"type": "ephemeral"},
	}
	if ClaudeSupportsAdaptiveThinking(model) {
		claude["thinking"] = map[string]string{
			"type":    "adaptive",
			"display": "summarized",
		}
	}

	// Max tokens (required by Claude)
	maxTokens := openAI.MaxTokens
	if maxTokens == 0 {
		maxTokens = 8192
	}
	claude["max_tokens"] = maxTokens

	if openAI.Temperature != nil {
		claude["temperature"] = *openAI.Temperature
	}
	if openAI.TopP != nil {
		claude["top_p"] = *openAI.TopP
	}
	if openAI.Stop != nil {
		claude["stop_sequences"] = openAI.Stop
	}

	// Split system and convert messages
	var systemParts []string
	var claudeMessages []map[string]any

	for _, rawMsg := range openAI.Messages {
		var msg struct {
			Role             string          `json:"role"`
			Content          json.RawMessage `json:"content"`
			ReasoningContent string          `json:"reasoning_content,omitempty"`
			ToolCalls        json.RawMessage `json:"tool_calls,omitempty"`
			ToolCallID       string          `json:"tool_call_id,omitempty"`
		}
		if json.Unmarshal(rawMsg, &msg) != nil {
			continue
		}

		switch msg.Role {
		case "system":
			// System goes into the top-level field
			text := extractTextFromContent(msg.Content)
			if text != "" {
				systemParts = append(systemParts, text)
			}

		case "user":
			claudeMsg := map[string]any{
				"role":    "user",
				"content": convertContent(normalizeGooseUserContent(msg.Content)),
			}
			claudeMessages = append(claudeMessages, claudeMsg)

		case "assistant":
			claudeMsg := map[string]any{
				"role": "assistant",
			}
			// If there are tool_calls, convert them into content blocks
			if msg.ToolCalls != nil {
				var thinkingBlocks []map[string]any
				if resolveThinking != nil {
					thinkingBlocks = resolveThinking(msg.ToolCalls, mapModelToAnthropic(model))
				}
				content := convertAssistantWithToolCalls(msg.Content, msg.ToolCalls, thinkingBlocks)
				claudeMsg["content"] = content
			} else {
				// OpenAI-compatible clients can replay visible reasoning in
				// reasoning_content, but Anthropic only accepts its original
				// signed thinking blocks. Outside a tool-use continuation those
				// blocks may be omitted, so never turn replayed reasoning into
				// visible assistant text.
				claudeMsg["content"] = convertContent(msg.Content)
			}
			claudeMessages = append(claudeMessages, claudeMsg)

		case "tool":
			// Tool result → Claude tool_result block
			claudeMsg := map[string]any{
				"role": "user",
				"content": []map[string]any{{
					"type":        "tool_result",
					"tool_use_id": msg.ToolCallID,
					"content":     convertContent(msg.Content),
				}},
			}
			claudeMessages = append(claudeMessages, claudeMsg)
		}
	}

	if len(systemParts) > 0 {
		claude["system"] = strings.Join(systemParts, "\n\n")
	}
	claude["messages"] = repairToolPairing(claudeMessages)

	// Convert tools
	if openAI.Tools != nil {
		claudeTools := convertToolsToAnthropic(openAI.Tools)
		if claudeTools != nil {
			claude["tools"] = claudeTools
		}
	}

	// Convert tool_choice
	if openAI.ToolChoice != nil {
		claude["tool_choice"] = convertToolChoiceToAnthropic(openAI.ToolChoice)
	}

	result, err := json.Marshal(claude)
	return result, "application/json", err
}

// --- Anthropic SSE → OpenAI SSE ---

// TranslateAnthropicStreamToOpenAI translates an SSE event from Claude into OpenAI format
func TranslateAnthropicStreamToOpenAI(data []byte) ([]byte, error) {
	var event struct {
		Type  string          `json:"type"`
		Index int             `json:"index"`
		Delta json.RawMessage `json:"delta,omitempty"`

		// message_start
		Message *struct {
			ID    string `json:"id"`
			Model string `json:"model"`
			Usage *struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		} `json:"message,omitempty"`

		// content_block_start
		ContentBlock *struct {
			Type  string `json:"type"`
			ID    string `json:"id,omitempty"`
			Name  string `json:"name,omitempty"`
			Text  string `json:"text,omitempty"`
			Input any    `json:"input,omitempty"`
		} `json:"content_block,omitempty"`
	}

	if err := json.Unmarshal(data, &event); err != nil {
		return nil, nil // Ignore lines that don't parse
	}

	chatID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	switch event.Type {
	case "message_start":
		// Send role
		if event.Message != nil {
			chatID = event.Message.ID
		}
		chunk := map[string]any{
			"id":      chatID,
			"object":  "chat.completion.chunk",
			"created": created,
			"choices": []map[string]any{{
				"index": 0,
				"delta": map[string]any{"role": "assistant"},
			}},
		}
		return json.Marshal(chunk)

	case "content_block_start":
		if event.ContentBlock != nil && event.ContentBlock.Type == "tool_use" {
			// Start of tool_use
			chunk := map[string]any{
				"id":      chatID,
				"object":  "chat.completion.chunk",
				"created": created,
				"choices": []map[string]any{{
					"index": 0,
					"delta": map[string]any{
						"tool_calls": []map[string]any{{
							"index": event.Index,
							"id":    event.ContentBlock.ID,
							"type":  "function",
							"function": map[string]any{
								"name":      event.ContentBlock.Name,
								"arguments": "",
							},
						}},
					},
				}},
			}
			return json.Marshal(chunk)
		}
		return nil, nil

	case "content_block_delta":
		var delta struct {
			Type        string `json:"type"`
			Text        string `json:"text,omitempty"`
			Thinking    string `json:"thinking,omitempty"`
			PartialJSON string `json:"partial_json,omitempty"`
		}
		if json.Unmarshal(event.Delta, &delta) != nil {
			return nil, nil
		}

		switch delta.Type {
		case "thinking_delta":
			chunk := map[string]any{
				"id":      chatID,
				"object":  "chat.completion.chunk",
				"created": created,
				"choices": []map[string]any{{
					"index": 0,
					"delta": map[string]any{"reasoning_content": delta.Thinking},
				}},
			}
			return json.Marshal(chunk)

		case "signature_delta":
			// Signatures are captured by AnthropicAPI for exact replay
			// during tool-use continuations. They are opaque and should not
			// be exposed as visible reasoning text.
			return nil, nil

		case "text_delta":
			chunk := map[string]any{
				"id":      chatID,
				"object":  "chat.completion.chunk",
				"created": created,
				"choices": []map[string]any{{
					"index": 0,
					"delta": map[string]any{"content": delta.Text},
				}},
			}
			return json.Marshal(chunk)

		case "input_json_delta":
			chunk := map[string]any{
				"id":      chatID,
				"object":  "chat.completion.chunk",
				"created": created,
				"choices": []map[string]any{{
					"index": 0,
					"delta": map[string]any{
						"tool_calls": []map[string]any{{
							"index": event.Index,
							"function": map[string]any{
								"arguments": delta.PartialJSON,
							},
						}},
					},
				}},
			}
			return json.Marshal(chunk)
		}
		return nil, nil

	case "message_delta":
		var delta struct {
			StopReason string `json:"stop_reason"`
		}
		if json.Unmarshal(event.Delta, &delta) != nil {
			return nil, nil
		}

		finishReason := "stop"
		if delta.StopReason == "tool_use" {
			finishReason = "tool_calls"
		}

		chunk := map[string]any{
			"id":      chatID,
			"object":  "chat.completion.chunk",
			"created": created,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         map[string]any{},
				"finish_reason": finishReason,
			}},
		}
		return json.Marshal(chunk)

	case "message_stop":
		return nil, nil // Handled in proxyStream as [DONE]
	}

	return nil, nil
}

// --- Anthropic non-stream → OpenAI ---

// TranslateAnthropicResponseToOpenAI translates a complete Claude response into OpenAI format
func TranslateAnthropicResponseToOpenAI(data []byte) ([]byte, error) {
	var claude struct {
		ID         string `json:"id"`
		Type       string `json:"type"`
		Role       string `json:"role"`
		Model      string `json:"model"`
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type      string          `json:"type"`
			Text      string          `json:"text,omitempty"`
			Thinking  string          `json:"thinking,omitempty"`
			Signature string          `json:"signature,omitempty"`
			Data      string          `json:"data,omitempty"`
			ID        string          `json:"id,omitempty"`
			Name      string          `json:"name,omitempty"`
			Input     json.RawMessage `json:"input,omitempty"`
		} `json:"content"`
		Usage struct {
			InputTokens              int `json:"input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			OutputTokens             int `json:"output_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(data, &claude); err != nil {
		return nil, err
	}

	// Extract text and tool_calls
	var textParts []string
	var reasoningParts []string
	var toolCalls []map[string]any

	for _, block := range claude.Content {
		switch block.Type {
		case "thinking":
			if block.Thinking != "" {
				reasoningParts = append(reasoningParts, block.Thinking)
			}
		case "text":
			textParts = append(textParts, block.Text)
		case "tool_use":
			args, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, map[string]any{
				"id":   block.ID,
				"type": "function",
				"function": map[string]any{
					"name":      block.Name,
					"arguments": string(args),
				},
			})
		}
	}

	content := strings.Join(textParts, "")
	finishReason := "stop"
	if claude.StopReason == "tool_use" {
		finishReason = "tool_calls"
	}

	message := map[string]any{
		"role":    "assistant",
		"content": content,
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}
	if len(reasoningParts) > 0 {
		message["reasoning_content"] = strings.Join(reasoningParts, "")
	}

	promptTokens := claude.Usage.InputTokens +
		claude.Usage.CacheCreationInputTokens +
		claude.Usage.CacheReadInputTokens
	openAI := map[string]any{
		"id":      claude.ID,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   claude.Model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       message,
			"finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens":     promptTokens,
			"completion_tokens": claude.Usage.OutputTokens,
			"total_tokens":      promptTokens + claude.Usage.OutputTokens,
			"prompt_tokens_details": map[string]any{
				"cached_tokens":         claude.Usage.CacheReadInputTokens,
				"cache_creation_tokens": claude.Usage.CacheCreationInputTokens,
			},
		},
	}

	return json.Marshal(openAI)
}

// --- Helpers ---

func extractTextFromContent(raw json.RawMessage) string {
	if len(raw) == 0 || strings.TrimSpace(string(raw)) == "null" {
		return ""
	}

	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		var texts []string
		for _, p := range parts {
			if p.Type == "text" {
				texts = append(texts, p.Text)
			}
		}
		return strings.Join(texts, "\n")
	}

	return string(raw)
}

func convertContent(raw json.RawMessage) any {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var arr []map[string]any
	if json.Unmarshal(raw, &arr) == nil {
		converted := make([]map[string]any, 0, len(arr))
		for _, part := range arr {
			converted = append(converted, convertContentPart(part))
		}
		return converted
	}
	return string(raw)
}

const (
	gooseTurnContextOpen  = "<turn-context>"
	gooseTurnContextClose = "</turn-context>"
)

// Goose injects time and working-directory metadata at the front of the
// current human turn, then removes it from that historical message when the
// next human turn starts. Forwarding the transient wrapper makes an otherwise
// identical Anthropic prompt prefix change retroactively and invalidates its
// cache. The sidecar already owns the workspace, so strip only Goose's exact
// generated wrapper while preserving the user's content byte-for-byte.
func normalizeGooseUserContent(raw json.RawMessage) json.RawMessage {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		normalized := stripGooseTurnContext(text)
		if normalized == text {
			return raw
		}
		encoded, err := json.Marshal(normalized)
		if err == nil {
			return encoded
		}
		return raw
	}

	var parts []map[string]any
	if json.Unmarshal(raw, &parts) != nil {
		return raw
	}
	for index, part := range parts {
		if part["type"] != "text" {
			continue
		}
		partText, ok := part["text"].(string)
		if !ok {
			continue
		}
		normalized := stripGooseTurnContext(partText)
		if normalized == partText {
			return raw
		}
		parts[index]["text"] = normalized
		encoded, err := json.Marshal(parts)
		if err == nil {
			return encoded
		}
		return raw
	}
	return raw
}

func stripGooseTurnContext(text string) string {
	if !strings.HasPrefix(text, gooseTurnContextOpen) {
		return text
	}
	end := strings.Index(text, gooseTurnContextClose)
	if end < 0 {
		return text
	}
	wrapper := text[len(gooseTurnContextOpen):end]
	if !strings.Contains(wrapper, "<current-time>") ||
		!strings.Contains(wrapper, "<working-directory>") {
		return text
	}
	contentStart := end + len(gooseTurnContextClose)
	return strings.TrimLeft(text[contentStart:], "\r\n")
}

func convertContentPart(part map[string]any) map[string]any {
	if part["type"] != "image_url" {
		return part
	}

	imageURL, ok := part["image_url"].(map[string]any)
	if !ok {
		return part
	}
	url, ok := imageURL["url"].(string)
	if !ok || url == "" {
		return part
	}

	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return map[string]any{
			"type": "image",
			"source": map[string]any{
				"type": "url",
				"url":  url,
			},
		}
	}

	const dataPrefix = "data:"
	if !strings.HasPrefix(url, dataPrefix) {
		return part
	}
	comma := strings.IndexByte(url, ',')
	if comma < len(dataPrefix) {
		return part
	}
	metadata := url[len(dataPrefix):comma]
	if !strings.HasSuffix(metadata, ";base64") {
		return part
	}
	mediaType := strings.TrimSuffix(metadata, ";base64")
	data := url[comma+1:]
	if mediaType == "" || data == "" {
		return part
	}

	return map[string]any{
		"type": "image",
		"source": map[string]any{
			"type":       "base64",
			"media_type": mediaType,
			"data":       data,
		},
	}
}

// repairToolPairing enforces Anthropic's pairing rule on a translated message
// list: every tool_use block in an assistant turn must be answered by a
// tool_result block in the very next user turn, and no tool_result may refer
// to a tool_use that is not there. OpenAI-shaped clients such as Goose have no
// such rule, so a cancelled tool call, a stream that died mid tool-call, or a
// retry that replays a partial assistant turn hands us a history Anthropic
// rejects outright ("tool_use ids were found without tool_result blocks
// immediately after"). Rather than fail the whole request, answer the orphaned
// calls with an explicit error result and drop results nothing asked for.
//
// Consecutive user messages are merged first: OpenAI clients send one message
// per tool result, and Anthropic treats a run of user messages as one turn.
func repairToolPairing(messages []map[string]any) []map[string]any {
	messages = mergeConsecutiveUserMessages(messages)

	var out []map[string]any
	answered := map[string]bool{}
	var pendingOrder []string
	dropped := 0

	unanswered := func() []map[string]any {
		var blocks []map[string]any
		for _, id := range pendingOrder {
			if !answered[id] {
				blocks = append(blocks, syntheticToolResult(id))
			}
		}
		pendingOrder = nil
		answered = map[string]bool{}
		return blocks
	}

	for _, msg := range messages {
		role, _ := msg["role"].(string)
		switch role {
		case "assistant":
			if missing := unanswered(); len(missing) > 0 {
				// The previous assistant turn's calls were never answered and the
				// conversation moved on; answer them before it does.
				out = append(out, map[string]any{"role": "user", "content": missing})
			}
			for _, block := range contentBlocks(msg["content"]) {
				if block["type"] == "tool_use" {
					if id, ok := block["id"].(string); ok && id != "" {
						pendingOrder = append(pendingOrder, id)
						answered[id] = false
					}
				}
			}
			out = append(out, msg)

		case "user":
			var kept []map[string]any
			for _, block := range contentBlocks(msg["content"]) {
				if block["type"] == "tool_result" {
					id, _ := block["tool_use_id"].(string)
					if _, expected := answered[id]; !expected {
						dropped++
						continue
					}
					answered[id] = true
				}
				kept = append(kept, block)
			}
			// Anthropic wants the results first; anything the client did not
			// answer gets an explicit error result ahead of the rest.
			content := append(unanswered(), kept...)
			if len(content) == 0 {
				continue
			}
			out = append(out, map[string]any{"role": "user", "content": content})

		default:
			out = append(out, msg)
		}
	}

	if missing := unanswered(); len(missing) > 0 {
		// The history ends on the assistant's own tool call with nothing after it.
		out = append(out, map[string]any{"role": "user", "content": missing})
	}
	if dropped > 0 {
		log.Printf("[TRANSLATE] Dropped %d tool_result block(s) with no matching tool_use", dropped)
	}
	return out
}

func syntheticToolResult(toolUseID string) map[string]any {
	return map[string]any{
		"type":        "tool_result",
		"tool_use_id": toolUseID,
		"is_error":    true,
		"content":     "No result was recorded for this tool call. Treat it as failed; if the information is still needed, call the tool again.",
	}
}

// mergeConsecutiveUserMessages folds runs of user messages into one so that
// pairing can be judged per turn, the way Anthropic judges it.
func mergeConsecutiveUserMessages(messages []map[string]any) []map[string]any {
	var out []map[string]any
	for _, msg := range messages {
		role, _ := msg["role"].(string)
		if role == "user" && len(out) > 0 {
			if prev := out[len(out)-1]; prev["role"] == "user" {
				merged := append(contentBlocks(prev["content"]), contentBlocks(msg["content"])...)
				out[len(out)-1] = map[string]any{"role": "user", "content": merged}
				continue
			}
		}
		out = append(out, msg)
	}
	return out
}

// contentBlocks normalizes a message's content to a block list: a bare string
// becomes one text block, a block list is returned as is.
func contentBlocks(content any) []map[string]any {
	switch typed := content.(type) {
	case string:
		if typed == "" {
			return nil
		}
		return []map[string]any{{"type": "text", "text": typed}}
	case []map[string]any:
		return typed
	case []any:
		blocks := make([]map[string]any, 0, len(typed))
		for _, item := range typed {
			if block, ok := item.(map[string]any); ok {
				blocks = append(blocks, block)
			}
		}
		return blocks
	default:
		return nil
	}
}

func convertAssistantWithToolCalls(
	content, toolCalls json.RawMessage,
	thinkingBlocks []map[string]any,
) []map[string]any {
	blocks := append([]map[string]any(nil), thinkingBlocks...)

	// Add text block if present
	text := extractTextFromContent(content)
	if text != "" {
		blocks = append(blocks, map[string]any{
			"type": "text",
			"text": text,
		})
	}

	// Convert tool_calls into tool_use blocks
	var calls []struct {
		ID       string `json:"id"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	if json.Unmarshal(toolCalls, &calls) == nil {
		for _, tc := range calls {
			var input any
			json.Unmarshal([]byte(tc.Function.Arguments), &input)
			if input == nil {
				input = map[string]any{}
			}
			blocks = append(blocks, map[string]any{
				"type":  "tool_use",
				"id":    tc.ID,
				"name":  tc.Function.Name,
				"input": input,
			})
		}
	}

	return blocks
}

func convertToolsToAnthropic(raw json.RawMessage) []map[string]any {
	var openAITools []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &openAITools) != nil {
		return nil
	}

	var claudeTools []map[string]any
	for _, t := range openAITools {
		tool := map[string]any{
			"name":         t.Function.Name,
			"description":  t.Function.Description,
			"input_schema": json.RawMessage(t.Function.Parameters),
		}
		claudeTools = append(claudeTools, tool)
	}
	return claudeTools
}

func convertToolChoiceToAnthropic(raw json.RawMessage) any {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		switch s {
		case "auto":
			return map[string]string{"type": "auto"}
		case "none":
			return map[string]string{"type": "auto"} // Claude doesn't have "none", use "auto"
		case "required":
			return map[string]string{"type": "any"}
		}
	}

	// OpenAI object format: {"type":"function","function":{"name":"..."}}
	var obj struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &obj) == nil && obj.Function.Name != "" {
		return map[string]string{
			"type": "tool",
			"name": obj.Function.Name,
		}
	}

	return map[string]string{"type": "auto"}
}

func mapModelToAnthropic(model string) string {
	lower := strings.ToLower(model)

	// If it's already full anthropic format
	if strings.HasPrefix(lower, "claude-") {
		return model
	}

	// Short family names follow the catalog's newest release rather than a
	// pinned ID, so they track new Claude drops with the rest of the proxy.
	family := "sonnet"
	switch {
	case strings.Contains(lower, "fable"):
		family = "fable"
	case strings.Contains(lower, "opus"):
		family = "opus"
	case strings.Contains(lower, "haiku"):
		family = "haiku"
	}
	if id, ok := NewestClaudeModelID(family, ClaudeOAuthModels()); ok {
		return id
	}
	return "claude-sonnet-5"
}
