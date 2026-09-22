package provider

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

const anthropicBaseURL = "https://api.anthropic.com"
const anthropicVersion = "2023-06-01"

const anthropicThinkingReplayTTL = time.Hour

// anthropicResponseHeaderTimeout bounds how long we wait for response headers,
// i.e. time to first byte. It deliberately does NOT bound the body: a streamed
// completion can legitimately run for many minutes, and http.Client.Timeout
// would cut it off mid-stream. Without this a stalled large request hangs
// indefinitely and eventually degrades into an opaque transport error.
const anthropicResponseHeaderTimeout = 2 * time.Minute

// anthropicHTTPClient replaces http.DefaultClient, which has no timeouts at all.
var anthropicHTTPClient = &http.Client{
	Transport: &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: time.Second,
		ResponseHeaderTimeout: anthropicResponseHeaderTimeout,
	},
}

type cachedAnthropicThinking struct {
	model     string
	blocks    []map[string]any
	createdAt time.Time
}

// AnthropicAPI is the provider for the Anthropic Messages API with format translation
type AnthropicAPI struct {
	thinkingMu         sync.Mutex
	thinkingByToolCall map[string]cachedAnthropicThinking
}

// NewAnthropicAPI creates an Anthropic API provider
func NewAnthropicAPI() *AnthropicAPI {
	return &AnthropicAPI{
		thinkingByToolCall: make(map[string]cachedAnthropicThinking),
	}
}

func (p *AnthropicAPI) Name() string      { return "anthropic-api" }
func (p *AnthropicAPI) Category() string  { return "api" }
func (p *AnthropicAPI) IsAvailable() bool { return true }

func (p *AnthropicAPI) Models() []Model {
	return []Model{
		{ID: "anthropic/claude-opus-4-6", Name: "Claude Opus 4.6", OwnedBy: "anthropic"},
		{ID: "anthropic/claude-sonnet-4-6", Name: "Claude Sonnet 4.6", OwnedBy: "anthropic"},
		{ID: "anthropic/claude-haiku-4-5", Name: "Claude Haiku 4.5", OwnedBy: "anthropic"},
	}
}

// DiscoverClaudeOAuthModels returns the model catalog visible to the connected
// Claude subscription. Unlike the proxy's fallback catalog, this stays current
// when Anthropic releases or retires models.
func DiscoverClaudeOAuthModels(ctx context.Context, accessToken string) ([]Model, error) {
	if accessToken == "" {
		return nil, errors.New("Claude OAuth access token is empty")
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", anthropicBaseURL+"/v1/models?limit=100", nil)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+accessToken)
	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("anthropic-version", anthropicVersion)
	httpReq.Header.Set("anthropic-beta", "oauth-2025-04-20,claude-code-20250219")
	httpReq.Header.Set("User-Agent", "claude-cli/"+claudeCodeVersion()+" (external, sdk-cli)")
	httpReq.Header.Set("x-app", "cli")
	httpReq.Header.Set("anthropic-dangerous-direct-browser-access", "true")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("model discovery request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, &UpstreamError{StatusCode: resp.StatusCode, Body: string(body)}
	}
	return parseClaudeOAuthModels(body)
}

func parseClaudeOAuthModels(body []byte) ([]Model, error) {
	var response struct {
		Data []struct {
			ID             string `json:"id"`
			DisplayName    string `json:"display_name"`
			MaxInputTokens *int   `json:"max_input_tokens"`
			MaxTokens      *int   `json:"max_tokens"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, err
	}

	models := make([]Model, 0, len(response.Data))
	for _, item := range response.Data {
		if !strings.HasPrefix(item.ID, "claude-") {
			continue
		}
		name := item.DisplayName
		if name == "" {
			name = item.ID
		}
		maxInputTokens, maxOutputTokens := ClaudeModelLimits(item.ID)
		if item.MaxInputTokens != nil && *item.MaxInputTokens > 0 {
			maxInputTokens = *item.MaxInputTokens
		}
		if item.MaxTokens != nil && *item.MaxTokens > 0 {
			maxOutputTokens = *item.MaxTokens
		}
		models = append(models, Model{
			ID:              "cc/" + item.ID,
			Name:            name,
			OwnedBy:         "anthropic",
			MaxInputTokens:  maxInputTokens,
			MaxOutputTokens: maxOutputTokens,
		})
	}
	if len(models) == 0 {
		return nil, errors.New("Anthropic returned no Claude models")
	}
	return models, nil
}

func (p *AnthropicAPI) Execute(ctx context.Context, req *Request) (<-chan Event, error) {
	if req.Account == nil {
		return nil, fmt.Errorf("Anthropic API requires a configured account")
	}

	// Translate request OpenAI -> Claude Messages API
	claudeBody, _, err := translateOpenAIToAnthropic(
		req.RawBody,
		req.Model,
		p.replayThinkingBlocks,
	)
	if err != nil {
		return nil, fmt.Errorf("translate to anthropic: %w", err)
	}
	claudeBody, err = applyAnthropicEffort(claudeBody, req.Effort)
	if err != nil {
		return nil, fmt.Errorf("apply Claude effort: %w", err)
	}

	oauth := req.Account.AuthMode == "oauth" && req.Account.AccessToken != ""
	translatedBody := claudeBody

	var resp *http.Response
	// One retry: when Anthropic rejects the presented Claude Code version, the
	// rejection names the version it wants. Adopt it and resend rather than
	// surfacing a failure that a newer client would not have hit.
	for attempt := 0; ; attempt++ {
		claudeBody = translatedBody
		var claudeCodeSessionID string
		if oauth {
			claudeBody, claudeCodeSessionID, err = prepareClaudeCodeOAuthBody(translatedBody, req.Account.ID)
			if err != nil {
				return nil, fmt.Errorf("prepare claude oauth request: %w", err)
			}
		}

		// Build HTTP request
		url := anthropicBaseURL + "/v1/messages"
		if claudeCodeSessionID != "" {
			url += "?beta=true"
		}
		httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(claudeBody))
		if err != nil {
			return nil, fmt.Errorf("create request: %w", err)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("anthropic-version", anthropicVersion)

		// Auth: OAuth token (Bearer) or API key (x-api-key)
		if oauth {
			httpReq.Header.Set("Authorization", "Bearer "+req.Account.AccessToken)
			httpReq.Header.Set("anthropic-beta", "oauth-2025-04-20,claude-code-20250219")
			httpReq.Header.Set("User-Agent", "claude-cli/"+claudeCodeVersion()+" (external, sdk-cli)")
			httpReq.Header.Set("x-app", "cli")
			httpReq.Header.Set("anthropic-dangerous-direct-browser-access", "true")
			httpReq.Header.Set("x-claude-code-session-id", claudeCodeSessionID)
		} else {
			httpReq.Header.Set("x-api-key", req.Account.AuthToken())
		}

		log.Printf("[ANTHROPIC] %s → %s (stream=%v, %d bytes)", req.Model, url, req.Stream, len(claudeBody))

		resp, err = anthropicHTTPClient.Do(httpReq)
		if err != nil {
			return nil, fmt.Errorf("upstream request: %w", err)
		}
		if resp.StatusCode < 400 {
			break
		}

		errBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if attempt == 0 && oauth && resp.StatusCode == http.StatusBadRequest &&
			learnClaudeCodeVersionRequirement(string(errBody)) {
			continue
		}
		return nil, &UpstreamError{
			StatusCode: resp.StatusCode,
			Body:       string(errBody),
			RetryAfter: ParseRetryAfter(resp.Header),
		}
	}

	events := make(chan Event, 128)
	anthropicModel := mapModelToAnthropic(req.Model)

	if req.Stream {
		go p.streamResponse(resp, events, anthropicModel)
	} else {
		go p.nonStreamResponse(resp, events, anthropicModel)
	}

	return events, nil
}

func (p *AnthropicAPI) rememberThinkingBlocks(
	toolCallIDs []string,
	model string,
	blocks []map[string]any,
) {
	if len(toolCallIDs) == 0 || len(blocks) == 0 {
		return
	}

	now := time.Now()
	p.thinkingMu.Lock()
	defer p.thinkingMu.Unlock()
	if p.thinkingByToolCall == nil {
		p.thinkingByToolCall = make(map[string]cachedAnthropicThinking)
	}
	for id, cached := range p.thinkingByToolCall {
		if now.Sub(cached.createdAt) > anthropicThinkingReplayTTL {
			delete(p.thinkingByToolCall, id)
		}
	}

	cached := cachedAnthropicThinking{
		model:     model,
		blocks:    cloneAnthropicBlocks(blocks),
		createdAt: now,
	}
	for _, id := range toolCallIDs {
		if id != "" {
			p.thinkingByToolCall[id] = cached
		}
	}
}

func (p *AnthropicAPI) replayThinkingBlocks(
	rawToolCalls json.RawMessage,
	model string,
) []map[string]any {
	var calls []struct {
		ID string `json:"id"`
	}
	if json.Unmarshal(rawToolCalls, &calls) != nil {
		return nil
	}

	now := time.Now()
	p.thinkingMu.Lock()
	defer p.thinkingMu.Unlock()
	for id, cached := range p.thinkingByToolCall {
		if now.Sub(cached.createdAt) > anthropicThinkingReplayTTL {
			delete(p.thinkingByToolCall, id)
			continue
		}
		if cached.model != model {
			continue
		}
		for _, call := range calls {
			if call.ID == id {
				return cloneAnthropicBlocks(cached.blocks)
			}
		}
	}
	return nil
}

func cloneAnthropicBlocks(blocks []map[string]any) []map[string]any {
	cloned := make([]map[string]any, 0, len(blocks))
	for _, block := range blocks {
		copyBlock := make(map[string]any, len(block))
		for key, value := range block {
			copyBlock[key] = value
		}
		cloned = append(cloned, copyBlock)
	}
	return cloned
}

func applyAnthropicEffort(body []byte, effort string) ([]byte, error) {
	if effort == "" {
		return body, nil
	}
	switch effort {
	case "low", "medium", "high", "xhigh", "max":
	default:
		return nil, fmt.Errorf("unsupported effort level %q", effort)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}

	outputConfig, _ := payload["output_config"].(map[string]any)
	if outputConfig == nil {
		outputConfig = make(map[string]any)
	}
	outputConfig["effort"] = effort
	payload["output_config"] = outputConfig

	return json.Marshal(payload)
}

// prepareClaudeCodeOAuthBody adds the request envelope required by Anthropic's
// Claude subscription OAuth endpoint. The OAuth token alone is not sufficient:
// Claude Code sends a billing marker and a structured user identifier too.
func prepareClaudeCodeOAuthBody(body []byte, accountID string) ([]byte, string, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, "", err
	}

	system := []any{
		map[string]any{
			"type": "text",
			"text": "x-anthropic-billing-header: cc_version=" + claudeCodeBillingVersion() + "; cc_entrypoint=sdk-cli;",
		},
		map[string]any{
			"type": "text",
			"text": "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
		},
	}

	switch existing := payload["system"].(type) {
	case string:
		if existing != "" {
			system = append(system, map[string]any{"type": "text", "text": existing})
		}
	case []any:
		system = append(system, existing...)
	}
	payload["system"] = system

	sessionID, err := randomUUID()
	if err != nil {
		return nil, "", err
	}
	userID, err := json.Marshal(map[string]string{
		"device_id":    stableUUID("device", accountID),
		"account_uuid": stableUUID("account", accountID),
		"session_id":   sessionID,
	})
	if err != nil {
		return nil, "", err
	}

	metadata, _ := payload["metadata"].(map[string]any)
	if metadata == nil {
		metadata = make(map[string]any)
	}
	metadata["user_id"] = string(userID)
	payload["metadata"] = metadata

	result, err := json.Marshal(payload)
	if err != nil {
		return nil, "", err
	}
	return result, sessionID, nil
}

func stableUUID(namespace, value string) string {
	sum := sha256.Sum256([]byte(namespace + "\x00" + value))
	return formatUUID(sum[:16])
}

func randomUUID() (string, error) {
	raw := make([]byte, 16)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	return formatUUID(raw), nil
}

func formatUUID(raw []byte) string {
	value := append([]byte(nil), raw...)
	value[6] = (value[6] & 0x0f) | 0x40
	value[8] = (value[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		value[0:4], value[4:6], value[6:8], value[8:10], value[10:16])
}

type anthropicStreamUsage struct {
	inputTokens              int
	cacheCreationInputTokens int
	cacheReadInputTokens     int
	outputTokens             int
	model                    string
	seen                     bool
}

type anthropicUsageFields struct {
	InputTokens              *int `json:"input_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
	OutputTokens             *int `json:"output_tokens"`
}

func (u *anthropicStreamUsage) observe(data []byte) {
	var event struct {
		Type    string `json:"type"`
		Message *struct {
			Model string                `json:"model"`
			Usage *anthropicUsageFields `json:"usage"`
		} `json:"message,omitempty"`
		Usage *anthropicUsageFields `json:"usage,omitempty"`
	}
	if json.Unmarshal(data, &event) != nil {
		return
	}

	if event.Message != nil {
		if event.Message.Model != "" {
			u.model = event.Message.Model
		}
		u.update(event.Message.Usage)
	}
	u.update(event.Usage)
}

func (u *anthropicStreamUsage) update(usage *anthropicUsageFields) {
	if usage == nil {
		return
	}
	u.seen = true
	if usage.InputTokens != nil {
		u.inputTokens = *usage.InputTokens
	}
	if usage.CacheCreationInputTokens != nil {
		u.cacheCreationInputTokens = *usage.CacheCreationInputTokens
	}
	if usage.CacheReadInputTokens != nil {
		u.cacheReadInputTokens = *usage.CacheReadInputTokens
	}
	if usage.OutputTokens != nil {
		// Anthropic reports cumulative output usage in message_delta events.
		u.outputTokens = *usage.OutputTokens
	}
}

func (u anthropicStreamUsage) openAIChunk(chatID string) ([]byte, error) {
	promptTokens := u.inputTokens + u.cacheCreationInputTokens + u.cacheReadInputTokens
	chunk := map[string]any{
		"id":      chatID,
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"model":   u.model,
		"choices": []any{},
		"usage": map[string]any{
			"prompt_tokens":     promptTokens,
			"completion_tokens": u.outputTokens,
			"total_tokens":      promptTokens + u.outputTokens,
			"prompt_tokens_details": map[string]any{
				"cached_tokens":         u.cacheReadInputTokens,
				"cache_creation_tokens": u.cacheCreationInputTokens,
			},
		},
	}
	return json.Marshal(chunk)
}

func parseAnthropicStreamError(data []byte) (string, bool) {
	var event struct {
		Type  string `json:"type"`
		Error *struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error,omitempty"`
		RequestID string `json:"request_id,omitempty"`
	}
	if json.Unmarshal(data, &event) != nil || event.Type != "error" {
		return "", false
	}

	errorType := "stream_error"
	message := "Anthropic stream failed"
	if event.Error != nil {
		if event.Error.Type != "" {
			errorType = event.Error.Type
		}
		if event.Error.Message != "" {
			message = event.Error.Message
		}
	}

	text := fmt.Sprintf("Anthropic %s: %s", errorType, message)
	if event.RequestID != "" {
		text += fmt.Sprintf(" (request_id: %s)", event.RequestID)
	}
	return text, true
}

type anthropicThinkingBlock struct {
	blockType string
	thinking  string
	signature string
	data      string
}

func (b anthropicThinkingBlock) apiBlock() map[string]any {
	switch b.blockType {
	case "thinking":
		if b.signature == "" {
			return nil
		}
		return map[string]any{
			"type":      "thinking",
			"thinking":  b.thinking,
			"signature": b.signature,
		}
	case "redacted_thinking":
		if b.data == "" {
			return nil
		}
		return map[string]any{
			"type": "redacted_thinking",
			"data": b.data,
		}
	default:
		return nil
	}
}

type anthropicThinkingTracker struct {
	api      *AnthropicAPI
	model    string
	active   map[int]*anthropicThinkingBlock
	finished []map[string]any
}

func newAnthropicThinkingTracker(api *AnthropicAPI, model string) *anthropicThinkingTracker {
	return &anthropicThinkingTracker{
		api:    api,
		model:  model,
		active: make(map[int]*anthropicThinkingBlock),
	}
}

func (t *anthropicThinkingTracker) observe(data []byte) {
	var event struct {
		Type         string `json:"type"`
		Index        int    `json:"index"`
		ContentBlock *struct {
			Type      string `json:"type"`
			ID        string `json:"id,omitempty"`
			Thinking  string `json:"thinking,omitempty"`
			Signature string `json:"signature,omitempty"`
			Data      string `json:"data,omitempty"`
		} `json:"content_block,omitempty"`
		Delta *struct {
			Type      string `json:"type"`
			Thinking  string `json:"thinking,omitempty"`
			Signature string `json:"signature,omitempty"`
		} `json:"delta,omitempty"`
	}
	if json.Unmarshal(data, &event) != nil {
		return
	}

	switch event.Type {
	case "content_block_start":
		if event.ContentBlock == nil {
			return
		}
		switch event.ContentBlock.Type {
		case "thinking", "redacted_thinking":
			t.active[event.Index] = &anthropicThinkingBlock{
				blockType: event.ContentBlock.Type,
				thinking:  event.ContentBlock.Thinking,
				signature: event.ContentBlock.Signature,
				data:      event.ContentBlock.Data,
			}
		case "tool_use":
			if event.ContentBlock.ID != "" {
				t.api.rememberThinkingBlocks(
					[]string{event.ContentBlock.ID},
					t.model,
					t.finished,
				)
			}
		}

	case "content_block_delta":
		block := t.active[event.Index]
		if block == nil || event.Delta == nil {
			return
		}
		switch event.Delta.Type {
		case "thinking_delta":
			block.thinking += event.Delta.Thinking
		case "signature_delta":
			block.signature += event.Delta.Signature
		}

	case "content_block_stop":
		block := t.active[event.Index]
		if block == nil {
			return
		}
		if apiBlock := block.apiBlock(); apiBlock != nil {
			t.finished = append(t.finished, apiBlock)
		}
		delete(t.active, event.Index)
	}
}

// streamResponse reads Claude SSE and translates to OpenAI format
func (p *AnthropicAPI) streamResponse(
	resp *http.Response,
	events chan<- Event,
	model string,
) {
	defer close(events)
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	var eventType string
	var chatID string
	toolIndexes := make(map[int]int)
	var usage anthropicStreamUsage
	thinking := newAnthropicThinkingTracker(p, model)
	var streamError string

	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			continue
		}

		// Claude SSE uses "event: xxx" followed by "data: {json}"
		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		// Inject type in JSON if event type is available
		if eventType != "" {
			var obj map[string]json.RawMessage
			if json.Unmarshal([]byte(data), &obj) == nil {
				if _, hasType := obj["type"]; !hasType {
					typeJSON, _ := json.Marshal(eventType)
					obj["type"] = typeJSON
					newData, _ := json.Marshal(obj)
					data = string(newData)
				}
			}
			eventType = ""
		}

		usage.observe([]byte(data))
		thinking.observe([]byte(data))

		if message, ok := parseAnthropicStreamError([]byte(data)); ok {
			streamError = message
			break
		}

		// Translate to OpenAI
		translated, err := TranslateAnthropicStreamToOpenAI([]byte(data))
		if err != nil {
			log.Printf("[ANTHROPIC] Translate error: %v", err)
			continue
		}
		if translated != nil {
			translated, chatID = normalizeAnthropicOpenAIChunk(translated, chatID, toolIndexes)
			events <- Event{Type: "sse_chunk", JSON: string(translated)}
		}
	}

	if err := scanner.Err(); err != nil {
		streamError = fmt.Sprintf("Anthropic stream read failed: %v", err)
	}

	if usage.seen {
		if chatID == "" {
			chatID = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
		}
		translated, err := usage.openAIChunk(chatID)
		if err != nil {
			log.Printf("[ANTHROPIC] Usage translation error: %v", err)
		} else {
			events <- Event{Type: "sse_chunk", JSON: string(translated)}
		}
	}

	if streamError != "" {
		log.Printf("[ANTHROPIC] Stream error: %s", streamError)
		events <- Event{Type: "error", Text: streamError}
		return
	}

	events <- Event{Type: "done"}
}

func normalizeAnthropicOpenAIChunk(chunk []byte, chatID string, toolIndexes map[int]int) ([]byte, string) {
	var payload map[string]any
	if json.Unmarshal(chunk, &payload) != nil {
		return chunk, chatID
	}

	if id, ok := payload["id"].(string); ok && chatID == "" {
		chatID = id
	}
	if chatID != "" {
		payload["id"] = chatID
	}

	choices, _ := payload["choices"].([]any)
	if len(choices) == 0 {
		result, _ := json.Marshal(payload)
		return result, chatID
	}
	choice, _ := choices[0].(map[string]any)
	delta, _ := choice["delta"].(map[string]any)
	calls, _ := delta["tool_calls"].([]any)
	for _, rawCall := range calls {
		call, _ := rawCall.(map[string]any)
		sourceIndex, _ := call["index"].(float64)
		source := int(sourceIndex)
		index, ok := toolIndexes[source]
		if !ok {
			index = len(toolIndexes)
			toolIndexes[source] = index
		}
		call["index"] = index
	}

	result, err := json.Marshal(payload)
	if err != nil {
		return chunk, chatID
	}
	return result, chatID
}

// nonStreamResponse reads the full response and translates it
func (p *AnthropicAPI) nonStreamResponse(
	resp *http.Response,
	events chan<- Event,
	model string,
) {
	defer close(events)
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		events <- Event{Type: "error", Text: err.Error()}
		return
	}
	p.rememberNonStreamThinking(body, model)

	translated, err := TranslateAnthropicResponseToOpenAI(body)
	if err != nil {
		log.Printf("[ANTHROPIC] Translate response error: %v", err)
		// Fallback: emit raw text
		events <- Event{Type: "text", Text: string(body)}
		events <- Event{Type: "done"}
		return
	}

	// Preserve the complete OpenAI response. Emitting only message.content would
	// silently discard structured tool calls, which are how editors apply native
	// inline changes.
	events <- Event{Type: "sse_chunk", JSON: string(translated)}

	events <- Event{Type: "done"}
}

func (p *AnthropicAPI) rememberNonStreamThinking(body []byte, model string) {
	var response struct {
		Content []struct {
			Type      string `json:"type"`
			ID        string `json:"id,omitempty"`
			Thinking  string `json:"thinking,omitempty"`
			Signature string `json:"signature,omitempty"`
			Data      string `json:"data,omitempty"`
		} `json:"content"`
	}
	if json.Unmarshal(body, &response) != nil {
		return
	}

	var thinkingBlocks []map[string]any
	var toolCallIDs []string
	for _, block := range response.Content {
		switch block.Type {
		case "thinking", "redacted_thinking":
			apiBlock := (anthropicThinkingBlock{
				blockType: block.Type,
				thinking:  block.Thinking,
				signature: block.Signature,
				data:      block.Data,
			}).apiBlock()
			if apiBlock != nil {
				thinkingBlocks = append(thinkingBlocks, apiBlock)
			}
		case "tool_use":
			if block.ID != "" {
				toolCallIDs = append(toolCallIDs, block.ID)
			}
		}
	}
	p.rememberThinkingBlocks(toolCallIDs, model, thinkingBlocks)
}
