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
	"net/http"
	"strings"
)

const anthropicBaseURL = "https://api.anthropic.com"
const anthropicVersion = "2023-06-01"
const claudeCodeVersion = "2.1.219"
const claudeCodeBillingVersion = "2.1.219.526"

// AnthropicAPI is the provider for the Anthropic Messages API with format translation
type AnthropicAPI struct{}

// NewAnthropicAPI creates an Anthropic API provider
func NewAnthropicAPI() *AnthropicAPI {
	return &AnthropicAPI{}
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
	httpReq.Header.Set("User-Agent", "claude-cli/"+claudeCodeVersion+" (external, sdk-cli)")
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
			ID          string `json:"id"`
			DisplayName string `json:"display_name"`
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
		models = append(models, Model{
			ID:      "cc/" + item.ID,
			Name:    name,
			OwnedBy: "anthropic",
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
	claudeBody, _, err := TranslateOpenAIToAnthropic(req.RawBody, req.Model)
	if err != nil {
		return nil, fmt.Errorf("translate to anthropic: %w", err)
	}
	claudeBody, err = applyAnthropicEffort(claudeBody, req.Effort)
	if err != nil {
		return nil, fmt.Errorf("apply Claude effort: %w", err)
	}

	var claudeCodeSessionID string
	if req.Account.AuthMode == "oauth" && req.Account.AccessToken != "" {
		claudeBody, claudeCodeSessionID, err = prepareClaudeCodeOAuthBody(claudeBody, req.Account.ID)
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
	if req.Account.AuthMode == "oauth" && req.Account.AccessToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+req.Account.AccessToken)
		httpReq.Header.Set("anthropic-beta", "oauth-2025-04-20,claude-code-20250219")
		httpReq.Header.Set("User-Agent", "claude-cli/"+claudeCodeVersion+" (external, sdk-cli)")
		httpReq.Header.Set("x-app", "cli")
		httpReq.Header.Set("anthropic-dangerous-direct-browser-access", "true")
		httpReq.Header.Set("x-claude-code-session-id", claudeCodeSessionID)
	} else {
		httpReq.Header.Set("x-api-key", req.Account.AuthToken())
	}

	log.Printf("[ANTHROPIC] %s → %s (stream=%v, %d bytes)", req.Model, url, req.Stream, len(claudeBody))

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("upstream request: %w", err)
	}

	// Check for HTTP error
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		errBody, _ := io.ReadAll(resp.Body)
		return nil, &UpstreamError{
			StatusCode: resp.StatusCode,
			Body:       string(errBody),
		}
	}

	events := make(chan Event, 128)

	if req.Stream {
		go p.streamResponse(resp, events)
	} else {
		go p.nonStreamResponse(resp, events)
	}

	return events, nil
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
			"text": "x-anthropic-billing-header: cc_version=" + claudeCodeBillingVersion + "; cc_entrypoint=sdk-cli;",
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

// streamResponse reads Claude SSE and translates to OpenAI format
func (p *AnthropicAPI) streamResponse(resp *http.Response, events chan<- Event) {
	defer close(events)
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	var eventType string
	var chatID string
	toolIndexes := make(map[int]int)

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
		log.Printf("[ANTHROPIC] Stream scan error: %v", err)
		events <- Event{Type: "error", Text: err.Error()}
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
func (p *AnthropicAPI) nonStreamResponse(resp *http.Response, events chan<- Event) {
	defer close(events)
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		events <- Event{Type: "error", Text: err.Error()}
		return
	}

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
