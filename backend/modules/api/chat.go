package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"code-proxy/modules/account"
	"code-proxy/modules/database"
	"code-proxy/modules/provider"
)

const maxRetries = 3

// handleChat handles POST /v1/chat/completions (OpenAI-compatible)
func handleChat(registry *provider.Registry, acctMgr *account.Manager, defaultModel string, db *database.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			writeError(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			writeError(w, "Failed to read request", http.StatusBadRequest)
			return
		}

		var req ChatRequest
		if err := json.Unmarshal(body, &req); err != nil {
			log.Printf("[CHAT] Invalid JSON: %v | body: %.200s", err, string(body))
			writeError(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		// A model suffix remains supported for clients without a native effort
		// control. Zed sends the selected value as reasoning_effort instead.
		modelStr, effort := parseModelAndEffort(req.Model)
		if effort == "" {
			effort = normalizeEffort(req.ReasoningEffort)
		}
		if modelStr == "" {
			modelStr = defaultModel
		}

		// Check if this is a combo (no "/" prefix = potential combo name)
		if db != nil && !strings.Contains(modelStr, "/") {
			combo, err := db.GetComboByName(modelStr)
			if err == nil && combo != nil && len(combo.Models) > 0 {
				handleComboChat(w, r, body, req, combo, effort, registry, acctMgr, db)
				return
			}
		}

		executeSingleModel(w, r, body, req, modelStr, effort, registry, acctMgr, db)
	}
}

// handleComboChat tries each model in the combo sequentially until one succeeds
func handleComboChat(w http.ResponseWriter, r *http.Request, body []byte, req ChatRequest, combo *database.Combo, effort string, registry *provider.Registry, acctMgr *account.Manager, db *database.DB) {
	log.Printf("[COMBO] Using combo %q with %d models", combo.Name, len(combo.Models))

	var lastErr error
	for i, comboModel := range combo.Models {
		// Parse effort from combo model entry too (e.g. "cc/opus:high")
		comboModelStr, comboEffort := parseModelAndEffort(comboModel)
		if comboEffort == "" {
			comboEffort = effort // use effort from original request
		}

		log.Printf("[COMBO] Trying model %d/%d: %s", i+1, len(combo.Models), comboModelStr)

		err := executeSingleModelForCombo(w, r, body, req, comboModelStr, comboEffort, registry, acctMgr, db)
		if err == nil {
			return // success
		}

		log.Printf("[COMBO] Model %s failed: %v, trying next...", comboModelStr, err)
		lastErr = err
	}

	// All combo models failed
	errMsg := "All models in combo failed"
	if lastErr != nil {
		errMsg = fmt.Sprintf("All models in combo %q failed. Last error: %s", combo.Name, lastErr.Error())
	}
	writeError(w, errMsg, http.StatusServiceUnavailable)
}

// executeSingleModelForCombo tries a single model and returns nil on success or error on failure
// Does NOT write to ResponseWriter on error (so the combo can try the next model)
func executeSingleModelForCombo(w http.ResponseWriter, r *http.Request, body []byte, req ChatRequest, modelStr, effort string, registry *provider.Registry, acctMgr *account.Manager, db *database.DB) error {
	startTime := time.Now()

	p, providerType, cleanModel, err := registry.ResolveProvider(modelStr)
	if err != nil {
		return fmt.Errorf("provider resolve: %w", err)
	}

	apiKeyID := GetApiKeyID(r)
	inputTokens := estimateInputTokens(body)

	authRecoveryAttempted := false
	for attempt := 0; attempt < maxRetries; attempt++ {
		acct, err := acctMgr.Select(providerType, cleanModel)
		if err != nil {
			return fmt.Errorf("no available account: %w", err)
		}

		provReq := &provider.Request{
			RawBody: json.RawMessage(body),
			Model:   cleanModel,
			Effort:  effort,
			Stream:  req.Stream,
			Account: acct,
		}

		events, err := p.Execute(r.Context(), provReq)
		if err != nil {
			status := providerErrorStatus(err)
			if status == http.StatusUnauthorized && !authRecoveryAttempted && acct != nil {
				authRecoveryAttempted = true
				if _, refreshErr := acctMgr.RefreshAfterAuthFailure(acct); refreshErr == nil {
					log.Printf("[CHAT] OAuth token refreshed after upstream 401; retrying %s", cleanModel)
					continue
				} else {
					log.Printf("[CHAT] OAuth recovery failed after upstream 401: %v", refreshErr)
				}
			}
			if acct != nil {
				acctMgr.ReportError(acct.ID, cleanModel, status, err.Error())
			}
			if status >= 400 && status < 500 {
				return err
			}
			continue
		}

		// Success
		if acct != nil {
			acctMgr.ReportSuccess(acct.ID, cleanModel)
		}

		accountID := ""
		if acct != nil {
			accountID = acct.ID
		}

		var tokenUsage requestTokenUsage
		var cost float64
		if req.Stream {
			tokenUsage, cost = streamResponse(w, events, cleanModel, req.Model)
		} else {
			tokenUsage, cost = nonStreamResponse(w, events, cleanModel, req.Model)
		}
		tokenUsage = tokenUsage.forLogging(inputTokens)

		if db != nil {
			durationMs := time.Since(startTime).Milliseconds()
			if cost == 0 {
				inRate, outRate := database.ModelCostRates(cleanModel)
				cost = float64(tokenUsage.InputTokens)/1_000_000*inRate +
					float64(tokenUsage.OutputTokens)/1_000_000*outRate
			}
			db.LogRequest(
				apiKeyID, providerType, cleanModel, effort, accountID,
				tokenUsage.InputTokens, tokenUsage.OutputTokens,
				tokenUsage.CacheCreationInputTokens, tokenUsage.CacheReadInputTokens,
				cost, durationMs,
			)
		}
		return nil
	}

	return fmt.Errorf("all %d retries failed for %s", maxRetries, modelStr)
}

// executeSingleModel handles a single model request (non-combo path)
func executeSingleModel(w http.ResponseWriter, r *http.Request, body []byte, req ChatRequest, modelStr, effort string, registry *provider.Registry, acctMgr *account.Manager, db *database.DB) {
	startTime := time.Now()

	p, providerType, cleanModel, err := registry.ResolveProvider(modelStr)
	if err != nil {
		log.Printf("[CHAT] Provider resolve error: %v", err)
		writeError(w, "No active provider for model: "+req.Model, http.StatusServiceUnavailable)
		return
	}

	model := cleanModel
	log.Printf("[CHAT] Model: %s -> %s (provider: %s, effort: %s, stream: %v)",
		req.Model, model, providerType, effort, req.Stream)

	apiKeyID := GetApiKeyID(r)
	inputTokens := estimateInputTokens(body)

	var lastErr error
	lastStatus := http.StatusInternalServerError
	authRecoveryAttempted := false
	for attempt := 0; attempt < maxRetries; attempt++ {
		acct, err := acctMgr.Select(providerType, model)
		if err != nil {
			log.Printf("[CHAT] Account select error: %v", err)
			writeError(w, "No available account: "+err.Error(), http.StatusServiceUnavailable)
			return
		}

		provReq := &provider.Request{
			RawBody: json.RawMessage(body),
			Model:   model,
			Effort:  effort,
			Stream:  req.Stream,
			Account: acct,
		}

		events, err := p.Execute(r.Context(), provReq)
		if err != nil {
			log.Printf("[CHAT] Execute error (attempt %d): %v", attempt+1, err)
			status := providerErrorStatus(err)
			if status == http.StatusUnauthorized && !authRecoveryAttempted && acct != nil {
				authRecoveryAttempted = true
				if _, refreshErr := acctMgr.RefreshAfterAuthFailure(acct); refreshErr == nil {
					log.Printf("[CHAT] OAuth token refreshed after upstream 401; retrying %s", model)
					continue
				} else {
					log.Printf("[CHAT] OAuth recovery failed after upstream 401: %v", refreshErr)
				}
			}
			if acct != nil {
				acctMgr.ReportError(acct.ID, model, status, err.Error())
			}
			lastErr = err
			lastStatus = status
			if status >= 400 && status < 500 {
				break
			}
			continue
		}

		// Success — report and serve response
		if acct != nil {
			acctMgr.ReportSuccess(acct.ID, model)
		}

		accountID := ""
		if acct != nil {
			accountID = acct.ID
		}

		var tokenUsage requestTokenUsage
		var cost float64
		if req.Stream {
			tokenUsage, cost = streamResponse(w, events, model, req.Model)
		} else {
			tokenUsage, cost = nonStreamResponse(w, events, model, req.Model)
		}
		tokenUsage = tokenUsage.forLogging(inputTokens)

		// Log the request
		if db != nil {
			durationMs := time.Since(startTime).Milliseconds()
			if cost == 0 {
				inRate, outRate := database.ModelCostRates(model)
				cost = float64(tokenUsage.InputTokens)/1_000_000*inRate +
					float64(tokenUsage.OutputTokens)/1_000_000*outRate
			}
			db.LogRequest(
				apiKeyID, providerType, model, effort, accountID,
				tokenUsage.InputTokens, tokenUsage.OutputTokens,
				tokenUsage.CacheCreationInputTokens, tokenUsage.CacheReadInputTokens,
				cost, durationMs,
			)
		}
		return
	}

	// All attempts failed
	errMsg := "Provider error"
	if lastErr != nil {
		errMsg = lastErr.Error()
	}
	writeError(w, errMsg, lastStatus)
}

func providerErrorStatus(err error) int {
	var upstreamErr *provider.UpstreamError
	if errors.As(err, &upstreamErr) {
		return upstreamErr.StatusCode
	}
	return http.StatusInternalServerError
}

type requestTokenUsage struct {
	InputTokens              int
	OutputTokens             int
	CacheCreationInputTokens int
	CacheReadInputTokens     int
	Actual                   bool
}

func (u requestTokenUsage) forLogging(estimatedInputTokens int) requestTokenUsage {
	if !u.Actual {
		u.InputTokens = estimatedInputTokens
	}
	return u
}

func extractResponseUsage(jsonStr string) (requestTokenUsage, bool) {
	var payload struct {
		Usage *Usage `json:"usage"`
	}
	if json.Unmarshal([]byte(jsonStr), &payload) != nil || payload.Usage == nil {
		return requestTokenUsage{}, false
	}

	usage := requestTokenUsage{
		InputTokens:  payload.Usage.PromptTokens,
		OutputTokens: payload.Usage.CompletionTokens,
		Actual:       true,
	}
	if details := payload.Usage.PromptTokensDetails; details != nil {
		usage.CacheCreationInputTokens = details.CacheCreationTokens
		usage.CacheReadInputTokens = details.CachedTokens
	}
	return usage, true
}

// streamResponse streams SSE events and returns actual token usage when the
// provider reports it, falling back to output-text estimation otherwise.
func streamResponse(w http.ResponseWriter, events <-chan provider.Event, model string, originalModel string) (requestTokenUsage, float64) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return requestTokenUsage{}, 0
	}

	chatID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	sentRole := false
	sentFinish := false
	var totalText int
	var cost float64
	var tokenUsage requestTokenUsage

	for event := range events {
		switch event.Type {
		case "text":
			if !sentRole {
				sendSSE(w, flusher, ChatResponse{
					ID: chatID, Object: "chat.completion.chunk", Created: created, Model: originalModel,
					Choices: []Choice{{Index: 0, Delta: &Delta{Role: "assistant"}}},
				})
				sentRole = true
			}
			sendSSE(w, flusher, ChatResponse{
				ID: chatID, Object: "chat.completion.chunk", Created: created, Model: originalModel,
				Choices: []Choice{{Index: 0, Delta: &Delta{Content: event.Text}}},
			})
			totalText += len(event.Text)

		case "sse_chunk":
			fmt.Fprintf(w, "data: %s\n\n", event.JSON)
			flusher.Flush()
			sentRole = true
			sentFinish = sentFinish || chunkHasFinishReason(event.JSON)
			if usage, ok := extractResponseUsage(event.JSON); ok {
				tokenUsage = usage
			}
			// Try to extract token count from the chunk
			totalText += extractChunkTextLen(event.JSON)

		case "tool_use":
			// Internal tool use (CLI) — log only

		case "done":
			if event.Cost > 0 {
				cost = event.Cost
			}
			if !sentRole {
				sendSSE(w, flusher, ChatResponse{
					ID: chatID, Object: "chat.completion.chunk", Created: created, Model: originalModel,
					Choices: []Choice{{Index: 0, Delta: &Delta{Role: "assistant"}}},
				})
			}
			if !sentFinish {
				sendSSE(w, flusher, ChatResponse{
					ID: chatID, Object: "chat.completion.chunk", Created: created, Model: originalModel,
					Choices: []Choice{{Index: 0, Delta: &Delta{}, FinishReason: "stop"}},
				})
			}
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			if !tokenUsage.Actual {
				tokenUsage.OutputTokens = totalText / 4
			}
			return tokenUsage, cost

		case "error":
			log.Printf("[CHAT] Stream error: %s", event.Text)
			sendSSEError(w, flusher, event.Text)
			if !tokenUsage.Actual {
				tokenUsage.OutputTokens = totalText / 4
			}
			return tokenUsage, cost
		}
	}

	// Stream ended without done event
	if !sentRole {
		sendSSE(w, flusher, ChatResponse{
			ID: chatID, Object: "chat.completion.chunk", Created: created, Model: originalModel,
			Choices: []Choice{{Index: 0, Delta: &Delta{Role: "assistant", Content: "(no response)"}}},
		})
	}
	sendSSE(w, flusher, ChatResponse{
		ID: chatID, Object: "chat.completion.chunk", Created: created, Model: originalModel,
		Choices: []Choice{{Index: 0, Delta: &Delta{}, FinishReason: "stop"}},
	})
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
	if !tokenUsage.Actual {
		tokenUsage.OutputTokens = totalText / 4
	}
	return tokenUsage, cost
}

// nonStreamResponse collects all events and returns a complete response
func nonStreamResponse(w http.ResponseWriter, events <-chan provider.Event, model string, originalModel string) (requestTokenUsage, float64) {
	w.Header().Set("Content-Type", "application/json")

	var fullText strings.Builder
	var fullJSON string
	var cost float64

	for event := range events {
		switch event.Type {
		case "text":
			fullText.WriteString(event.Text)
		case "sse_chunk":
			fullJSON = event.JSON
		case "done":
			if event.Cost > 0 {
				cost = event.Cost
			}
		}
	}

	response := fullText.String()

	// If we received sse_chunk, try to extract content from last chunk
	if response == "" && fullJSON != "" {
		var chunk ChatResponse
		if json.Unmarshal([]byte(fullJSON), &chunk) == nil && len(chunk.Choices) > 0 {
			if chunk.Choices[0].Message != nil {
				w.Write([]byte(fullJSON))
				outputTokens := len(chunk.Choices[0].Message.Content) / 4
				tokenUsage, ok := extractResponseUsage(fullJSON)
				if !ok {
					tokenUsage.OutputTokens = outputTokens
				}
				return tokenUsage, cost
			}
		}
	}

	outputTokens := len(response) / 4

	resp := ChatResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   originalModel,
		Choices: []Choice{{
			Index:        0,
			Message:      &MsgOut{Role: "assistant", Content: response},
			FinishReason: "stop",
		}},
		Usage: &Usage{
			PromptTokens:     len(response) / 4,
			CompletionTokens: outputTokens,
			TotalTokens:      len(response)/4 + outputTokens,
		},
	}

	json.NewEncoder(w).Encode(resp)
	return requestTokenUsage{OutputTokens: outputTokens}, cost
}

func sendSSE(w http.ResponseWriter, flusher http.Flusher, chunk ChatResponse) {
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func sendSSEError(w http.ResponseWriter, flusher http.Flusher, message string) {
	payload := map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "upstream_error",
			"param":   nil,
			"code":    "upstream_stream_error",
		},
	}
	data, _ := json.Marshal(payload)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func writeError(w http.ResponseWriter, message string, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{
			"message": message,
			"type":    "error",
		},
	})
}

// parseModelAndEffort extracts an effort suffix from a model ID.
// E.g. "cc/claude-opus-4-6:low" -> ("cc/claude-opus-4-6", "low")
func parseModelAndEffort(m string) (string, string) {
	m = strings.TrimSpace(m)
	if m == "" {
		return "", ""
	}

	if idx := strings.LastIndex(m, ":"); idx > 0 {
		if effort := normalizeEffort(m[idx+1:]); effort != "" {
			return m[:idx], effort
		}
	}

	return m, ""
}

func normalizeEffort(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "minimal":
		// Anthropic has no "minimal" level. Low is its closest equivalent.
		return "low"
	case "low":
		return "low"
	case "medium", "med":
		return "medium"
	case "high":
		return "high"
	case "xhigh":
		return "xhigh"
	case "max":
		return "max"
	default:
		// "none", an omitted value, and unknown OpenAI-specific levels mean
		// that no Anthropic output_config.effort should be sent.
		return ""
	}
}

// estimateInputTokens estimates input tokens from request body size
func estimateInputTokens(body []byte) int {
	return len(body) / 4
}

// extractChunkTextLen extracts the text length from an SSE chunk JSON
func extractChunkTextLen(jsonStr string) int {
	var chunk struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal([]byte(jsonStr), &chunk) == nil && len(chunk.Choices) > 0 {
		return len(chunk.Choices[0].Delta.Content)
	}
	return 0
}

func chunkHasFinishReason(jsonStr string) bool {
	var chunk struct {
		Choices []struct {
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	if json.Unmarshal([]byte(jsonStr), &chunk) != nil {
		return false
	}
	for _, choice := range chunk.Choices {
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			return true
		}
	}
	return false
}
