package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"code-proxy/modules/account"
	"code-proxy/modules/database"
	"code-proxy/modules/provider"
)

const maxRetries = 3

// maxRetryWait caps how long a single request will block waiting for a
// cooldown or backoff to elapse. Anything longer is handed back to the client
// as a 503 with Retry-After instead of holding the connection open.
const maxRetryWait = 15 * time.Second

// upstreamRetryAfter returns the Retry-After hint carried by an upstream
// error, or 0 when the response did not send one.
func upstreamRetryAfter(err error) time.Duration {
	var upstreamErr *provider.UpstreamError
	if errors.As(err, &upstreamErr) {
		return upstreamErr.RetryAfter
	}
	return 0
}

// retryBackoff is how long to wait before the next attempt. Retrying with no
// delay at all cannot outlast even the shortest cooldown, which is how a
// transient upstream blip used to turn into "requires a configured account".
func retryBackoff(attempt int, hint time.Duration) time.Duration {
	if hint > 0 {
		return hint
	}
	return time.Duration(1<<uint(attempt)) * time.Second
}

// waitBeforeRetry blocks for wait. It reports false when the wait is longer
// than the budget or the client went away, meaning the caller should give up
// rather than retry.
func waitBeforeRetry(ctx context.Context, wait time.Duration) bool {
	if wait <= 0 {
		return true
	}
	if wait > maxRetryWait {
		return false
	}
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// writeRetryAfter advertises when the client may try again.
func writeRetryAfter(w http.ResponseWriter, wait time.Duration) {
	if wait <= 0 {
		return
	}
	seconds := int(math.Ceil(wait.Seconds()))
	if seconds < 1 {
		seconds = 1
	}
	w.Header().Set("Retry-After", strconv.Itoa(seconds))
}

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
			// All accounts cooling down: wait it out if it is short, otherwise
			// let the combo move on to the next model.
			var coolingDown *account.CoolingDownError
			if errors.As(err, &coolingDown) {
				if attempt < maxRetries-1 && waitBeforeRetry(r.Context(), coolingDown.RetryAfter()) {
					continue
				}
			}
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
			hint := upstreamRetryAfter(err)
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
				acctMgr.ReportError(acct.ID, cleanModel, status, hint, err.Error())
			}
			if status >= 400 && status < 500 {
				return err
			}
			if attempt < maxRetries-1 && !waitBeforeRetry(r.Context(), retryBackoff(attempt, hint)) {
				return err
			}
			continue
		}

		// Headers are on the wire now; the response still has to survive the
		// stream before it counts as a success.
		accountID := ""
		if acct != nil {
			accountID = acct.ID
		}

		var tokenUsage requestTokenUsage
		var cost float64
		var outcome streamOutcome
		if req.Stream {
			tokenUsage, cost, outcome = streamResponse(w, events, cleanModel, req.Model)
		} else {
			tokenUsage, cost, outcome = nonStreamResponse(w, events, cleanModel, req.Model)
		}
		tokenUsage = tokenUsage.forLogging(inputTokens)

		if acct != nil {
			if outcome.failed {
				acctMgr.ReportError(acct.ID, cleanModel, http.StatusBadGateway, 0, outcome.reason)
			} else {
				acctMgr.ReportSuccess(acct.ID, cleanModel)
			}
		}

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
	var lastRetryAfter time.Duration
	authRecoveryAttempted := false
	for attempt := 0; attempt < maxRetries; attempt++ {
		acct, err := acctMgr.Select(providerType, model)
		if err != nil {
			// Every account is briefly cooling down. This is NOT "no account
			// configured" — never let it reach the provider as a nil account,
			// which would report a bogus authentication failure.
			var coolingDown *account.CoolingDownError
			if errors.As(err, &coolingDown) {
				wait := coolingDown.RetryAfter()
				if attempt < maxRetries-1 && waitBeforeRetry(r.Context(), wait) {
					continue
				}
				// Out of budget. Report the failure that caused the cooldown
				// if we have it, rather than the cooldown itself.
				if lastErr == nil {
					lastErr = err
					lastStatus = http.StatusServiceUnavailable
				}
				if lastRetryAfter <= 0 {
					lastRetryAfter = wait
				}
				break
			}
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
			hint := upstreamRetryAfter(err)
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
				acctMgr.ReportError(acct.ID, model, status, hint, err.Error())
			}
			// Keep the real failure so the client is told what actually went
			// wrong, not whatever a later attempt happened to produce.
			lastErr = err
			lastStatus = status
			lastRetryAfter = hint
			if status >= 400 && status < 500 {
				break
			}
			if attempt < maxRetries-1 && !waitBeforeRetry(r.Context(), retryBackoff(attempt, hint)) {
				break
			}
			continue
		}

		// Headers are on the wire now; the response still has to survive the
		// stream before it counts as a success.
		accountID := ""
		if acct != nil {
			accountID = acct.ID
		}

		var tokenUsage requestTokenUsage
		var cost float64
		var outcome streamOutcome
		if req.Stream {
			tokenUsage, cost, outcome = streamResponse(w, events, model, req.Model)
		} else {
			tokenUsage, cost, outcome = nonStreamResponse(w, events, model, req.Model)
		}
		tokenUsage = tokenUsage.forLogging(inputTokens)

		if acct != nil {
			if outcome.failed {
				// Too late to change the HTTP status on a stream, but this must
				// not clear the account's backoff or count as a clean request.
				acctMgr.ReportError(acct.ID, model, http.StatusBadGateway, 0, outcome.reason)
			} else {
				acctMgr.ReportSuccess(acct.ID, model)
			}
		}

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
	writeRetryAfter(w, lastRetryAfter)
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
// streamOutcome reports whether a response actually completed. A stream that
// ended in an upstream error, or simply stopped mid-message, must not be
// treated as a success: that clears the account's backoff, logs the request as
// good, and — worst of all — hands the caller a truncated answer wearing a
// clean finish_reason.
type streamOutcome struct {
	failed bool
	reason string
}

func streamResponse(w http.ResponseWriter, events <-chan provider.Event, model string, originalModel string) (requestTokenUsage, float64, streamOutcome) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return requestTokenUsage{}, 0, streamOutcome{failed: true, reason: "streaming not supported"}
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
			return tokenUsage, cost, streamOutcome{}

		case "error":
			log.Printf("[CHAT] Stream error: %s", event.Text)
			sendSSEError(w, flusher, event.Text)
			if !tokenUsage.Actual {
				tokenUsage.OutputTokens = totalText / 4
			}
			return tokenUsage, cost, streamOutcome{failed: true, reason: event.Text}
		}
	}

	// The event channel closed without a done event.
	if !tokenUsage.Actual {
		tokenUsage.OutputTokens = totalText / 4
	}

	// If the upstream already signalled completion in-band, this is a clean
	// end and there is nothing more to say.
	if sentFinish {
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return tokenUsage, cost, streamOutcome{}
	}

	// Otherwise the connection ended mid-message. Synthesizing finish_reason
	// "stop" here would dress a truncated answer up as a complete one, which
	// on a long tool-calling run is silently wrong rather than loudly broken.
	reason := "upstream stream ended before the message was complete"
	if !sentRole {
		reason = "upstream stream closed without returning any content"
	}
	log.Printf("[CHAT] Truncated stream for %s: %s (%d bytes of text)", model, reason, totalText)
	sendSSEError(w, flusher, reason)
	return tokenUsage, cost, streamOutcome{failed: true, reason: reason}
}

// nonStreamResponse collects all events and returns a complete response
func nonStreamResponse(w http.ResponseWriter, events <-chan provider.Event, model string, originalModel string) (requestTokenUsage, float64, streamOutcome) {
	w.Header().Set("Content-Type", "application/json")

	var fullText strings.Builder
	var fullJSON string
	var cost float64
	var streamErr string

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
		case "error":
			// Previously dropped on the floor, so an upstream failure came back
			// as a 200 with whatever partial text had arrived.
			streamErr = event.Text
		}
	}

	// Nothing has been written yet on this path, so unlike the streaming case
	// we can still answer with an honest status.
	if streamErr != "" {
		log.Printf("[CHAT] Stream error (non-streaming) for %s: %s", model, streamErr)
		writeError(w, streamErr, http.StatusBadGateway)
		return requestTokenUsage{}, cost, streamOutcome{failed: true, reason: streamErr}
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
				return tokenUsage, cost, streamOutcome{}
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
	return requestTokenUsage{OutputTokens: outputTokens}, cost, streamOutcome{}
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
