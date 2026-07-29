package provider

import "strings"

// ClaudeOAuthModels is the authoritative catalog for direct Claude
// subscription models. Consumers such as the Zed settings synchronizer use
// this same list so editor pickers cannot drift from proxy routing.
func ClaudeOAuthModels() []Model {
	return []Model{
		claudeOAuthModel("claude-opus-5", "Claude Opus 5"),
		claudeOAuthModel("claude-sonnet-5", "Claude Sonnet 5"),
		claudeOAuthModel("claude-fable-5", "Claude Fable 5"),
	}
}

func claudeOAuthModel(id, name string) Model {
	maxInputTokens, maxOutputTokens := ClaudeModelLimits(id)
	return Model{
		ID:              "cc/" + id,
		Name:            name,
		OwnedBy:         "anthropic",
		MaxInputTokens:  maxInputTokens,
		MaxOutputTokens: maxOutputTokens,
	}
}

// ClaudeSupportsAdaptiveThinking reports whether the model accepts
// thinking.type=adaptive. Older 4.5 models require the legacy fixed-budget
// thinking mode instead.
func ClaudeSupportsAdaptiveThinking(modelID string) bool {
	modelID = strings.ToLower(strings.TrimPrefix(modelID, "cc/"))
	for _, family := range []string{
		"claude-fable-5",
		"claude-mythos-5",
		"claude-mythos-preview",
		"claude-opus-4-6",
		"claude-opus-4-7",
		"claude-opus-4-8",
		"claude-opus-5",
		"claude-sonnet-4-6",
		"claude-sonnet-5",
	} {
		if strings.HasPrefix(modelID, family) {
			return true
		}
	}
	return false
}

// ClaudeModelLimits provides conservative fallbacks for older Models API
// responses. Live discovery values take precedence whenever Anthropic supplies
// max_input_tokens and max_tokens.
func ClaudeModelLimits(modelID string) (maxInputTokens, maxOutputTokens int) {
	modelID = strings.ToLower(strings.TrimPrefix(modelID, "cc/"))
	for _, family := range []string{
		"claude-fable-5",
		"claude-mythos-5",
		"claude-mythos-preview",
		"claude-opus-4-6",
		"claude-opus-4-7",
		"claude-opus-4-8",
		"claude-opus-5",
		"claude-sonnet-4-6",
		"claude-sonnet-5",
	} {
		if strings.HasPrefix(modelID, family) {
			return 1_000_000, 128_000
		}
	}
	return 200_000, 64_000
}
