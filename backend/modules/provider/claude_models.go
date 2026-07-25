package provider

// ClaudeOAuthModels is the authoritative catalog for direct Claude
// subscription models. Consumers such as the Zed settings synchronizer use
// this same list so editor pickers cannot drift from proxy routing.
func ClaudeOAuthModels() []Model {
	return []Model{
		{ID: "cc/claude-opus-5", Name: "Claude Opus 5", OwnedBy: "anthropic"},
		{ID: "cc/claude-sonnet-5", Name: "Claude Sonnet 5", OwnedBy: "anthropic"},
		{ID: "cc/claude-fable-5", Name: "Claude Fable 5", OwnedBy: "anthropic"},
	}
}
