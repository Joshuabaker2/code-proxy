package provider

import (
	"strconv"
	"strings"
)

// ClaudeOAuthModels is the offline catalog for direct Claude subscription
// models: what the embedded sidecar and the Zed synchronizer offer before (or
// without) a successful live discovery against Anthropic's models endpoint.
// Live discovery extends this list; consumers pick the newest release per
// family from the merged catalog (see ClaudeModelRelease), so a new Claude
// drop reaches their pickers without an entry here. Bump these only so the
// catalog is right before the first discovery.
func ClaudeOAuthModels() []Model {
	return []Model{
		claudeOAuthModel("claude-fable-5-1", "Claude Fable 5.1"),
		claudeOAuthModel("claude-opus-5-5", "Claude Opus 5.5"),
		claudeOAuthModel("claude-sonnet-5", "Claude Sonnet 5"),
		claudeOAuthModel("claude-haiku-4-5", "Claude Haiku 4.5"),
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

// ClaudeModelRelease is a Claude model ID broken into the parts routing cares
// about: `claude-<family>-<major>[-<minor>][-<yyyymmdd>]`.
type ClaudeModelRelease struct {
	// ID is the bare model ID with any provider prefix or `:effort` suffix removed.
	ID string
	// Family is the alphabetic run after `claude-`, e.g. "opus" or "mythos-preview".
	Family string
	// Version holds the numeric segments, e.g. claude-opus-5-5 -> [5, 5]. Empty
	// for unversioned IDs such as claude-mythos-preview.
	Version []int
	// Dated marks a pinned snapshot (claude-haiku-4-5-20251001); the undated
	// alias wins ties against it.
	Dated bool
}

// ParseClaudeModelID understands current-generation Claude IDs, optionally
// carrying a `cc/`, `cli-cc/` or `anthropic/` prefix or an `:effort` suffix.
// Legacy IDs with the version before the family (claude-3-5-sonnet-…) and
// non-Claude models report ok=false.
func ParseClaudeModelID(rawID string) (release ClaudeModelRelease, ok bool) {
	id := strings.ToLower(rawID)
	for _, prefix := range []string{"cli-cc/", "cc/", "anthropic/"} {
		id = strings.TrimPrefix(id, prefix)
	}
	if colon := strings.IndexByte(id, ':'); colon >= 0 {
		id = id[:colon]
	}
	parts := strings.Split(id, "-")
	if len(parts) < 2 || parts[0] != "claude" {
		return ClaudeModelRelease{}, false
	}

	release = ClaudeModelRelease{ID: id}
	index := 1
	var family []string
	for ; index < len(parts); index++ {
		if isDigits(parts[index]) {
			break
		}
		if parts[index] == "" {
			return ClaudeModelRelease{}, false
		}
		family = append(family, parts[index])
	}
	if len(family) == 0 {
		return ClaudeModelRelease{}, false
	}
	release.Family = strings.Join(family, "-")

	for ; index < len(parts); index++ {
		part := parts[index]
		if !isDigits(part) {
			return ClaudeModelRelease{}, false
		}
		// Release dates are the only segments this wide; a version never is.
		if len(part) >= 8 {
			release.Dated = true
			if index != len(parts)-1 {
				return ClaudeModelRelease{}, false
			}
			break
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			return ClaudeModelRelease{}, false
		}
		release.Version = append(release.Version, value)
	}
	return release, true
}

func isDigits(value string) bool {
	if value == "" {
		return false
	}
	for _, r := range value {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// CompareClaudeModelVersions orders version segments numerically, reading a
// missing segment as 0 so that [5] == [5, 0] < [5, 1].
func CompareClaudeModelVersions(a, b []int) int {
	length := len(a)
	if len(b) > length {
		length = len(b)
	}
	for index := 0; index < length; index++ {
		var left, right int
		if index < len(a) {
			left = a[index]
		}
		if index < len(b) {
			right = b[index]
		}
		if left != right {
			if left < right {
				return -1
			}
			return 1
		}
	}
	return 0
}

// Newer reports whether r should replace incumbent as the family's current
// release: a higher version wins, and an undated alias beats a dated snapshot
// of the same version.
func (r ClaudeModelRelease) Newer(incumbent ClaudeModelRelease) bool {
	if by := CompareClaudeModelVersions(r.Version, incumbent.Version); by != 0 {
		return by > 0
	}
	return incumbent.Dated && !r.Dated
}

func (r ClaudeModelRelease) major() int {
	if len(r.Version) == 0 {
		return 0
	}
	return r.Version[0]
}

func (r ClaudeModelRelease) minor() int {
	if len(r.Version) < 2 {
		return 0
	}
	return r.Version[1]
}

// SupportsAdaptiveThinking reports whether the release accepts
// thinking.type=adaptive. Fable and Mythos always do; the Opus/Sonnet/Haiku
// lines gained it at 4.6, and every 5.x release has it. Haiku 4.5 and Opus 4.5
// still require the legacy fixed-budget thinking mode.
func (r ClaudeModelRelease) SupportsAdaptiveThinking() bool {
	switch {
	case strings.HasPrefix(r.Family, "fable"), strings.HasPrefix(r.Family, "mythos"):
		return true
	case r.major() >= 5:
		return true
	case r.major() == 4:
		return r.minor() >= 6
	}
	return false
}

// SupportsEffort reports whether output_config.effort is accepted. Everything
// with adaptive thinking has it, plus Opus 4.5 (low/medium/high only).
func (r ClaudeModelRelease) SupportsEffort() bool {
	if r.SupportsAdaptiveThinking() {
		return true
	}
	return r.Family == "opus" && r.major() == 4 && r.minor() == 5
}

// ClaudeSupportsAdaptiveThinking reports whether the model accepts
// thinking.type=adaptive. Older 4.5 models require the legacy fixed-budget
// thinking mode instead.
func ClaudeSupportsAdaptiveThinking(modelID string) bool {
	release, ok := ParseClaudeModelID(modelID)
	return ok && release.SupportsAdaptiveThinking()
}

// ClaudeSupportsEffort reports whether output_config.effort is accepted.
func ClaudeSupportsEffort(modelID string) bool {
	release, ok := ParseClaudeModelID(modelID)
	return ok && release.SupportsEffort()
}

// ClaudeModelLimits provides conservative fallbacks for older Models API
// responses. Live discovery values take precedence whenever Anthropic supplies
// max_input_tokens and max_tokens. The 1M/128K window arrived with the same
// releases as adaptive thinking.
func ClaudeModelLimits(modelID string) (maxInputTokens, maxOutputTokens int) {
	if ClaudeSupportsAdaptiveThinking(modelID) {
		return 1_000_000, 128_000
	}
	return 200_000, 64_000
}

// NewestClaudeModelID returns the bare ID of the newest release of family in
// catalog, so short names like "opus" follow the catalog instead of a pinned
// release.
func NewestClaudeModelID(family string, catalog []Model) (string, bool) {
	var newest ClaudeModelRelease
	found := false
	for _, model := range catalog {
		release, ok := ParseClaudeModelID(model.ID)
		if !ok || release.Family != family {
			continue
		}
		if !found || release.Newer(newest) {
			newest = release
			found = true
		}
	}
	return newest.ID, found
}
