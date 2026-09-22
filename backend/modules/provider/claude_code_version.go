package provider

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Anthropic gates new models on the Claude Code version in the User-Agent
// (Fable 5.1 and Opus 5.5 needed 2.1.280). Nothing here is pinned: the version
// the proxy presents is the newest of
//   - the latest @anthropic-ai/claude-code release on the npm registry,
//   - the `claude` CLI on PATH, and
//   - whatever a "version X or newer is required" rejection last told us,
// re-resolved every claudeCodeVersionTTL. A cold start with no network and no
// CLI presents 0.0.0; the first rejection then teaches the real requirement
// and the request is retried (see AnthropicAPI.Execute).

const claudeCodeNpmLatestURL = "https://registry.npmjs.org/@anthropic-ai/claude-code/latest"
const claudeCodeVersionTTL = 6 * time.Hour
const claudeCodeResolveTimeout = 5 * time.Second

// claudeCodeBuildSuffix is the fourth segment the CLI reports in its billing
// header. Anthropic does not gate on it, so a fixed suffix is enough.
const claudeCodeBuildSuffix = "0"

var (
	claudeCodeVersionPattern     = regexp.MustCompile(`(\d+\.\d+\.\d+)`)
	claudeCodeRequirementPattern = regexp.MustCompile(`version (\d+\.\d+\.\d+) or newer is required`)
)

type claudeCodeVersionState struct {
	mu         sync.Mutex
	version    string
	resolvedAt time.Time
}

var claudeCodeVersions claudeCodeVersionState

// claudeCodeVersion returns the Claude Code version to present upstream.
func claudeCodeVersion() string {
	return claudeCodeVersions.current()
}

// claudeCodeBillingVersion is the four-segment form used in the billing header.
func claudeCodeBillingVersion() string {
	return claudeCodeVersion() + "." + claudeCodeBuildSuffix
}

// learnClaudeCodeVersionRequirement reads the minimum version out of an
// upstream rejection and adopts it. It reports whether the presented version
// was raised, i.e. whether the caller should retry.
func learnClaudeCodeVersionRequirement(upstreamBody string) bool {
	match := claudeCodeRequirementPattern.FindStringSubmatch(upstreamBody)
	if match == nil {
		return false
	}
	return claudeCodeVersions.raise(match[1], "upstream requirement")
}

func (s *claudeCodeVersionState) current() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.version == "" || time.Since(s.resolvedAt) > claudeCodeVersionTTL {
		s.refreshLocked()
	}
	if s.version == "" {
		return "0.0.0"
	}
	return s.version
}

func (s *claudeCodeVersionState) raise(candidate, source string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if candidate == "" || compareDottedVersions(candidate, s.version) <= 0 {
		return false
	}
	log.Printf("[ANTHROPIC] Presenting Claude Code %s (from %s, was %q)", candidate, source, s.version)
	s.version = candidate
	return true
}

// refreshLocked re-resolves from every source, keeping the newest. It never
// lowers the version: a requirement learned from upstream survives a refresh
// where npm or the CLI are unreachable or stale.
func (s *claudeCodeVersionState) refreshLocked() {
	ctx, cancel := context.WithTimeout(context.Background(), claudeCodeResolveTimeout)
	defer cancel()
	for _, source := range []struct {
		name    string
		version string
	}{
		{"npm registry", fetchLatestClaudeCodeVersion(ctx, claudeCodeNpmLatestURL)},
		{"installed CLI", installedClaudeCodeVersion(ctx)},
	} {
		if source.version != "" && compareDottedVersions(source.version, s.version) > 0 {
			log.Printf("[ANTHROPIC] Presenting Claude Code %s (from %s, was %q)", source.version, source.name, s.version)
			s.version = source.version
		}
	}
	s.resolvedAt = time.Now()
}

// fetchLatestClaudeCodeVersion asks the npm registry for the latest published
// Claude Code release. Returns "" on any failure.
func fetchLatestClaudeCodeVersion(ctx context.Context, registryURL string) string {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, registryURL, nil)
	if err != nil {
		return ""
	}
	request.Header.Set("Accept", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		log.Printf("[ANTHROPIC] Could not read latest Claude Code version from npm: %v", err)
		return ""
	}
	defer response.Body.Close()
	if response.StatusCode >= 400 {
		log.Printf("[ANTHROPIC] npm registry returned %d for the Claude Code version lookup", response.StatusCode)
		return ""
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return ""
	}
	var manifest struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(body, &manifest); err != nil {
		return ""
	}
	return parseClaudeCodeVersion(manifest.Version)
}

// installedClaudeCodeVersion reads `claude --version` when a CLI is on PATH.
func installedClaudeCodeVersion(ctx context.Context) string {
	path, err := exec.LookPath("claude")
	if err != nil {
		return ""
	}
	output, err := exec.CommandContext(ctx, path, "--version").Output()
	if err != nil {
		log.Printf("[ANTHROPIC] Could not read installed Claude Code version: %v", err)
		return ""
	}
	return parseClaudeCodeVersion(string(output))
}

// parseClaudeCodeVersion extracts "2.1.280" from strings such as
// "2.1.280 (Claude Code)". Returns "" when no version is present.
func parseClaudeCodeVersion(text string) string {
	match := claudeCodeVersionPattern.FindStringSubmatch(text)
	if match == nil {
		return ""
	}
	return match[1]
}

// compareDottedVersions orders "2.1.280"-style versions numerically; a missing
// segment reads as 0 and an empty string is older than anything.
func compareDottedVersions(a, b string) int {
	left := strings.Split(a, ".")
	right := strings.Split(b, ".")
	length := len(left)
	if len(right) > length {
		length = len(right)
	}
	for index := 0; index < length; index++ {
		var l, r int
		if index < len(left) {
			l, _ = strconv.Atoi(left[index])
		}
		if index < len(right) {
			r, _ = strconv.Atoi(right[index])
		}
		if l != r {
			if l < r {
				return -1
			}
			return 1
		}
	}
	return 0
}
