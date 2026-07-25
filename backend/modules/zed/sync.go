package zed

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"code-proxy/modules/provider"
)

const providerName = "Code Proxy"

type Capabilities struct {
	Tools                bool `json:"tools"`
	Images               bool `json:"images"`
	ParallelToolCalls    bool `json:"parallel_tool_calls"`
	PromptCacheKey       bool `json:"prompt_cache_key"`
	ChatCompletions      bool `json:"chat_completions"`
	InterleavedReasoning bool `json:"interleaved_reasoning"`
	MaxTokensParameter   bool `json:"max_tokens_parameter"`
}

type Model struct {
	Name            string       `json:"name"`
	DisplayName     string       `json:"display_name"`
	MaxTokens       int          `json:"max_tokens"`
	MaxOutputTokens int          `json:"max_output_tokens"`
	ReasoningEffort string       `json:"reasoning_effort,omitempty"`
	Capabilities    Capabilities `json:"capabilities"`
}

// CodeProxyModels returns the native-tool-compatible models that Code Proxy
// exposes to Zed through the Claude subscription OAuth route.
func CodeProxyModels(catalog []provider.Model) []Model {
	capabilities := Capabilities{
		Tools:                true,
		Images:               false,
		ParallelToolCalls:    false,
		PromptCacheKey:       false,
		ChatCompletions:      true,
		InterleavedReasoning: false,
		MaxTokensParameter:   true,
	}
	models := make([]Model, 0, len(catalog))
	for _, catalogModel := range catalog {
		reasoningEffort := ""
		if supportsClaudeEffort(catalogModel.ID) {
			reasoningEffort = "high"
		}
		models = append(models, Model{
			Name:            catalogModel.ID,
			DisplayName:     catalogModel.Name,
			MaxTokens:       200000,
			MaxOutputTokens: 64000,
			ReasoningEffort: reasoningEffort,
			Capabilities:    capabilities,
		})
	}
	return models
}

func supportsClaudeEffort(modelID string) bool {
	modelID = strings.ToLower(strings.TrimPrefix(modelID, "cc/"))
	for _, family := range []string{
		"claude-fable-5",
		"claude-mythos-5",
		"claude-mythos-preview",
		"claude-opus-4-5",
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

// DefaultSettingsPath returns Zed's default settings path on macOS and Linux.
func DefaultSettingsPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".config", "zed", "settings.json"), nil
}

// SyncCodeProxyModels updates only Code Proxy's available_models array in
// Zed's JSON-with-comments settings file. It leaves all other text untouched.
func SyncCodeProxyModels(path string, models []Model) (bool, error) {
	source, err := os.ReadFile(path)
	if err != nil {
		return false, err
	}

	updated, changed, err := replaceModels(source, models)
	if err != nil || !changed {
		return changed, err
	}

	info, err := os.Stat(path)
	if err != nil {
		return false, err
	}
	if err := writeBackupOnce(path+".code-proxy.bak", source, info.Mode().Perm()); err != nil {
		return false, err
	}
	temp, err := os.CreateTemp(filepath.Dir(path), ".code-proxy-zed-settings-*")
	if err != nil {
		return false, err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)

	if err := temp.Chmod(info.Mode().Perm()); err != nil {
		temp.Close()
		return false, err
	}
	if _, err := temp.Write(updated); err != nil {
		temp.Close()
		return false, err
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return false, err
	}
	if err := temp.Close(); err != nil {
		return false, err
	}
	if err := os.Rename(tempPath, path); err != nil {
		return false, err
	}
	return true, nil
}

func writeBackupOnce(path string, source []byte, mode os.FileMode) error {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if errors.Is(err, os.ErrExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if _, err := file.Write(source); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func replaceModels(source []byte, models []Model) ([]byte, bool, error) {
	providerStart, providerEnd, err := findObjectForKey(source, providerName, 0, len(source))
	if err != nil {
		return nil, false, fmt.Errorf("find %q provider: %w", providerName, err)
	}
	arrayStart, arrayEnd, err := findArrayForKey(source, "available_models", providerStart, providerEnd)
	if err != nil {
		return nil, false, fmt.Errorf("find available_models: %w", err)
	}

	pretty, err := json.MarshalIndent(models, "", "  ")
	if err != nil {
		return nil, false, err
	}
	indent := lineIndent(source, arrayStart)
	pretty = bytes.ReplaceAll(pretty, []byte("\n"), []byte("\n"+indent))

	var current []Model
	if json.Unmarshal(source[arrayStart:arrayEnd], &current) == nil && reflect.DeepEqual(current, models) {
		return source, false, nil
	}

	result := make([]byte, 0, len(source)-arrayEnd+arrayStart+len(pretty))
	result = append(result, source[:arrayStart]...)
	result = append(result, pretty...)
	result = append(result, source[arrayEnd:]...)
	return result, true, nil
}

func findObjectForKey(source []byte, key string, start, end int) (int, int, error) {
	return findCompositeForKey(source, key, start, end, '{', '}')
}

func findArrayForKey(source []byte, key string, start, end int) (int, int, error) {
	return findCompositeForKey(source, key, start, end, '[', ']')
}

func findCompositeForKey(source []byte, key string, start, end int, open, close byte) (int, int, error) {
	for i := start; i < end; {
		i = skipSpaceAndComments(source, i, end)
		if i >= end {
			break
		}
		if source[i] != '"' {
			i++
			continue
		}

		tokenEnd, err := stringEnd(source, i, end)
		if err != nil {
			return 0, 0, err
		}
		var token string
		if err := json.Unmarshal(source[i:tokenEnd], &token); err != nil {
			return 0, 0, err
		}
		after := skipSpaceAndComments(source, tokenEnd, end)
		if token != key || after >= end || source[after] != ':' {
			i = tokenEnd
			continue
		}
		valueStart := skipSpaceAndComments(source, after+1, end)
		if valueStart >= end || source[valueStart] != open {
			return 0, 0, fmt.Errorf("%s is not a %q value", key, string(open))
		}
		valueEnd, err := compositeEnd(source, valueStart, end, open, close)
		if err != nil {
			return 0, 0, err
		}
		return valueStart, valueEnd, nil
	}
	return 0, 0, errors.New("key not found")
}

func compositeEnd(source []byte, start, end int, open, close byte) (int, error) {
	depth := 0
	for i := start; i < end; {
		switch source[i] {
		case '"':
			next, err := stringEnd(source, i, end)
			if err != nil {
				return 0, err
			}
			i = next
			continue
		case '/':
			next := skipComment(source, i, end)
			if next != i {
				i = next
				continue
			}
		case open:
			depth++
		case close:
			depth--
			if depth == 0 {
				return i + 1, nil
			}
		}
		i++
	}
	return 0, fmt.Errorf("unterminated %q value", string(open))
}

func stringEnd(source []byte, start, end int) (int, error) {
	for i := start + 1; i < end; i++ {
		if source[i] == '\\' {
			i++
			continue
		}
		if source[i] == '"' {
			return i + 1, nil
		}
	}
	return 0, errors.New("unterminated string")
}

func skipSpaceAndComments(source []byte, start, end int) int {
	for start < end {
		if strings.ContainsRune(" \t\r\n", rune(source[start])) {
			start++
			continue
		}
		next := skipComment(source, start, end)
		if next == start {
			return start
		}
		start = next
	}
	return start
}

func skipComment(source []byte, start, end int) int {
	if start+1 >= end || source[start] != '/' {
		return start
	}
	switch source[start+1] {
	case '/':
		for i := start + 2; i < end; i++ {
			if source[i] == '\n' {
				return i + 1
			}
		}
		return end
	case '*':
		for i := start + 2; i+1 < end; i++ {
			if source[i] == '*' && source[i+1] == '/' {
				return i + 2
			}
		}
		return end
	default:
		return start
	}
}

func lineIndent(source []byte, position int) string {
	lineStart := bytes.LastIndexByte(source[:position], '\n') + 1
	indentEnd := lineStart
	for indentEnd < position && (source[indentEnd] == ' ' || source[indentEnd] == '\t') {
		indentEnd++
	}
	return string(source[lineStart:indentEnd])
}
