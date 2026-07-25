package zed

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"code-proxy/modules/provider"
)

func TestReplaceModelsPreservesJSONCAndOtherSettings(t *testing.T) {
	source := []byte(`// user comment
{
  "language_models": {
    "openai_compatible": {
      "Code Proxy": {
        "api_url": "http://127.0.0.1:3456/v1",
        // managed by Code Proxy
        "available_models": [
          {"name": "old"}
        ],
      },
    },
  },
  "theme": "Keep Me",
}
`)

	updated, changed, err := replaceModels(source, CodeProxyModels(provider.ClaudeOAuthModels()))
	if err != nil {
		t.Fatal(err)
	}
	if !changed {
		t.Fatal("expected model list to change")
	}
	text := string(updated)
	for _, expected := range []string{
		"// user comment",
		"// managed by Code Proxy",
		`"theme": "Keep Me"`,
		`"name": "cc/claude-opus-5"`,
		`"reasoning_effort": "high"`,
		`"max_tokens_parameter": true`,
	} {
		if !strings.Contains(text, expected) {
			t.Fatalf("updated settings missing %q:\n%s", expected, text)
		}
	}
}

func TestSyncCodeProxyModelsIsAtomicAndIdempotent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "settings.json")
	source := `{"language_models":{"openai_compatible":{"Code Proxy":{"available_models":[]}}}}`
	if err := os.WriteFile(path, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}

	models := CodeProxyModels(provider.ClaudeOAuthModels())
	changed, err := SyncCodeProxyModels(path, models)
	if err != nil || !changed {
		t.Fatalf("first sync: changed=%v err=%v", changed, err)
	}
	changed, err = SyncCodeProxyModels(path, models)
	if err != nil || changed {
		t.Fatalf("second sync: changed=%v err=%v", changed, err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0600 {
		t.Fatalf("settings mode changed to %o", info.Mode().Perm())
	}
	backup, err := os.ReadFile(path + ".code-proxy.bak")
	if err != nil {
		t.Fatal(err)
	}
	if string(backup) != source {
		t.Fatalf("backup did not preserve original settings: %s", backup)
	}
}

func TestReplaceModelsDoesNotTouchOtherProvider(t *testing.T) {
	source := []byte(`{"Code Proxy Backup":{"available_models":[]}}`)
	_, _, err := replaceModels(source, CodeProxyModels(provider.ClaudeOAuthModels()))
	if err == nil {
		t.Fatal("expected missing Code Proxy provider error")
	}
}

func TestCodeProxyModelsFollowProviderCatalog(t *testing.T) {
	catalog := []provider.Model{
		{ID: "cc/claude-opus-5", Name: "Claude Opus 5"},
		{ID: "cc/claude-haiku-4-5", Name: "Claude Haiku 4.5"},
	}
	models := CodeProxyModels(catalog)
	if len(models) != 2 || models[0].Name != "cc/claude-opus-5" || models[0].DisplayName != "Claude Opus 5" {
		t.Fatalf("Zed models drifted from provider catalog: %#v", models)
	}
	if models[0].ReasoningEffort != "high" {
		t.Fatalf("effort-capable model was not advertised to Zed: %#v", models[0])
	}
	if models[1].ReasoningEffort != "" {
		t.Fatalf("unsupported model advertised effort controls: %#v", models[1])
	}
}
