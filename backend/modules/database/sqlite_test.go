package database

import (
	"path/filepath"
	"testing"
)

func TestRequestLogsPersistCacheUsage(t *testing.T) {
	db, err := Open(filepath.Join(t.TempDir(), "data.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	db.LogRequest(
		"key-id", "anthropic-api", "claude-opus-5", "high", "account-id",
		155, 20, 25, 120, 0, 1234,
	)

	logs, total, err := db.ListLogs(10, 0)
	if err != nil {
		t.Fatal(err)
	}
	if total != 1 || len(logs) != 1 {
		t.Fatalf("unexpected logs: total=%d logs=%#v", total, logs)
	}
	got := logs[0]
	if got.InputTokens != 155 ||
		got.OutputTokens != 20 ||
		got.CacheCreationInputTokens != 25 ||
		got.CacheReadInputTokens != 120 {
		t.Fatalf("cache usage was not persisted: %#v", got)
	}
}
