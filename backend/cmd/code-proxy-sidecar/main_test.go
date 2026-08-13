package main

import (
	"path/filepath"
	"testing"
	"time"
)

func TestSidecarLockAllowsOnlyOneDataDirectoryOwner(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".sidecar.lock")
	first, err := acquireSidecarLock(path, 100*time.Millisecond)
	if err != nil {
		t.Fatalf("acquire first lock: %v", err)
	}

	if second, err := acquireSidecarLock(path, 100*time.Millisecond); err == nil {
		second.close()
		t.Fatal("second sidecar acquired the same data directory lock")
	}
	first.close()

	replacement, err := acquireSidecarLock(path, 100*time.Millisecond)
	if err != nil {
		t.Fatalf("replacement sidecar could not acquire released lock: %v", err)
	}
	replacement.close()
}
