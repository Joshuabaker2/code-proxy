package tunnel

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"strings"
	"testing"
)

func TestCloudflaredDownloadAsset(t *testing.T) {
	tests := []struct {
		goos        string
		goarch      string
		wantSuffix  string
		wantArchive bool
		wantOK      bool
	}{
		{"darwin", "arm64", "cloudflared-darwin-arm64.tgz", true, true},
		{"darwin", "amd64", "cloudflared-darwin-amd64.tgz", true, true},
		{"linux", "arm64", "cloudflared-linux-arm64", false, true},
		{"linux", "amd64", "cloudflared-linux-amd64", false, true},
		{"windows", "amd64", "cloudflared-windows-amd64.exe", false, true},
		{"darwin", "ppc64", "", false, false},
	}

	for _, test := range tests {
		asset, ok := cloudflaredDownloadAsset(test.goos, test.goarch)
		if ok != test.wantOK {
			t.Fatalf("%s/%s availability = %v, want %v", test.goos, test.goarch, ok, test.wantOK)
		}
		if !ok {
			continue
		}
		if !strings.HasSuffix(asset.url, test.wantSuffix) || asset.archive != test.wantArchive {
			t.Fatalf("%s/%s asset = %#v", test.goos, test.goarch, asset)
		}
	}
}

func TestCopyCloudflaredPayloadExtractsArchive(t *testing.T) {
	var archive bytes.Buffer
	gzipWriter := gzip.NewWriter(&archive)
	tarWriter := tar.NewWriter(gzipWriter)

	payload := []byte("fake cloudflared executable")
	if err := tarWriter.WriteHeader(&tar.Header{
		Name: "cloudflared",
		Mode: 0755,
		Size: int64(len(payload)),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := tarWriter.Write(payload); err != nil {
		t.Fatal(err)
	}
	if err := tarWriter.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatal(err)
	}

	var extracted bytes.Buffer
	if err := copyCloudflaredPayload(&extracted, &archive, true); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(extracted.Bytes(), payload) {
		t.Fatalf("extracted payload = %q", extracted.Bytes())
	}
}

func TestCopyCloudflaredPayloadRejectsArchiveWithoutBinary(t *testing.T) {
	var archive bytes.Buffer
	gzipWriter := gzip.NewWriter(&archive)
	tarWriter := tar.NewWriter(gzipWriter)
	if err := tarWriter.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatal(err)
	}

	if err := copyCloudflaredPayload(&bytes.Buffer{}, &archive, true); err == nil {
		t.Fatal("expected missing executable error")
	}
}
