package httpx

import "testing"

func TestSplitKubernetesTimestamp(t *testing.T) {
	message, timestamp := splitKubernetesTimestamp("2026-08-30T08:12:34.123456789Z worker ready")
	if message != "worker ready" {
		t.Fatalf("message = %q", message)
	}
	if timestamp != "2026-08-30T08:12:34.123456789Z" {
		t.Fatalf("timestamp = %#v", timestamp)
	}
}

func TestSplitKubernetesTimestampWithoutTimestamp(t *testing.T) {
	message, timestamp := splitKubernetesTimestamp("plain application log")
	if message != "plain application log" || timestamp != nil {
		t.Fatalf("got message=%q timestamp=%#v", message, timestamp)
	}
}

func TestDetectLogLevelDoesNotTreatDNSNoErrorAsError(t *testing.T) {
	if level := detectLogLevel(`[INFO] query completed NOERROR`); level != "info" {
		t.Fatalf("level = %q", level)
	}
	if level := detectLogLevel(`[ERROR] upstream unavailable`); level != "error" {
		t.Fatalf("level = %q", level)
	}
}
