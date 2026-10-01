package workflow

import "testing"

func TestWorkflowTransitionsAllowApprovalResume(t *testing.T) {
	for _, transition := range [][2]Status{
		{StatusRunning, StatusWaitingApproval},
		{StatusWaitingApproval, StatusReady},
		{StatusReady, StatusRunning},
	} {
		if err := ValidateTransition(transition[0], transition[1]); err != nil {
			t.Fatalf("transition %s -> %s: %v", transition[0], transition[1], err)
		}
	}
}

func TestWorkflowTransitionsRejectTerminalResume(t *testing.T) {
	if err := ValidateTransition(StatusSucceeded, StatusRunning); err == nil {
		t.Fatal("expected terminal workflow to reject resume")
	}
}

func TestWorkflowTransitionsCoverReflectionAndReview(t *testing.T) {
	for _, transition := range [][2]Status{
		{StatusRunning, StatusVerifying},
		{StatusVerifying, StatusReflecting},
		{StatusReflecting, StatusReplanning},
		{StatusReplanning, StatusRunning},
		{StatusRunning, StatusReviewing},
		{StatusReviewing, StatusSucceeded},
	} {
		if err := ValidateTransition(transition[0], transition[1]); err != nil {
			t.Fatalf("transition %s -> %s: %v", transition[0], transition[1], err)
		}
	}
}
