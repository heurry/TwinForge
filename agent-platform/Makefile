.PHONY: fmt vet test race build check

fmt:
	gofmt -w ./cmd ./internal ./pkg

vet:
	GOWORK=off go vet ./...

test:
	GOWORK=off go test ./...

race:
	GOWORK=off go test -race ./...

build:
	GOWORK=off go build ./cmd/...

check: fmt vet test build
