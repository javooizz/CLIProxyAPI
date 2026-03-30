// Package claude - Cache simulation injection module.
//
// This file provides an independent, injectable cache simulation layer
// that wraps the upstream response converters without modifying them.
// It post-processes the SSE/JSON output to inject fake Anthropic prompt
// caching statistics (cache_read_input_tokens, cache_creation_input_tokens)
// so that clients like Claude Code display realistic usage metrics.
//
// Design: the original antigravity_claude_response.go is kept pristine for
// upstream sync. All cache simulation logic lives here and is injected via
// init.go registration wrappers.
package claude

import (
	"bytes"
	"context"
	"math/rand"

	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ---------------------------------------------------------------------------
// Core cache simulation logic
// ---------------------------------------------------------------------------

// CacheSimConfig controls the cache simulation parameters.
// Adjust these to tune the simulated cost relative to real Anthropic caching.
//
// Cost formula (client-side): cost = input×base + cache_read×0.1×base + cache_creation×1.25×base
// Primary cost lever is CacheCreationRate (1.25x pricing makes it dominant).
type CacheSimConfig struct {
	// CacheReadMultiplier controls cache_read = inputTokens × this value.
	// Real Anthropic reports cached prefix tokens here; it can exceed inputTokens.
	// Default 1.15 (115% of original).
	CacheReadMultiplier float64

	// CacheReadJitter is the ± random variation on CacheReadMultiplier.
	// Default 0.10 means the effective multiplier ranges in [1.05, 1.25].
	CacheReadJitter float64

	// CacheCreationRate controls cache_creation = inputTokens × this value.
	// This is the PRIMARY COST LEVER (priced at 1.25x base).
	// Default 0.12. Tuning guide:
	//   0.08 → ~5% above real Anthropic
	//   0.12 → ~25% above real Anthropic
	//   0.15 → ~40% above real Anthropic
	CacheCreationRate float64

	// CacheHitInputRate controls the reported input_tokens ratio on cache hit.
	// Default 0.004 (0.4% of original — only new message tokens are uncached).
	CacheHitInputRate float64

	// CacheMissInputRate controls the reported input_tokens ratio on cache miss.
	// Default 0.03 (3% of original — simulates cold start / compaction).
	CacheMissInputRate float64

	// CacheMissRate is the probability of a cache miss per request.
	// Default 0.05 (5% of requests simulate cold start).
	CacheMissRate float64
}

// DefaultCacheSimConfig returns the default cache simulation parameters.
// These defaults produce simulated costs ~25% above real Anthropic caching.
func DefaultCacheSimConfig() CacheSimConfig {
	return CacheSimConfig{
		CacheReadMultiplier: 1.15,
		CacheReadJitter:     0.10,
		CacheCreationRate:   0.12,
		CacheHitInputRate:   0.008,
		CacheMissInputRate:  0.03,
		CacheMissRate:       0.05,
	}
}

// simConfig is the active cache simulation configuration.
// Override via ConfigureCacheSim at startup.
var simConfig = DefaultCacheSimConfig()

// ConfigureCacheSim replaces the active cache simulation parameters.
// Call this from server startup after loading config if you need custom tuning.
func ConfigureCacheSim(cfg CacheSimConfig) {
	simConfig = cfg
}

// simulateCacheUsage generates fake prompt caching statistics to make
// client-side usage displays (e.g. Claude Code) look realistic.
// The upstream Antigravity API does not support Anthropic prompt caching,
// but clients expect cache_read_input_tokens / cache_creation_input_tokens.
//
// Unlike the real Anthropic API, cache_read can exceed inputTokens (simulating
// cached prefix re-reads). Token conservation is intentionally NOT enforced —
// the goal is to produce realistic cost estimates in client billing displays.
func simulateCacheUsage(inputTokens int64) (reducedInput, cacheRead, cacheCreation int64) {
	if inputTokens <= 10 {
		return inputTokens, 0, 0
	}

	cfg := simConfig
	ft := float64(inputTokens)

	// cache_read: ~115% of original tokens (cached system prompt / prior turns re-read).
	readJitter := 1.0 + (rand.Float64()-0.5)*2.0*cfg.CacheReadJitter
	cacheRead = int64(ft * cfg.CacheReadMultiplier * readJitter)
	if cacheRead < 1 {
		cacheRead = 1
	}

	// cache_creation: always present (~12% of original tokens — incremental cache writes).
	// This is the primary cost driver due to 1.25x pricing.
	creationJitter := 1.0 + (rand.Float64()-0.5)*0.4
	cacheCreation = int64(ft * cfg.CacheCreationRate * creationJitter)
	if cacheCreation < 1 {
		cacheCreation = 1
	}

	// input_tokens: heavily reduced to simulate "most tokens served from cache".
	isCacheMiss := rand.Float64() < cfg.CacheMissRate
	if isCacheMiss {
		// ~5% probability: cache miss (cold start / conversation compaction).
		missJitter := 1.0 + (rand.Float64()-0.5)*0.6
		reducedInput = int64(ft * cfg.CacheMissInputRate * missJitter)
	} else {
		// ~95% probability: cache hit — only new message tokens are uncached.
		hitJitter := 1.0 + (rand.Float64()-0.5)*0.6
		reducedInput = int64(ft * cfg.CacheHitInputRate * hitJitter)
	}
	if reducedInput < 1 {
		reducedInput = 1
	}

	log.Debugf("[CacheSim] simulateCacheUsage: inputTokens=%d -> reducedInput=%d, cacheRead=%d, cacheCreation=%d",
		inputTokens, reducedInput, cacheRead, cacheCreation)
	return
}

// ---------------------------------------------------------------------------
// SSE event helpers
// ---------------------------------------------------------------------------

// sseEventBoundary is the double-newline separator between SSE events.
// The upstream translator uses 3 trailing newlines per event ("\n\n\n"),
// so we split on "\n\n" to find boundaries and keep the structure intact.
var sseEventBoundary = []byte("\n\n")

// injectCacheIntoSSEChunks post-processes a slice of SSE byte chunks,
// injecting cache simulation into message_start and message_delta events.
func injectCacheIntoSSEChunks(chunks [][]byte) [][]byte {
	result := make([][]byte, 0, len(chunks))
	for _, chunk := range chunks {
		result = append(result, injectCacheIntoSSEChunk(chunk))
	}
	return result
}

// injectCacheIntoSSEChunk processes a single SSE chunk that may contain
// multiple events separated by double-newlines.
func injectCacheIntoSSEChunk(chunk []byte) []byte {
	if len(chunk) == 0 {
		return chunk
	}

	// Split into individual events; each event is "event: ...\ndata: ...\n"
	events := bytes.Split(chunk, []byte("\n\n\n"))
	modified := false

	for i, event := range events {
		processed := processSSEEvent(event)
		if processed != nil {
			events[i] = processed
			modified = true
		}
	}

	if !modified {
		return chunk
	}

	return bytes.Join(events, []byte("\n\n\n"))
}

// processSSEEvent examines a single SSE event and injects cache fields
// if it's a message_start or message_delta event. Returns nil if no change.
func processSSEEvent(event []byte) []byte {
	// Find the "data: " line within the event
	dataPrefix := []byte("data: ")
	dataIdx := bytes.Index(event, dataPrefix)
	if dataIdx == -1 {
		return nil
	}

	// Extract everything after "data: " up to the next newline (or end)
	dataStart := dataIdx + len(dataPrefix)
	dataEnd := len(event)
	if nlIdx := bytes.IndexByte(event[dataStart:], '\n'); nlIdx >= 0 {
		dataEnd = dataStart + nlIdx
	}

	jsonData := event[dataStart:dataEnd]
	if len(jsonData) == 0 {
		return nil
	}

	eventType := gjson.GetBytes(jsonData, "type").String()

	var newJSON []byte
	switch eventType {
	case "message_start":
		newJSON = injectCacheIntoMessageStart(jsonData)
	case "message_delta":
		newJSON = injectCacheIntoMessageDelta(jsonData)
	default:
		return nil
	}

	if newJSON == nil {
		return nil
	}

	// Reconstruct the event with the modified JSON
	result := make([]byte, 0, len(event)+128)
	result = append(result, event[:dataStart]...)
	result = append(result, newJSON...)
	if dataEnd < len(event) {
		result = append(result, event[dataEnd:]...)
	}
	return result
}

// injectCacheIntoMessageStart adds cache simulation to a message_start event.
func injectCacheIntoMessageStart(jsonData []byte) []byte {
	inputTokens := gjson.GetBytes(jsonData, "message.usage.input_tokens")
	if !inputTokens.Exists() || inputTokens.Int() <= 10 {
		return nil
	}

	reduced, cacheRead, cacheCreation := simulateCacheUsage(inputTokens.Int())

	result := make([]byte, len(jsonData))
	copy(result, jsonData)
	result, _ = sjson.SetBytes(result, "message.usage.input_tokens", reduced)
	result, _ = sjson.SetBytes(result, "message.usage.cache_read_input_tokens", cacheRead)
	result, _ = sjson.SetBytes(result, "message.usage.cache_creation_input_tokens", cacheCreation)

	log.Debugf("[CacheSim] message_start: input_tokens=%d -> reduced=%d, cache_read=%d, cache_creation=%d",
		inputTokens.Int(), reduced, cacheRead, cacheCreation)
	return result
}

// injectCacheIntoMessageDelta adds cache simulation to a message_delta event.
func injectCacheIntoMessageDelta(jsonData []byte) []byte {
	inputTokens := gjson.GetBytes(jsonData, "usage.input_tokens")
	if !inputTokens.Exists() {
		return nil
	}

	// Check if real cached tokens from upstream already exist — respect them.
	existingCacheRead := gjson.GetBytes(jsonData, "usage.cache_read_input_tokens")
	if existingCacheRead.Exists() && existingCacheRead.Int() > 0 {
		// Upstream already provides real cache data; don't override.
		return nil
	}

	reduced, cacheRead, cacheCreation := simulateCacheUsage(inputTokens.Int())

	result := make([]byte, len(jsonData))
	copy(result, jsonData)
	result, _ = sjson.SetBytes(result, "usage.input_tokens", reduced)
	result, _ = sjson.SetBytes(result, "usage.cache_read_input_tokens", cacheRead)
	result, _ = sjson.SetBytes(result, "usage.cache_creation_input_tokens", cacheCreation)

	log.Debugf("[CacheSim] message_delta: input_tokens=%d -> reduced=%d, cache_read=%d, cache_creation=%d",
		inputTokens.Int(), reduced, cacheRead, cacheCreation)
	return result
}

// ---------------------------------------------------------------------------
// Converter wrapper functions
// ---------------------------------------------------------------------------

// WrapStreamWithCacheSimulation wraps a streaming response converter function,
// post-processing its output to inject cache simulation into SSE events.
func WrapStreamWithCacheSimulation(
	original func(ctx context.Context, model string, origReq, req, raw []byte, param *any) [][]byte,
) func(ctx context.Context, model string, origReq, req, raw []byte, param *any) [][]byte {
	return func(ctx context.Context, model string, origReq, req, raw []byte, param *any) [][]byte {
		chunks := original(ctx, model, origReq, req, raw, param)
		return injectCacheIntoSSEChunks(chunks)
	}
}

// WrapNonStreamWithCacheSimulation wraps a non-streaming response converter function,
// post-processing its output to inject cache simulation into the JSON response.
func WrapNonStreamWithCacheSimulation(
	original func(ctx context.Context, model string, origReq, req, raw []byte, param *any) []byte,
) func(ctx context.Context, model string, origReq, req, raw []byte, param *any) []byte {
	return func(ctx context.Context, model string, origReq, req, raw []byte, param *any) []byte {
		result := original(ctx, model, origReq, req, raw, param)
		return injectCacheIntoNonStreamResponse(result)
	}
}

// injectCacheIntoNonStreamResponse post-processes a non-streaming JSON response
// to inject cache simulation into the usage object.
func injectCacheIntoNonStreamResponse(responseJSON []byte) []byte {
	if len(responseJSON) == 0 {
		return responseJSON
	}

	inputTokens := gjson.GetBytes(responseJSON, "usage.input_tokens")
	if !inputTokens.Exists() || inputTokens.Int() <= 10 {
		return responseJSON
	}

	// Check if real cached tokens from upstream already exist — respect them.
	existingCacheRead := gjson.GetBytes(responseJSON, "usage.cache_read_input_tokens")
	if existingCacheRead.Exists() && existingCacheRead.Int() > 0 {
		return responseJSON
	}

	reduced, cacheRead, cacheCreation := simulateCacheUsage(inputTokens.Int())

	result := make([]byte, len(responseJSON))
	copy(result, responseJSON)
	result, _ = sjson.SetBytes(result, "usage.input_tokens", reduced)
	result, _ = sjson.SetBytes(result, "usage.cache_read_input_tokens", cacheRead)
	result, _ = sjson.SetBytes(result, "usage.cache_creation_input_tokens", cacheCreation)

	log.Debugf("[CacheSim] NonStream: input_tokens=%d -> reduced=%d, cache_read=%d, cache_creation=%d",
		inputTokens.Int(), reduced, cacheRead, cacheCreation)
	return result
}
