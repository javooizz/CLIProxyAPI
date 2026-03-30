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
// Uses a probability model: most requests are pure cache hits (cache_creation=0),
// only a small fraction of requests trigger cache writes.
//
// Token conservation: input_tokens + cache_read + cache_creation ≈ totalInput
// This matches real Anthropic API behavior.
type CacheSimConfig struct {
	// CacheCreationProbability is the probability of a cache write per request.
	// Real Anthropic creates cache only on new/changed system prompts or tool defs.
	// Default 0.02 (2% of requests trigger cache creation).
	CacheCreationProbability float64

	// CacheCreationMinRatio is the minimum fraction of total tokens used for
	// cache_creation when a cache write occurs. Default 0.20.
	CacheCreationMinRatio float64

	// CacheCreationMaxRatio is the maximum fraction of total tokens used for
	// cache_creation when a cache write occurs. Default 0.50.
	CacheCreationMaxRatio float64
}

// DefaultCacheSimConfig returns the default cache simulation parameters.
// These defaults model realistic Anthropic caching behavior:
//   - 98% of requests are pure cache hits (cache_creation = 0)
//   - 2% of requests trigger cache writes (20%-50% of tokens)
//   - Token conservation: input(1) + cache_read + cache_creation ≈ totalInput
func DefaultCacheSimConfig() CacheSimConfig {
	return CacheSimConfig{
		CacheCreationProbability: 0.02,
		CacheCreationMinRatio:    0.20,
		CacheCreationMaxRatio:    0.50,
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

// simulateCacheUsage generates fake prompt caching statistics using a
// probability model that matches real Anthropic API behavior.
//
// Probability model:
//   - 98% of requests → pure cache hit: input=1, cache_read=total-1, cache_creation=0
//   - 2% of requests  → cache write:    input=1, cache_read=remainder, cache_creation=20~50%
//
// Token conservation: reducedInput + cacheRead + cacheCreation = totalInput
// This ensures client-side billing calculations remain accurate.
func simulateCacheUsage(inputTokens int64) (reducedInput, cacheRead, cacheCreation int64) {
	if inputTokens <= 1 {
		return inputTokens, 0, 0
	}

	cfg := simConfig
	cacheTokens := inputTokens - 1 // reserve 1 token for input_tokens

	if rand.Float64() < cfg.CacheCreationProbability {
		// Cache write: 2% probability — new/changed system prompt or tool definitions.
		// cache_creation = 20%~50% of total tokens.
		ratio := cfg.CacheCreationMinRatio + rand.Float64()*(cfg.CacheCreationMaxRatio-cfg.CacheCreationMinRatio)
		cacheCreation = int64(float64(inputTokens) * ratio)
		if cacheCreation < 1 {
			cacheCreation = 1
		}
		cacheRead = cacheTokens - cacheCreation
		if cacheRead < 0 {
			cacheRead = 0
		}

		log.Debugf("[CacheSim] CACHE WRITE: inputTokens=%d -> input=1, cache_read=%d, cache_creation=%d (ratio=%.2f)",
			inputTokens, cacheRead, cacheCreation, ratio)
	} else {
		// Cache hit: 98% probability — everything served from cache.
		cacheRead = cacheTokens
		cacheCreation = 0

		log.Debugf("[CacheSim] cache hit: inputTokens=%d -> input=1, cache_read=%d, cache_creation=0",
			inputTokens, cacheRead)
	}

	reducedInput = 1
	return
}

// ---------------------------------------------------------------------------
// SSE event helpers
// ---------------------------------------------------------------------------

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
