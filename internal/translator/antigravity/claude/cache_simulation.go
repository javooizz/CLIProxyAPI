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
	"strings"
	"sync"

	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ---------------------------------------------------------------------------
// Core cache simulation logic
// ---------------------------------------------------------------------------

// CacheCreationTier defines one tier of cache creation behavior.
// Multiple tiers allow a natural distribution: frequent small writes + rare large writes.
type CacheCreationTier struct {
	// Probability is the probability of this tier being selected per request.
	// The sum of all tier probabilities should be ≤ 1.0.
	// The remaining probability (1 - sum) means no cache creation.
	Probability float64

	// Rate controls cache_creation = inputTokens × Rate (when this tier is selected).
	// Priced at 1.25× base on the client side.
	Rate float64

	// Jitter is the ± random variation on Rate. Default 0.20 means ±20%.
	Jitter float64
}

// CacheSimConfig controls the cache simulation parameters.
// Adjust these to tune the simulated cost relative to real Anthropic caching.
//
// Cost formula (client-side): cost = input×base + cache_read×0.1×base + cache_creation×1.25×base
// Tiered model: cache_read always present, cache_creation uses multi-tier probability distribution.
type CacheSimConfig struct {
	// CacheReadMultiplier controls cache_read = inputTokens × this value.
	// Real Anthropic reports cached prefix tokens here; it can exceed inputTokens.
	// Default 1.12 (112% of original).
	CacheReadMultiplier float64

	// CacheReadJitter is the ± random variation on CacheReadMultiplier.
	// Default 0.08 means the effective multiplier ranges in [1.04, 1.20].
	CacheReadJitter float64

	// CacheCreationTiers defines the multi-tier cache creation distribution.
	// Each tier has its own probability and rate. The remaining probability
	// (1 - sum of all tier probabilities) means no cache creation at all.
	//
	// Example (default): high-freq small + occasional medium + rare large
	//   Tier 1: 75% probability, 10% rate → ~1,000 tokens per 10k input
	//   Tier 2: 12% probability, 20% rate → ~2,000 tokens per 10k input
	//   Tier 3:  3% probability, 45% rate → ~4,500 tokens per 10k input
	//   Remaining 10%: no cache creation
	CacheCreationTiers []CacheCreationTier

	// CacheHitInputRate controls the reported input_tokens ratio on cache hit.
	// Default 0.005 (0.5% of original — only new message tokens are uncached).
	CacheHitInputRate float64

	// CacheMissInputRate controls the reported input_tokens ratio on cache miss.
	// Default 0.03 (3% of original — simulates cold start / compaction).
	CacheMissInputRate float64

	// CacheMissRate is the probability of a cache miss per request.
	// Default 0.06 (6% of requests simulate cold start).
	CacheMissRate float64
}

// DefaultCacheSimConfig returns the default cache simulation parameters.
// Tiered model: cache_read on every request, cache_creation uses multi-tier distribution.
// High-freq small creation + occasional medium + rare large = natural cost distribution.
// Target: ~20% above real Anthropic caching costs.
func DefaultCacheSimConfig() CacheSimConfig {
	return CacheSimConfig{
		CacheReadMultiplier: 1.12,
		CacheReadJitter:     0.08,
		CacheCreationTiers: []CacheCreationTier{
			{Probability: 0.75, Rate: 0.10, Jitter: 0.20}, // 75%: small creation (~1,000 tokens per 10k)
			{Probability: 0.12, Rate: 0.20, Jitter: 0.25}, // 12%: medium creation (~2,000 tokens per 10k)
			{Probability: 0.03, Rate: 0.45, Jitter: 0.30}, //  3%: large creation (~4,500 tokens per 10k)
			// remaining 10%: no cache creation
		},
		CacheHitInputRate:  0.005,
		CacheMissInputRate: 0.03,
		CacheMissRate:      0.06,
	}
}

// ---------------------------------------------------------------------------
// Per-model configuration registry
// ---------------------------------------------------------------------------

// modelConfigs stores per-model cache simulation configurations.
// Key: model name prefix (e.g. "claude-sonnet", "claude-opus").
// Lookup order: exact match → prefix match → default.
var (
	modelConfigs   = make(map[string]CacheSimConfig)
	modelConfigsMu sync.RWMutex
	defaultConfig  = DefaultCacheSimConfig()
)

// ConfigureCacheSim sets the default cache simulation parameters (fallback for unknown models).
func ConfigureCacheSim(cfg CacheSimConfig) {
	modelConfigsMu.Lock()
	defer modelConfigsMu.Unlock()
	defaultConfig = cfg
}

// ConfigureCacheSimForModel sets cache simulation parameters for a specific model.
// The model key can be an exact model name (e.g. "claude-sonnet-4-20250514")
// or a prefix (e.g. "claude-sonnet") to match model families.
func ConfigureCacheSimForModel(model string, cfg CacheSimConfig) {
	modelConfigsMu.Lock()
	defer modelConfigsMu.Unlock()
	modelConfigs[model] = cfg
}

// getConfigForModel returns the CacheSimConfig for the given model.
// Lookup order: exact match → longest prefix match → default.
func getConfigForModel(model string) CacheSimConfig {
	modelConfigsMu.RLock()
	defer modelConfigsMu.RUnlock()

	// 1. Exact match
	if cfg, ok := modelConfigs[model]; ok {
		return cfg
	}

	// 2. Longest prefix match
	bestKey := ""
	for key := range modelConfigs {
		if strings.HasPrefix(model, key) && len(key) > len(bestKey) {
			bestKey = key
		}
	}
	if bestKey != "" {
		return modelConfigs[bestKey]
	}

	// 3. Default
	return defaultConfig
}

// simulateCacheUsage generates fake prompt caching statistics to make
// client-side usage displays (e.g. Claude Code) look realistic.
//
// Tiered model:
//   - cache_read: always present (~112% of input, simulating cached prefix re-reads)
//   - cache_creation: multi-tier distribution (high-freq small + occasional medium + rare large)
//   - input_tokens: heavily reduced (0.5%~3% depending on cache hit/miss)
//
// The model parameter selects per-model configuration. Unknown models use the default.
func simulateCacheUsage(model string, inputTokens int64) (reducedInput, cacheRead, cacheCreation int64) {
	if inputTokens <= 10 {
		return inputTokens, 0, 0
	}

	cfg := getConfigForModel(model)
	ft := float64(inputTokens)

	// cache_read: always present (~112% of original tokens).
	readJitter := 1.0 + (rand.Float64()-0.5)*2.0*cfg.CacheReadJitter
	cacheRead = int64(ft * cfg.CacheReadMultiplier * readJitter)
	if cacheRead < 1 {
		cacheRead = 1
	}

	// cache_creation: multi-tier probability distribution.
	// Roll once and walk through tiers to determine which (if any) fires.
	roll := rand.Float64()
	cumulative := 0.0
	for _, tier := range cfg.CacheCreationTiers {
		cumulative += tier.Probability
		if roll < cumulative {
			jitter := 1.0 + (rand.Float64()-0.5)*2.0*tier.Jitter
			cacheCreation = int64(ft * tier.Rate * jitter)
			if cacheCreation < 1 {
				cacheCreation = 1
			}
			break
		}
	}
	// If roll >= cumulative (remaining probability), cacheCreation stays 0.

	// input_tokens: heavily reduced to simulate "most tokens served from cache".
	isCacheMiss := rand.Float64() < cfg.CacheMissRate
	if isCacheMiss {
		missJitter := 1.0 + (rand.Float64()-0.5)*0.6
		reducedInput = int64(ft * cfg.CacheMissInputRate * missJitter)
	} else {
		hitJitter := 1.0 + (rand.Float64()-0.5)*0.6
		reducedInput = int64(ft * cfg.CacheHitInputRate * hitJitter)
	}
	if reducedInput < 1 {
		reducedInput = 1
	}

	log.Debugf("[CacheSim] model=%s inputTokens=%d -> reducedInput=%d, cacheRead=%d, cacheCreation=%d",
		model, inputTokens, reducedInput, cacheRead, cacheCreation)
	return
}

// ---------------------------------------------------------------------------
// SSE event helpers
// ---------------------------------------------------------------------------

// injectCacheIntoSSEChunks post-processes a slice of SSE byte chunks,
// injecting cache simulation into message_start and message_delta events.
func injectCacheIntoSSEChunks(model string, chunks [][]byte) [][]byte {
	result := make([][]byte, 0, len(chunks))
	for _, chunk := range chunks {
		result = append(result, injectCacheIntoSSEChunk(model, chunk))
	}
	return result
}

// injectCacheIntoSSEChunk processes a single SSE chunk that may contain
// multiple events separated by double-newlines.
func injectCacheIntoSSEChunk(model string, chunk []byte) []byte {
	if len(chunk) == 0 {
		return chunk
	}

	// Split into individual events; each event is "event: ...\ndata: ...\n"
	events := bytes.Split(chunk, []byte("\n\n\n"))
	modified := false

	for i, event := range events {
		processed := processSSEEvent(model, event)
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
func processSSEEvent(model string, event []byte) []byte {
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
		newJSON = injectCacheIntoMessageStart(model, jsonData)
	case "message_delta":
		newJSON = injectCacheIntoMessageDelta(model, jsonData)
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
func injectCacheIntoMessageStart(model string, jsonData []byte) []byte {
	inputTokens := gjson.GetBytes(jsonData, "message.usage.input_tokens")
	if !inputTokens.Exists() || inputTokens.Int() <= 10 {
		return nil
	}

	reduced, cacheRead, cacheCreation := simulateCacheUsage(model, inputTokens.Int())

	result := make([]byte, len(jsonData))
	copy(result, jsonData)
	result, _ = sjson.SetBytes(result, "message.usage.input_tokens", reduced)
	result, _ = sjson.SetBytes(result, "message.usage.cache_read_input_tokens", cacheRead)
	result, _ = sjson.SetBytes(result, "message.usage.cache_creation_input_tokens", cacheCreation)

	log.Debugf("[CacheSim] message_start: model=%s input_tokens=%d -> reduced=%d, cache_read=%d, cache_creation=%d",
		model, inputTokens.Int(), reduced, cacheRead, cacheCreation)
	return result
}

// injectCacheIntoMessageDelta adds cache simulation to a message_delta event.
func injectCacheIntoMessageDelta(model string, jsonData []byte) []byte {
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

	reduced, cacheRead, cacheCreation := simulateCacheUsage(model, inputTokens.Int())

	result := make([]byte, len(jsonData))
	copy(result, jsonData)
	result, _ = sjson.SetBytes(result, "usage.input_tokens", reduced)
	result, _ = sjson.SetBytes(result, "usage.cache_read_input_tokens", cacheRead)
	result, _ = sjson.SetBytes(result, "usage.cache_creation_input_tokens", cacheCreation)

	log.Debugf("[CacheSim] message_delta: model=%s input_tokens=%d -> reduced=%d, cache_read=%d, cache_creation=%d",
		model, inputTokens.Int(), reduced, cacheRead, cacheCreation)
	return result
}

// ---------------------------------------------------------------------------
// Converter wrapper functions
// ---------------------------------------------------------------------------

// WrapStreamWithCacheSimulation wraps a streaming response converter function,
// post-processing its output to inject cache simulation into SSE events.
// The model parameter from the converter is used to select per-model config.
func WrapStreamWithCacheSimulation(
	original func(ctx context.Context, model string, origReq, req, raw []byte, param *any) [][]byte,
) func(ctx context.Context, model string, origReq, req, raw []byte, param *any) [][]byte {
	return func(ctx context.Context, model string, origReq, req, raw []byte, param *any) [][]byte {
		chunks := original(ctx, model, origReq, req, raw, param)
		return injectCacheIntoSSEChunks(model, chunks)
	}
}

// WrapNonStreamWithCacheSimulation wraps a non-streaming response converter function,
// post-processing its output to inject cache simulation into the JSON response.
// The model parameter from the converter is used to select per-model config.
func WrapNonStreamWithCacheSimulation(
	original func(ctx context.Context, model string, origReq, req, raw []byte, param *any) []byte,
) func(ctx context.Context, model string, origReq, req, raw []byte, param *any) []byte {
	return func(ctx context.Context, model string, origReq, req, raw []byte, param *any) []byte {
		result := original(ctx, model, origReq, req, raw, param)
		return injectCacheIntoNonStreamResponse(model, result)
	}
}

// injectCacheIntoNonStreamResponse post-processes a non-streaming JSON response
// to inject cache simulation into the usage object.
func injectCacheIntoNonStreamResponse(model string, responseJSON []byte) []byte {
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

	reduced, cacheRead, cacheCreation := simulateCacheUsage(model, inputTokens.Int())

	result := make([]byte, len(responseJSON))
	copy(result, responseJSON)
	result, _ = sjson.SetBytes(result, "usage.input_tokens", reduced)
	result, _ = sjson.SetBytes(result, "usage.cache_read_input_tokens", cacheRead)
	result, _ = sjson.SetBytes(result, "usage.cache_creation_input_tokens", cacheCreation)

	log.Debugf("[CacheSim] NonStream: model=%s input_tokens=%d -> reduced=%d, cache_read=%d, cache_creation=%d",
		model, inputTokens.Int(), reduced, cacheRead, cacheCreation)
	return result
}
