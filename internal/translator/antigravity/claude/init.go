package claude

import (
	. "github.com/router-for-me/CLIProxyAPI/v6/internal/constant"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/translator"
)

func init() {
	// Register per-model cache simulation configurations.
	// Prefix matching: "claude-opus" matches "claude-opus-4-6", "claude-opus-4-5-20251101", etc.
	//
	// Opus: expensive model, higher cache_creation rate per hit, moderate probability.
	ConfigureCacheSimForModel("claude-opus", CacheSimConfig{
		CacheReadMultiplier:      1.12,
		CacheReadJitter:          0.08,
		CacheCreationProbability: 0.55,
		CacheCreationRate:        0.06,
		CacheHitInputRate:        0.005,
		CacheMissInputRate:       0.03,
		CacheMissRate:            0.05,
	})

	// Sonnet: mid-tier model, balanced cache behavior.
	ConfigureCacheSimForModel("claude-sonnet", CacheSimConfig{
		CacheReadMultiplier:      1.15,
		CacheReadJitter:          0.08,
		CacheCreationProbability: 0.55,
		CacheCreationRate:        0.06,
		CacheHitInputRate:        0.005,
		CacheMissInputRate:       0.03,
		CacheMissRate:            0.05,
	})

	// Haiku: lightweight model, high cache hit rate, minimal creation.
	ConfigureCacheSimForModel("claude-haiku", CacheSimConfig{
		CacheReadMultiplier:      1.08,
		CacheReadJitter:          0.06,
		CacheCreationProbability: 0.35,
		CacheCreationRate:        0.05,
		CacheHitInputRate:        0.004,
		CacheMissInputRate:       0.02,
		CacheMissRate:            0.03,
	})

	translator.Register(
		Claude,
		Antigravity,
		ConvertClaudeRequestToAntigravity,
		interfaces.TranslateResponse{
			Stream:     WrapStreamWithCacheSimulation(ConvertAntigravityResponseToClaude),
			NonStream:  WrapNonStreamWithCacheSimulation(ConvertAntigravityResponseToClaudeNonStream),
			TokenCount: ClaudeTokenCount,
		},
	)
}
