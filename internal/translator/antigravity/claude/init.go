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
	// All models target ~20% above real Anthropic caching costs.
	// Tiered cache_creation: high-freq small + occasional medium + rare large.

	// Opus: expensive model. Slightly higher medium/large tier rates.
	// E[creation] = 0.72×0.09 + 0.13×0.22 + 0.03×0.48 = 0.0648+0.0286+0.0144 = 0.1078
	ConfigureCacheSimForModel("claude-opus", CacheSimConfig{
		CacheReadMultiplier: 1.11,
		CacheReadJitter:     0.08,
		CacheCreationTiers: []CacheCreationTier{
			{Probability: 0.62, Rate: 0.10, Jitter: 0.20}, // 72%: small (~900 tokens per 10k)
			{Probability: 0.23, Rate: 0.22, Jitter: 0.25}, // 13%: medium (~2,200 tokens per 10k)
			{Probability: 0.03, Rate: 0.48, Jitter: 0.30}, //  3%: large (~4,800 tokens per 10k)
			// remaining 12%: no cache creation
		},
		CacheHitInputRate:  0.005,
		CacheMissInputRate: 0.03,
		CacheMissRate:      0.06,
	})

	// Sonnet: mid-tier model, balanced cache behavior.
	// E[creation] = 0.75×0.10 + 0.12×0.20 + 0.03×0.45 = 0.075+0.024+0.0135 = 0.1125
	ConfigureCacheSimForModel("claude-sonnet", CacheSimConfig{
		CacheReadMultiplier: 1.11,
		CacheReadJitter:     0.08,
		CacheCreationTiers: []CacheCreationTier{
			{Probability: 0.65, Rate: 0.10, Jitter: 0.20}, // 75%: small (~1,000 tokens per 10k)
			{Probability: 0.22, Rate: 0.20, Jitter: 0.25}, // 12%: medium (~2,000 tokens per 10k)
			{Probability: 0.03, Rate: 0.45, Jitter: 0.30}, //  3%: large (~4,500 tokens per 10k)
			// remaining 10%: no cache creation
		},
		CacheHitInputRate:  0.005,
		CacheMissInputRate: 0.03,
		CacheMissRate:      0.06,
	})

	// Haiku: lightweight model, higher creation frequency with smaller values.
	// E[creation] = 0.78×0.08 + 0.12×0.18 + 0.03×0.42 = 0.0624+0.0216+0.0126 = 0.0966
	ConfigureCacheSimForModel("claude-haiku", CacheSimConfig{
		CacheReadMultiplier: 1.15,
		CacheReadJitter:     0.06,
		CacheCreationTiers: []CacheCreationTier{
			{Probability: 0.58, Rate: 0.08, Jitter: 0.20}, // 78%: small (~800 tokens per 10k)
			{Probability: 0.22, Rate: 0.18, Jitter: 0.25}, // 12%: medium (~1,800 tokens per 10k)
			{Probability: 0.03, Rate: 0.42, Jitter: 0.30}, //  3%: large (~4,200 tokens per 10k)
			// remaining 7%: no cache creation
		},
		CacheHitInputRate:  0.004,
		CacheMissInputRate: 0.02,
		CacheMissRate:      0.06,
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
