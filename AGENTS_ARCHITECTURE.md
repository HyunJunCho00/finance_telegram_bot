# Finance Telegram Bot: Multi-Agent Architecture

Ground truth document. All descriptions reflect what is **actually implemented** in code.
Internal API keys and credentials are omitted.

---

## LangGraph Execution Flow

```
collect_data
    → context_gathering
        (perplexity narrative, RAG ingest, funding, CVD, liquidations,
         RAG query, telegram news, self-correction feedback,
         microstructure, macro, deribit, fear&greed)
    → meta_agent          ← Market Regime Classifier
    → triage              ← Math-based anomaly detection + budget init
    → liquidity_expert    ← CVD / whale / liquidation analysis
    → microstructure_expert  ← Orderbook / spread / slippage analysis
    → macro_expert        ← Deribit options + macro economic analysis
    → generate_chart      ← Multi-panel chart (price + CVD + funding + heatmap)
    → vlm_expert          ← SOLE visual analyst; reads raw chart → blackboard["vlm_geometry"]
    → blackboard_synthesis ← Distills 4 expert JSONs into structured conflict map
    → judge_agent         ← Reads blackboard + synthesis; NO raw chart
    → risk_manager        ← CRO veto; enforces exchange limits
    → portfolio_leverage_guard  ← Hardcoded math; 2.0x account leverage cap
    → execute_trade
    → generate_report
    → data_synthesis
    → END
```

**Key design decisions:**
- Experts always run (triage no longer short-circuits to generate_chart when quiet)
- VLM is the sole chart consumer; Judge receives structured text, not the raw image
- `blackboard_synthesis` sits between VLM and Judge to pre-digest expert conflicts

---

## Agents

### Decision Core

#### JudgeAgent (`agents/judge_agent.py`)
- **Model**: `claude-sonnet-4-6` (via `use_premium=True`)
- **Role**: Final `LONG / SHORT / HOLD / CANCEL_AND_CLOSE` decision maker.
- **Receives**: Full blackboard (incl. `vlm_geometry` + `synthesis`) + market data + derivatives context. Does **NOT** receive raw chart image.
- **Key mechanics**:
  - Reads `blackboard["synthesis"]` conflict map — no need to scan raw expert JSONs independently.
  - Reads `regime_context.trust_directive` from MetaAgent.
  - Mandatory **Falsifiability Analysis**: must name the strongest counter-argument before confirming a trade.
  - **EV Veto** (post-processing): if `(win_prob × profit) − (loss_prob × loss) ≤ 0` → forced HOLD.
  - Allocation scaled by `risk_budget_pct` from MetaAgent before CRO review.
  - Reads episodic memory (LightRAG semantic search via `post_mortem.retrieve_similar_memories`).
  - Reads previous decision from DB for consistency.
- **Output schema**:
  ```json
  {
    "decision": "LONG|SHORT|HOLD|CANCEL_AND_CLOSE",
    "allocation_pct": 0-100,
    "leverage": 1-3,
    "entry_price": float,
    "stop_loss": float,
    "take_profit": float,
    "hold_duration": "string",
    "reasoning": {
      "counter_scenario": "...",
      "meta_agent_context": "...",
      "technical": "...",
      "derivatives": "...",
      "experts": "...",
      "narrative": "...",
      "final_logic": "..."
    },
    "win_probability_pct": 0-100,
    "expected_profit_pct": float,
    "expected_loss_pct": float,
    "ev_rationale": "...",
    "key_factors": ["..."]
  }
  ```

#### MetaAgent (`agents/meta_agent.py`)
- **Model**: `gemini-3-flash-preview` (via `role="macro"`)
- **Role**: Classifies Market Regime before the Expert Swarm runs.
- **Regime types**: `BULL_MOMENTUM | BEAR_MOMENTUM | RANGE_BOUND | VOLATILITY_PANIC | SIDEWAYS_ACCUMULATION`
- **Inputs**: market snapshot, Deribit context, funding, macro, RAG context, recent Telegram news.
- **Output**: `regime`, `rationale`, `trust_directive` (which expert to weight), `risk_budget_pct` (0–100), `risk_bias`
- `trust_directive` is passed to Judge verbatim. Experts always run regardless of regime — regime affects *weighting*, not *whether* experts execute.

#### RiskManagerAgent (`agents/risk_manager_agent.py`)
- **Model**: `gemini-3-flash-preview` (via `role="macro"`)
- **Role**: CRO. Has absolute veto over Judge's draft.
- **Rules**:
  1. Max leverage: 3x SWING, 1x POSITION.
  2. Volatility Tax: DVOL/2s10s panic → cut allocation 50%.
  3. Contrarian Veto: funding >+0.03% on LONG or massively negative on SHORT → HOLD.
  4. Liquidation Trap: >$50M opposite-direction liquidations → reduce sizing.
  5. Exchange routing: UPBIT (spot LONG only, ≤2,000,000 KRW, 1x), BINANCE (LONG/SHORT, ≤$2,000, ≤3x).
  6. Execution style: `PASSIVE_MAKER | SMART_DCA | MOMENTUM_SNIPER | CASINO_EXIT`.
- Fast-pass: if Judge decided HOLD or CANCEL_AND_CLOSE → CRO concurs, no LLM call.

#### portfolio_leverage_guard (node, no class)
- **No LLM** — pure Python math.
- Enforces 2.0x total account leverage cap across all positions.
- Runs after RiskManagerAgent.

---

### Expert Swarm (Blackboard Contributors)

All expert agents post a JSON result to `state["blackboard"]`. Experts always run every cycle.
`blackboard_synthesis` distills all expert outputs before Judge reads them.

#### LiquidityAgent (`agents/liquidity_agent.py`)
- **Model**: `gemini-3-flash-preview` (via `role="liquidity"`)
- **Inputs**: CVD context, Liquidation context.
- **Specialty**: "Liquidity Sweeps" and orderflow inconsistencies.
- **Output**:
  ```json
  { "anomaly": "whale_cvd_divergence|liquidation_hunting|none",
    "confidence": 0.0-1.0, "target_entry": float, "rationale": "..." }
  ```

#### MicrostructureAgent (`agents/microstructure_agent.py`)
- **Model**: `gemini-3-flash-preview` (via `role="microstructure"`)
- **Inputs**: Spread, orderbook imbalance, 100k USD slippage.
- **Specialty**: Spoofing detection, "Orderbook Fragility".
- **Output**:
  ```json
  { "anomaly": "microstructure_imbalance|fake_wall|none",
    "confidence": 0.0-1.0, "directional_bias": "UP|DOWN|NEUTRAL", "rationale": "..." }
  ```

#### MacroOptionsAgent (`agents/macro_options_agent.py`)
- **Model**: `gemini-3-flash-preview` (via `role="macro"`)
- **Inputs**: Deribit (DVOL, PCR, IV Term Structure, 25d Skew), Macro (DGS10, DXY, Nasdaq).
- **Specialty**: Institutional hedging patterns, systemic regime signals.
- **Output**:
  ```json
  { "anomaly": "options_panic|macro_divergence|none",
    "options_bias": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "rationale": "..." }
  ```

#### VLMGeometricAgent (`agents/vlm_geometric_agent.py`)
- **Model**: `gemini-3.1-pro-preview` (via `role="vlm_geometric"` → `settings.MODEL_VLM_GEOMETRIC`)
- **Input**: Chart image (base64), trading mode. **Only agent that receives raw chart.**
- **Position**: Runs after `generate_chart`, before `blackboard_synthesis`.
- **Specialty**: Geometric patterns, Fibonacci/trendline levels, liquidation heatmap traps, CVD visual divergence.
- **Output** (enriched schema):
  ```json
  { "anomaly": "geometric_trap|fake_breakout|exhaustion_top|capitulation_bottom|clean_breakout|accumulation_range|none",
    "directional_bias": "BULLISH|BEARISH|NEUTRAL",
    "confidence": 0.0-1.0,
    "key_levels": {
      "immediate_support": float, "immediate_resistance": float,
      "major_support": float, "major_resistance": float,
      "liquidation_pool_above": float, "liquidation_pool_below": float,
      "volume_poc": float
    },
    "fibonacci_context": "string with specific price levels",
    "pattern": "H&S|inv_H&S|double_top|double_bottom|bull_flag|bear_flag|ascending_triangle|wedge|none|other",
    "cvd_signal": "divergence_bullish|divergence_bearish|confirming|neutral",
    "funding_visual": "extreme_positive|high_positive|neutral|negative|extreme_negative",
    "rationale": "..." }
  ```

---

### Synthesis Layer

#### node_blackboard_synthesis (inline, no class)
- **Model**: `gemini-3-flash-preview` (via `role="liquidity"` → Flash)
- **Position**: Runs after all four experts (including VLM), before Judge.
- **Role**: Distills the full blackboard into a structured conflict map. Reduces Judge's cognitive load from scanning 4+ raw JSONs to resolving explicit, pre-identified conflicts.
- **Blackboard key**: `"synthesis"`
- **Output**:
  ```json
  { "consensus_signals": ["list of points all experts agree on, with price levels"],
    "conflicts": [
      { "between": ["expert_a", "expert_b"],
        "expert_a_claim": "...", "expert_b_claim": "...",
        "tiebreaker": "What data would resolve this" }
    ],
    "dominant_signal": "BULLISH|BEARISH|NEUTRAL",
    "highest_confidence_expert": "liquidity|microstructure|macro|vlm_geometry",
    "key_uncertainty": "Single most important question for Judge to resolve",
    "regime_note": "Which expert to weight given current regime and trust_directive" }
  ```

---

### Memory System

#### Episodic Memory (`executors/post_mortem.py`)
Post-mortems are written after every non-HOLD trade:
1. LLM (Gemini Flash) generates a structured `SETUP / OUTCOME / FUTURE CONSTRAINT` case study.
2. Stored in **both**: JSONL (fast local fallback) and LightRAG (semantic vector index via Milvus).

Retrieval (`retrieve_similar_memories`):
- **Primary**: LightRAG semantic vector search filtered by `channel=SYSTEM_POST_MORTEM`. Finds structurally similar past setups regardless of time.
- **Fallback**: JSONL time-reverse scan (3 most recent for the same symbol). Used when LightRAG/Milvus unavailable.

---

## Shared Infrastructure

### AIClient (`agents/claude_client.py`)
Multi-LLM router. Dispatches to Gemini, Claude, or GPT based on `model_id` prefix.

**Role → Model routing** (`_get_role_model_and_cap`):

| Role | Model | Input cap |
|---|---|---|
| `"judge"` / `use_premium=True` | `claude-sonnet-4-6` | 25,000 chars |
| `"liquidity"` | `gemini-3-flash-preview` | 15,000 chars |
| `"microstructure"` | `gemini-3-flash-preview` | 15,000 chars |
| `"macro"` | `gemini-3-flash-preview` | 15,000 chars |
| `"vlm_geometric"` | `gemini-3.1-pro-preview` | 15,000 chars |
| `"self_correction"` / `"feedback"` / `"post_mortem"` | `claude-sonnet-4-6` | 10,000 chars |
| `"rag"` / `"rag_extraction"` | `gemini-3-flash-preview` | 5,000 chars |

Fallback chain: Claude → Gemini default, GPT → Gemini default on API error.

---

## Conviction Hierarchy (Judge resolves deadlocks)

| Scenario | Priority order |
|---|---|
| Short-term / Swing entry | Microstructure > Liquidity > Macro > VLM Geometry |
| Medium/Long-term / Position | Macro/Options > Liquidity > VLM Geometry > Microstructure |

---

## Model Summary

| Agent / Component | Model | Note |
|---|---|---|
| JudgeAgent | `claude-sonnet-4-6` | No raw chart; reads blackboard synthesis |
| MetaAgent | `gemini-3-flash-preview` | Regime classification |
| RiskManagerAgent | `gemini-3-flash-preview` | CRO veto layer |
| LiquidityAgent | `gemini-3-flash-preview` | Text-only |
| MicrostructureAgent | `gemini-3-flash-preview` | Text-only |
| MacroOptionsAgent | `gemini-3-flash-preview` | Text-only |
| VLMGeometricAgent | `gemini-3.1-pro-preview` | Sole chart consumer; enriched output schema |
| BlackboardSynthesis | `gemini-3-flash-preview` | Conflict map, pre-digests blackboard for Judge |
| Self-correction / feedback | `claude-sonnet-4-6` | Via evaluators |
| RAG extraction | `gemini-3-flash-preview` | LightRAG pipeline |
| Post-mortem lesson | `gemini-3-flash-preview` | Case study generation |
| Telegram chat bot | `gemini-3-flash-preview` | Interactive chat |

---

## Triage Logic

`node_triage` performs Python/Pandas anomaly detection (no LLM):
- Volume-backed breakout: ATR spike ≥150% + Volume spike ≥200% + OI increase → `true_breakout`
- Liquidation cascade: same spike but OI decreased → `liquidation_cascade`
- Liquidation cluster: >$100k liquidation in 1h → `liquidation_cluster`
- Microstructure breakdown: orderbook imbalance >70% → `microstructure_imbalance`
- Options panic: DVOL spike or IV term inversion → `options_panic`

`route_triage` always routes to `liquidity_expert` (full swarm). Anomaly context is embedded in state fields; experts see it through their input strings. MetaAgent `trust_directive` handles regime-based weighting at Judge level.

---

## Removed (deprecated)

The following files existed in earlier versions and have been deleted:
- `agents/bullish_agent.py` — Bull/Bear adversarial debate pattern, superseded by Expert Swarm.
- `agents/bearish_agent.py` — Same.
- `agents/risk_agent.py` — Intermediate risk synthesizer from Bull/Bear debate, superseded by RiskManagerAgent.

---

*Last updated: 2026-03-01*
