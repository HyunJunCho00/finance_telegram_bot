# Finance Telegram Bot: Multi-Agent Architecture

Ground truth document. All descriptions reflect what is **actually implemented** in code.
Internal API keys and credentials are omitted.

---

## LangGraph Execution Flow

```
collect_data
    → context_gathering
        (funding, CVD, liquidations, RAG query, telegram news,
         self-correction feedback, microstructure, macro, deribit, fear&greed)
    → meta_agent          ← Market Regime Classifier
    → triage              ← Budget allocation; decides which experts run
    → [liquidity_expert | microstructure_expert | macro_expert]  ← parallel
    → generate_chart      ← Builds multi-panel chart (price + CVD + funding + heatmap)
    → vlm_expert          ← Analyzes chart image, posts result to Blackboard
    → judge_agent         ← Reads full Blackboard + raw chart image; makes final decision
    → risk_manager        ← CRO veto layer; enforces exchange limits
    → portfolio_leverage_guard  ← Hardcoded math; enforces 2.0x account leverage cap
    → execute_trade
    → generate_report
    → data_synthesis
    → END
```

Sequential fallback (`_run_sequential`) mirrors the same node order when LangGraph is unavailable.

---

## Agents

### Decision Core

#### JudgeAgent (`agents/judge_agent.py`)
- **Model**: `claude-sonnet-4-6` (via `use_premium=True`)
- **Role**: Synthesizes the full Blackboard (all expert analyses) + raw chart image → outputs final `LONG / SHORT / HOLD / CANCEL_AND_CLOSE` decision.
- **Key mechanics**:
  - Receives `regime_context` from MetaAgent to weight experts appropriately.
  - Mandatory **Falsifiability Analysis**: must articulate the strongest counter-argument before confirming a trade.
  - **EV Veto** (post-processing in orchestrator): if `(win_prob × profit) − (loss_prob × loss) ≤ 0`, decision is forced to `HOLD`.
  - Allocation is scaled by `risk_budget_pct` from MetaAgent before passing to CRO.
  - Reads episodic memory (`retrieve_similar_memories`) for past similar trade patterns.
  - Reads previous decision from DB for consistency enforcement.
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
- **Model**: `gemini-3-flash-preview` (via `role="macro"` → `settings.MODEL_MACRO`)
- **Role**: Classifies the current Market Regime before expert swarm runs.
- **Regime types**: `BULL_MOMENTUM | BEAR_MOMENTUM | RANGE_BOUND | VOLATILITY_PANIC | SIDEWAYS_ACCUMULATION`
- **Inputs**: market data snapshot, Deribit context, funding context, macro context, RAG context, recent Telegram news.
- **Output**: `regime`, `rationale`, `trust_directive` (which experts to weight), `risk_budget_pct` (0–100), `risk_bias`
- `trust_directive` is passed verbatim into JudgeAgent's context so the Judge knows how to weight conflicting experts.

#### RiskManagerAgent (`agents/risk_manager_agent.py`)
- **Model**: `gemini-3-flash-preview` (via `role="macro"` → `settings.MODEL_MACRO`)
- **Role**: Chief Risk Officer. Has absolute veto over the Judge's draft.
- **Rules enforced**:
  1. Max leverage: 3x SWING, 1x POSITION.
  2. Volatility Tax: if DVOL or 2s10s spread shows panic → cut allocation 50%.
  3. Contrarian Veto: funding >+0.03% on LONG, or massively negative on SHORT → force HOLD.
  4. Liquidation Trap: >$50M liquidations in opposite direction → reduce confidence/sizing.
  5. Exchange routing: UPBIT (spot-only LONG, max 2,000,000 KRW, leverage always 1x), BINANCE (LONG/SHORT, max $2,000 USD, max 3x).
  6. Execution style selection: `PASSIVE_MAKER | SMART_DCA | MOMENTUM_SNIPER | CASINO_EXIT`.
- Fast-pass: if Judge already decided `HOLD` or `CANCEL_AND_CLOSE`, CRO concurs immediately without running the LLM.
- **Output**: merges CRO overrides back into Judge's draft; adds `cro_veto_applied`, `risk_manager_note`.

#### portfolio_leverage_guard (node, no separate class)
- **No LLM** — pure Python math.
- Enforces a hard **2.0x total account leverage cap** across all open positions.
- Runs after RiskManagerAgent.

---

### Expert Swarm (Blackboard Contributors)

All expert agents post a JSON result to `state["blackboard"]`. JudgeAgent reads the full Blackboard.

#### LiquidityAgent (`agents/liquidity_agent.py`)
- **Model**: `gemini-3-flash-preview` (via `role="liquidity"` → `settings.MODEL_LIQUIDITY`)
- **Inputs**: CVD context (Cumulative Volume Delta, whale accumulation/distribution), Liquidation context.
- **Specialty**: Identifies "Liquidity Sweeps" and orderflow inconsistencies.
- **Output**:
  ```json
  { "anomaly": "whale_cvd_divergence|liquidation_hunting|none",
    "confidence": 0.0-1.0,
    "target_entry": float,
    "rationale": "..." }
  ```

#### MicrostructureAgent (`agents/microstructure_agent.py`)
- **Model**: `gemini-3-flash-preview` (via `role="microstructure"` → `settings.MODEL_MICROSTRUCTURE`)
- **Inputs**: Bid/Ask spread, orderbook imbalance, 100k USD slippage estimate.
- **Specialty**: Identifies fake walls (spoofing) and real momentum before price moves ("Orderbook Fragility").
- **Output**:
  ```json
  { "anomaly": "microstructure_imbalance|fake_wall|none",
    "confidence": 0.0-1.0,
    "directional_bias": "UP|DOWN|NEUTRAL",
    "rationale": "..." }
  ```

#### MacroOptionsAgent (`agents/macro_options_agent.py`)
- **Model**: `gemini-3-flash-preview` (via `role="macro"` → `settings.MODEL_MACRO`)
- **Inputs**: Deribit context (DVOL, PCR, IV Term Structure, 25d Skew), Macro context (DGS10, DXY, Nasdaq).
- **Specialty**: Reads institutional hedging patterns and systemic regime signals.
- **Output**:
  ```json
  { "anomaly": "options_panic|macro_divergence|none",
    "options_bias": "BULLISH|BEARISH|NEUTRAL",
    "confidence": 0.0-1.0,
    "rationale": "..." }
  ```

#### VLMGeometricAgent (`agents/vlm_geometric_agent.py`)
- **Model**: `gemini-2.5-flash` (fallback default — see note below)
- **Input**: Chart image (base64), trading mode.
- **Specialty**: Reads geometric patterns (Fibonacci, trendlines), liquidation heatmap traps, fake breakouts.
- **Position in flow**: Runs AFTER `generate_chart`, BEFORE `judge_agent`. Blackboard key: `"vlm_geometry"`.
- **Note (known routing bug)**: `settings.MODEL_VLM_GEOMETRIC = "gemini-3.1-pro-preview"` is defined but the `"vlm_geometric"` role is not mapped in `AIClient._get_role_model_and_cap()`. The agent therefore falls through to `self.default_model_id = "gemini-2.5-flash"`. Fix: add `if role == "vlm_geometric": return settings.MODEL_VLM_GEOMETRIC, ...` to `claude_client.py`.
- **Output**:
  ```json
  { "anomaly": "geometric_trap|fake_breakout|none",
    "confidence": 0.0-1.0,
    "visual_target": float,
    "rationale": "..." }
  ```

---

## Shared Infrastructure

### AIClient (`agents/claude_client.py`)
Multi-LLM router. Dispatches to Gemini, Claude, or GPT based on `model_id` prefix. Shared by all agents.

**Role → Model routing** (`_get_role_model_and_cap`):
| Role | Model | Input cap |
|---|---|---|
| `"judge"` / `use_premium=True` | `claude-sonnet-4-6` | 25,000 chars |
| `"liquidity"` | `gemini-3-flash-preview` | 15,000 chars |
| `"microstructure"` | `gemini-3-flash-preview` | 15,000 chars |
| `"macro"` | `gemini-3-flash-preview` | 15,000 chars |
| `"self_correction"` / `"feedback"` / `"post_mortem"` | `claude-sonnet-4-6` | 10,000 chars |
| `"rag"` / `"rag_extraction"` | `gemini-3-flash-preview` | 5,000 chars |
| `"vlm_geometric"` | **BUG: falls to default** `gemini-2.5-flash` | 15,000 chars |

Fallback chain: Claude → Gemini default, GPT → Gemini default on API error.

---

## Conviction Hierarchy (Judge uses this to resolve deadlocks)

| Scenario | Priority order |
|---|---|
| Short-term / Swing entry | Microstructure > Liquidity > Macro > VLM Geometry |
| Medium/Long-term / Position | Macro/Options > Liquidity > VLM Geometry > Microstructure |

---

## Model Summary

| Agent | Actual model | Note |
|---|---|---|
| JudgeAgent | `claude-sonnet-4-6` | Only agent to receive raw chart image |
| MetaAgent | `gemini-3-flash-preview` | Regime classification |
| RiskManagerAgent | `gemini-3-flash-preview` | CRO veto layer |
| LiquidityAgent | `gemini-3-flash-preview` | Text-only |
| MicrostructureAgent | `gemini-3-flash-preview` | Text-only |
| MacroOptionsAgent | `gemini-3-flash-preview` | Text-only |
| VLMGeometricAgent | `gemini-2.5-flash` (**bug**) | Should be `gemini-3.1-pro-preview` |
| Self-correction / feedback | `claude-sonnet-4-6` | Via evaluators |
| RAG extraction | `gemini-3-flash-preview` | LightRAG pipeline |
| Telegram chat bot | `gemini-3-flash-preview` | Interactive chat |

---

## Removed (deprecated)

The following files existed in earlier versions and have been deleted:
- `agents/bullish_agent.py` — Bull/Bear adversarial debate pattern, superseded by Expert Swarm.
- `agents/bearish_agent.py` — Same.
- `agents/risk_agent.py` — Intermediate risk synthesizer from Bull/Bear debate, superseded by RiskManagerAgent.

---

*Last updated: 2026-03-01*
