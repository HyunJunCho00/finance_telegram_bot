from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

from config.settings import TradingMode, settings
from executors.playbook_guard import evaluate_playbook_consistency
from processors.flow_confirm_engine import flow_confirm_engine


class PolicyEngine:
    """Human-defined trading policy. Final authority over trade approval."""

    @staticmethod
    def _has_meaningful_playbook(playbook: Optional[Dict]) -> bool:
        if not isinstance(playbook, dict):
            return False
        entry = playbook.get("entry_conditions", [])
        return isinstance(entry, list) and len(entry) > 0

    def _fallback_playbook_from_active_setup(
        self,
        active_setup: Dict,
        *,
        allocation_pct: object = None,
    ) -> Dict:
        if not isinstance(active_setup, dict) or not active_setup:
            return {}

        entry_conditions = active_setup.get("trigger_conditions", [])
        if not isinstance(entry_conditions, list):
            entry_conditions = []

        invalidation_conditions = active_setup.get("invalidation_conditions", [])
        if not isinstance(invalidation_conditions, list):
            invalidation_conditions = []

        side = str(active_setup.get("side", "")).upper()
        allowed_sides = [side] if side in ("LONG", "SHORT") else []

        try:
            max_allocation_pct = float(allocation_pct) if allocation_pct not in (None, "", "N/A") else None
        except Exception:
            max_allocation_pct = None

        return {
            "entry_conditions": entry_conditions,
            "invalidation_conditions": invalidation_conditions,
            "allowed_sides": allowed_sides,
            "max_allocation_pct": max_allocation_pct if max_allocation_pct and max_allocation_pct > 0 else None,
            "strategy_version": "daily_playbook_v1",
        }

    def _ensure_playbook_fallback(self, final: Dict, market_data: Dict, mode: TradingMode) -> None:
        scenario_engine = market_data.get("scenario_engine", {}) if isinstance(market_data, dict) else {}
        active_setup = scenario_engine.get("active_setup", {}) if isinstance(scenario_engine, dict) else {}
        fallback = self._fallback_playbook_from_active_setup(
            active_setup if isinstance(active_setup, dict) else {},
            allocation_pct=final.get("allocation_pct"),
        )
        if not fallback:
            return

        monitoring = final.get("monitoring_playbook")
        if not self._has_meaningful_playbook(monitoring):
            final["monitoring_playbook"] = deepcopy(fallback)

        dual_plan = final.get("daily_dual_plan")
        if not isinstance(dual_plan, dict):
            dual_plan = {}

        lane_key = "swing"
        canonical_lane_key = f"{lane_key}_plan"
        primary_lane = dual_plan.get(canonical_lane_key) or dual_plan.get(lane_key)
        if not self._has_meaningful_playbook(primary_lane):
            primary_lane = deepcopy(
                final.get("monitoring_playbook")
                if self._has_meaningful_playbook(final.get("monitoring_playbook"))
                else fallback
            )
            dual_plan[canonical_lane_key] = primary_lane
            dual_plan[lane_key] = deepcopy(primary_lane)
        else:
            dual_plan[canonical_lane_key] = deepcopy(primary_lane)
            dual_plan[lane_key] = deepcopy(primary_lane)

        other_lane = "position" if lane_key == "swing" else "swing"
        other_canonical_lane_key = f"{other_lane}_plan"
        other_lane_plan = dual_plan.get(other_canonical_lane_key) or dual_plan.get(other_lane)
        if not self._has_meaningful_playbook(other_lane_plan):
            other_lane_plan = deepcopy(dual_plan.get(canonical_lane_key) or fallback)
        dual_plan[other_canonical_lane_key] = deepcopy(other_lane_plan)
        dual_plan[other_lane] = deepcopy(other_lane_plan)

        final["daily_dual_plan"] = dual_plan

    @staticmethod
    def _safe_float(value, default: Optional[float] = None) -> Optional[float]:
        try:
            if value in (None, "", "N/A"):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
        if value is None:
            return None
        return round(float(value), digits)

    def _primary_tf(self, mode: TradingMode) -> str:
        return "4h"

    def _effective_wallet_and_loss_per_alloc_pct(self, target_exchange: str, leverage: float) -> tuple[float, float]:
        target_exchange = str(target_exchange or "BINANCE").lower()
        leverage = max(float(leverage or 1.0), 1.0)
        try:
            from executors.paper_exchange import paper_engine

            if target_exchange == "split":
                binance_wallet = float(paper_engine.get_wallet_balance("binance_spot"))
                upbit_wallet = float(paper_engine.get_wallet_balance("upbit"))
                wallet_equity = binance_wallet + upbit_wallet
                notional_per_alloc_pct = ((binance_wallet * leverage * 0.5) + (upbit_wallet * 0.5)) * 0.01
            else:
                wallet_equity = float(paper_engine.get_wallet_balance(target_exchange))
                notional_per_alloc_pct = wallet_equity * leverage * 0.01
            if wallet_equity > 0 and notional_per_alloc_pct > 0:
                return wallet_equity, notional_per_alloc_pct
        except Exception:
            pass

        if target_exchange == "split":
            wallet_equity = settings.BINANCE_PAPER_BALANCE_USD + settings.UPBIT_PAPER_BALANCE_USD
            notional_per_alloc_pct = (
                (settings.BINANCE_PAPER_BALANCE_USD * leverage * 0.5)
                + (settings.UPBIT_PAPER_BALANCE_USD * 0.5)
            ) * 0.01
            return wallet_equity, notional_per_alloc_pct
        if target_exchange in ("binance", "binance_spot"):
            wallet_equity = settings.BINANCE_PAPER_BALANCE_USD
        else:
            wallet_equity = settings.UPBIT_PAPER_BALANCE_USD
        return wallet_equity, wallet_equity * leverage * 0.01

    def _find_confluence(self, market_data: Dict, entry_price: float) -> Dict:
        tolerance_pct = float(getattr(settings, "POLICY_ZONE_TOLERANCE_PCT", 0.8))
        tolerance = abs(entry_price) * tolerance_pct / 100.0
        matched_sources = set()
        nearest_zone = None

        for zone in market_data.get("confluence_zones", []) or []:
            low = self._safe_float(zone.get("price_low"))
            high = self._safe_float(zone.get("price_high"))
            if low is None or high is None:
                continue
            if (low - tolerance) <= entry_price <= (high + tolerance):
                nearest_zone = zone
                for source in zone.get("sources", []) or []:
                    src = str(source)
                    if src.startswith("swing_"):
                        matched_sources.add("sr")
                    elif src.startswith("diag_"):
                        matched_sources.add("diagonal")
                    elif src.startswith("fib_"):
                        matched_sources.add("fib")

        fvg_match = False
        for _, gaps in (market_data.get("fvg", {}) or {}).items():
            for gap in gaps or []:
                gap_low = self._safe_float(gap.get("gap_low"))
                gap_high = self._safe_float(gap.get("gap_high"))
                if gap_low is None or gap_high is None:
                    continue
                if (gap_low - tolerance) <= entry_price <= (gap_high + tolerance):
                    fvg_match = True
                    matched_sources.add("fvg")
                    break
            if fvg_match:
                break

        # Order Block confluence check (unmitigated OB zone contains entry price)
        ob_match = False
        for _, obs in (market_data.get("order_blocks", {}) or {}).items():
            for ob in obs or []:
                if ob.get("fully_mitigated"):
                    continue
                ob_low  = self._safe_float(ob.get("bottom"))
                ob_high = self._safe_float(ob.get("top"))
                if ob_low is None or ob_high is None:
                    continue
                if (ob_low - tolerance) <= entry_price <= (ob_high + tolerance):
                    ob_match = True
                    matched_sources.add("order_block")
                    break
            if ob_match:
                break

        return {
            "passed": len(matched_sources) >= 2,
            "matched_sources": sorted(matched_sources),
            "zone": nearest_zone,
        }

    def _structure_stop(self, market_data: Dict, direction: str, entry_price: float, mode: TradingMode) -> Dict:
        primary_tf = self._primary_tf(mode)
        swing_levels = (market_data.get("swing_levels", {}) or {}).get(primary_tf, {}) or {}
        structure = market_data.get("structure", {}) or {}
        tf_data = (market_data.get("timeframes", {}) or {}).get(primary_tf, {}) or {}
        atr = self._safe_float(tf_data.get("atr"), 0.0) or 0.0
        atr_buffer = atr * float(getattr(settings, "POLICY_ATR_BUFFER_MULTIPLIER", 0.5))

        candidates = []
        if direction == "LONG":
            nearest_support = self._safe_float(swing_levels.get("nearest_support"))
            diag_support = self._safe_float((structure.get(f"support_{primary_tf}") or {}).get("support_price"))
            if nearest_support is not None and nearest_support < entry_price:
                candidates.append(("swing_support", nearest_support))
            if diag_support is not None and diag_support < entry_price:
                candidates.append(("diagonal_support", diag_support))
            for gap in (market_data.get("fvg", {}) or {}).get(primary_tf, []) or []:
                if str(gap.get("type")) == "bullish":
                    gap_low = self._safe_float(gap.get("gap_low"))
                    if gap_low is not None and gap_low < entry_price:
                        candidates.append(("bullish_fvg", gap_low))
            # Bullish OB bottom as stop anchor (stop goes below OB zone)
            for ob in (market_data.get("order_blocks", {}) or {}).get(primary_tf, []) or []:
                if str(ob.get("type")) == "BULLISH" and not ob.get("fully_mitigated"):
                    ob_bottom = self._safe_float(ob.get("bottom"))
                    if ob_bottom is not None and ob_bottom < entry_price:
                        candidates.append(("bullish_ob", ob_bottom))
            if candidates:
                basis, base_price = max(candidates, key=lambda item: item[1])
                stop_price = max(0.0, base_price - atr_buffer)
                return {
                    "price": stop_price,
                    "basis": basis,
                    "atr": atr,
                    "atr_buffer": atr_buffer,
                }
        else:
            nearest_resistance = self._safe_float(swing_levels.get("nearest_resistance"))
            diag_resistance = self._safe_float((structure.get(f"resistance_{primary_tf}") or {}).get("resistance_price"))
            if nearest_resistance is not None and nearest_resistance > entry_price:
                candidates.append(("swing_resistance", nearest_resistance))
            if diag_resistance is not None and diag_resistance > entry_price:
                candidates.append(("diagonal_resistance", diag_resistance))
            for gap in (market_data.get("fvg", {}) or {}).get(primary_tf, []) or []:
                if str(gap.get("type")) == "bearish":
                    gap_high = self._safe_float(gap.get("gap_high"))
                    if gap_high is not None and gap_high > entry_price:
                        candidates.append(("bearish_fvg", gap_high))
            # Bearish OB top as stop anchor (stop goes above OB zone)
            for ob in (market_data.get("order_blocks", {}) or {}).get(primary_tf, []) or []:
                if str(ob.get("type")) == "BEARISH" and not ob.get("fully_mitigated"):
                    ob_top = self._safe_float(ob.get("top"))
                    if ob_top is not None and ob_top > entry_price:
                        candidates.append(("bearish_ob", ob_top))
            if candidates:
                basis, base_price = min(candidates, key=lambda item: item[1])
                stop_price = base_price + atr_buffer
                return {
                    "price": stop_price,
                    "basis": basis,
                    "atr": atr,
                    "atr_buffer": atr_buffer,
                }

        return {
            "price": None,
            "basis": "",
            "atr": atr,
            "atr_buffer": atr_buffer,
        }

    def _compute_rr(self, direction: str, entry_price: float, stop_price: float, take_profit: Optional[float]) -> Dict:
        if entry_price <= 0 or stop_price <= 0:
            return {"risk_per_unit": None, "reward_per_unit": None, "rr": 0.0}

        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            return {"risk_per_unit": risk_per_unit, "reward_per_unit": None, "rr": 0.0}

        if take_profit is None or take_profit <= 0:
            take_profit = (
                entry_price + (risk_per_unit * float(getattr(settings, "POLICY_TP1_R_MULTIPLE", 2.0)))
                if direction == "LONG"
                else entry_price - (risk_per_unit * float(getattr(settings, "POLICY_TP1_R_MULTIPLE", 2.0)))
            )
        reward_per_unit = abs(take_profit - entry_price)
        rr = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0
        return {
            "risk_per_unit": risk_per_unit,
            "reward_per_unit": reward_per_unit,
            "rr": rr,
            "take_profit": take_profit,
        }

    def _allocation_pct(self, target_exchange: str, leverage: float, entry_price: float, stop_price: float) -> Dict:
        wallet_equity, loss_per_alloc_pct_notional = self._effective_wallet_and_loss_per_alloc_pct(target_exchange, leverage)
        risk_budget_usd = wallet_equity * (float(getattr(settings, "POLICY_MAX_RISK_PER_TRADE_PCT", 1.0)) / 100.0)
        stop_distance_pct = abs(entry_price - stop_price) / entry_price if entry_price > 0 else 0.0
        loss_per_alloc_pct = loss_per_alloc_pct_notional * stop_distance_pct
        allocation_pct = 0.0
        if loss_per_alloc_pct > 0:
            allocation_pct = min(
                float(getattr(settings, "POLICY_MAX_SPLIT_ALLOCATION_PCT", 100.0)),
                (risk_budget_usd / loss_per_alloc_pct),
            )
        return {
            "wallet_equity": wallet_equity,
            "risk_budget_usd": risk_budget_usd,
            "stop_distance_pct": stop_distance_pct * 100.0,
            "allocation_pct": allocation_pct,
        }

    def enforce(
        self,
        decision: Dict,
        market_data: Dict,
        mode: TradingMode,
        raw_funding: Optional[dict] = None,
        cvd_df=None,
        liq_df=None,
        playbook_context: Optional[Dict] = None,
    ) -> Dict:
        final = dict(decision or {})
        direction = str(final.get("decision", "HOLD")).upper()
        reasoning = final.get("reasoning")
        if not isinstance(reasoning, dict):
            reasoning = {"final_logic": str(reasoning or "")}
        reasoning.setdefault("final_logic", "")

        self._ensure_playbook_fallback(final, market_data, mode)

        policy = {
            "status": "SKIPPED",
            "reasons": [],
            "matched_sources": [],
            "flow_confirmed": False,
            "flow_signals": [],
            "playbook_guard": {
                "status": "SKIPPED",
                "reasons": [],
            },
        }
        final["policy_checks"] = policy
        final["tp1_exit_pct"] = float(getattr(settings, "POLICY_TP1_EXIT_PCT", 50.0))
        final["tp1_r_multiple"] = float(getattr(settings, "POLICY_TP1_R_MULTIPLE", 2.0))

        if direction not in ("LONG", "SHORT"):
            policy["status"] = "NO_TRADE"
            final["reasoning"] = reasoning
            return final

        entry_price = self._safe_float(final.get("entry_price"), self._safe_float(market_data.get("current_price"), 0.0)) or 0.0
        if entry_price <= 0:
            final["decision"] = "HOLD"
            final["allocation_pct"] = 0
            final["leverage"] = 1
            policy["status"] = "VETO"
            policy["reasons"].append("Missing valid entry price.")
            reasoning["final_logic"] = "[POLICY VETO] Missing valid entry price. " + reasoning.get("final_logic", "")
            final["reasoning"] = reasoning
            return final

        playbook_guard = evaluate_playbook_consistency(final, playbook_context)
        policy["playbook_guard"] = playbook_guard
        if playbook_guard.get("status") == "VETO":
            final["decision"] = "HOLD"
            final["allocation_pct"] = 0
            final["leverage"] = 1
            policy["status"] = "VETO"
            policy["reasons"].extend(playbook_guard.get("reasons", []))
            reasoning["final_logic"] = (
                f"[PLAYBOOK VETO] {' '.join(playbook_guard.get('reasons', []))} "
                + reasoning.get("final_logic", "")
            )
            final["reasoning"] = reasoning
            return final

        confluence = self._find_confluence(market_data, entry_price)
        policy["matched_sources"] = confluence.get("matched_sources", [])
        if not confluence.get("passed"):
            policy["reasons"].append("Confluence rule failed (<2 structural sources).")

        scenario_engine = market_data.get("scenario_engine", {}) or {}
        active_setup = scenario_engine.get("active_setup", {}) or {}
        active_side = str(active_setup.get("side", "")).upper()
        if active_side and active_side != direction:
            policy["reasons"].append(f"Scenario side mismatch: active_setup={active_side}, decision={direction}.")
        if active_setup and active_setup.get("invalidation") in (None, "", "N/A"):
            policy["reasons"].append("Scenario invalidation missing.")

        flow = flow_confirm_engine.evaluate(direction=direction, raw_funding=raw_funding, cvd_df=cvd_df, liq_df=liq_df)
        policy["flow_confirmed"] = bool(flow.get("confirmed"))
        policy["flow_signals"] = flow.get("matched_signals", [])
        policy["flow_metrics"] = flow.get("metrics", {})
        if len(flow.get("matched_signals", [])) < int(getattr(settings, "POLICY_REQUIRE_CONFIRMATION_SIGNALS", 1)):
            policy["reasons"].append("Flow confirmation rule failed.")

        stop_info = self._structure_stop(market_data, direction, entry_price, mode)
        stop_price = self._safe_float(stop_info.get("price"))
        if stop_price is None or stop_price <= 0:
            policy["reasons"].append("No structural stop identified.")
        else:
            final["stop_loss"] = round(stop_price, 6)
            policy["stop_basis"] = stop_info.get("basis", "")
            policy["atr"] = self._round_or_none(stop_info.get("atr"))
            policy["atr_buffer"] = self._round_or_none(stop_info.get("atr_buffer"))

        take_profit = self._safe_float(final.get("take_profit"))
        rr_info = self._compute_rr(direction, entry_price, stop_price or 0.0, take_profit)
        rr = float(rr_info.get("rr") or 0.0)
        if rr_info.get("take_profit") is not None:
            final["take_profit"] = round(float(rr_info["take_profit"]), 6)
        min_rr = float(getattr(settings, "POLICY_MIN_RR", 2.0))
        policy["rr"] = round(rr, 4)
        if rr < min_rr:
            policy["reasons"].append(f"RR {rr:.2f} below policy minimum {min_rr:.2f}.")

        if not policy["reasons"]:
            allocation = self._allocation_pct(
                target_exchange=str(final.get("target_exchange", "BINANCE")),
                leverage=self._safe_float(final.get("leverage"), 1.0) or 1.0,
                entry_price=entry_price,
                stop_price=stop_price or entry_price,
            )
            policy.update({
                "wallet_equity": self._round_or_none(allocation.get("wallet_equity"), 2),
                "risk_budget_usd": self._round_or_none(allocation.get("risk_budget_usd"), 2),
                "stop_distance_pct": self._round_or_none(allocation.get("stop_distance_pct")),
            })
            final["allocation_pct"] = round(float(allocation.get("allocation_pct") or 0.0), 4)
            risk_per_unit = self._safe_float(rr_info.get("risk_per_unit"), 0.0) or 0.0
            tp1_price = (
                entry_price + (risk_per_unit * float(getattr(settings, "POLICY_TP1_R_MULTIPLE", 2.0)))
                if direction == "LONG"
                else entry_price - (risk_per_unit * float(getattr(settings, "POLICY_TP1_R_MULTIPLE", 2.0)))
            )
            final["tp1_price"] = round(tp1_price, 6)
            if isinstance(active_setup, dict) and active_setup:
                final["split_entry_plan"] = active_setup.get("split_entries", [])
                final["breakeven_rule"] = active_setup.get("breakeven_rule", "")
                final["scenario_plan_summary"] = {
                    "trigger": active_setup.get("trigger"),
                    "entry_zone_low": active_setup.get("entry_zone_low"),
                    "entry_zone_high": active_setup.get("entry_zone_high"),
                    "invalidation": active_setup.get("invalidation"),
                    "tp1": active_setup.get("tp1"),
                    "tp2": active_setup.get("tp2"),
                }
            final["policy_checks"] = policy
            policy["status"] = "APPROVED"
            reasoning["final_logic"] = (
                f"[POLICY PASS] confluence={','.join(policy['matched_sources'])} "
                f"flow={','.join(policy['flow_signals']) or 'none'} "
                f"stop={policy.get('stop_basis', 'n/a')} rr={rr:.2f}. "
                + reasoning.get("final_logic", "")
            )
        else:
            final["decision"] = "HOLD"
            final["allocation_pct"] = 0
            final["leverage"] = 1
            policy["status"] = "VETO"
            reasoning["final_logic"] = (
                f"[POLICY VETO] {' '.join(policy['reasons'])} " + reasoning.get("final_logic", "")
            )

        final["reasoning"] = reasoning
        final["policy_checks"] = policy
        return final


policy_engine = PolicyEngine()
