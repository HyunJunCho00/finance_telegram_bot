"""
Post-Mortem Engine: The Reflexive Memory Loop (V6)
Evaluates past trades and injects them into the Vector DB (LightRAG) for future Judge retrieval.
"""
import json
import os
from datetime import datetime, timezone
from loguru import logger
from config.database import db
from agents.claude_client import claude_client
from processors.light_rag import light_rag
from pathlib import Path

# [FIX MEDIUM-16] Absolute path — works regardless of systemd WorkingDirectory
_BASE_DIR = Path(__file__).parent.parent
MEMORY_FILE_PATH = str(_BASE_DIR / "data" / "episodic_memory.jsonl")

def initialize_memory_db():
    os.makedirs(os.path.dirname(MEMORY_FILE_PATH), exist_ok=True)
    if not os.path.exists(MEMORY_FILE_PATH):
        with open(MEMORY_FILE_PATH, 'w', encoding='utf-8') as f:
            pass

def assess_trade_outcome(decision: dict, entry_price: float, current_price: float) -> str:
    """Simple heuristic to determine if the trade was successful."""
    dir = decision.get("decision", "HOLD")
    if dir == "HOLD": return "NEUTRAL"
    if dir == "LONG" and current_price >= entry_price: return "SUCCESS"
    if dir == "SHORT" and current_price <= entry_price: return "SUCCESS"
    return "FAILURE"

def generate_llm_lesson(trade_data: dict, outcome: str) -> str:
    """Uses LLM to reflect on what went right or wrong based on the original Blackboard state."""
    prompt = f"""You are the Head of Quant Research conducting a post-mortem on a completed trade case study.
TRADE DATA (Original Context):
{json.dumps(trade_data, indent=2)}
OUTCOME: {outcome}

Analyze the original reasoning, the blackboard expert insights, and the final outcome. 
Produce a structured 'CASE STUDY' for our internal Knowledge Graph:

1. SETUP: Describe the market regime and key expert signals that led to the entry.
2. OUTCOME ANALYSIS: Explain WHY the trade resulted in {outcome}. Identify the specific signal that was either correctly followed or fatally ignored.
3. FUTURE CONSTRAINT: Provide a specific, hard rule for the Judge Agent to avoid this mistake or repeat this success (e.g., "Do not LONG if OI is dropping while price is at Fibonacci 61.8 resistance").

Focus strictly on objective analysis. No filler."""

    try:
        response = claude_client.generate_response(
            system_prompt="You are a strict, objective quantitative analyst specializing in Case Study documentation.",
            user_message=prompt,
            temperature=0.2,
            max_tokens=500,
            role="macro"
        )
        return response.strip()
    except Exception as e:
        logger.error(f"Post-Mortem LLM error: {e}")
        return f"Fallback case study: The trade resulted in {outcome}."


def write_post_mortem(report: dict, current_price: float):
    """Generates a post-mortem JSON, feeds it to LLM for a lesson, and saves it to LightRAG AND JSONL."""
    initialize_memory_db()
    
    decision = report.get('final_decision', {})
    if not isinstance(decision, dict) or decision.get('decision') == 'HOLD':
        return # Skip logging HOLDS
        
    entry_price = decision.get("entry_price", current_price)
    outcome = assess_trade_outcome(decision, entry_price, current_price)
    
    # Generate the lesson using LLM reflection
    trade_summary = {
        "symbol": report.get("symbol"),
        "direction": decision.get("decision"),
        "predicted_reasoning": decision.get("reasoning", ""),
        "anomalies_detected": report.get("blackboard", {}) # Context matters for the failing/winning setup
    }
    lesson = generate_llm_lesson(trade_summary, outcome)

    memory_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": report.get("symbol", "UNKNOWN"),
        "predicted_direction": decision.get("decision"),
        "confidence": decision.get("confidence", 0),
        "outcome": outcome,
        "reasoning": decision.get("reasoning", ""),
        "lessons_learned": lesson
    }

    # 1. Fallback JSONL Storage (for immediate deterministic retrieval)
    try:
        with open(MEMORY_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(memory_entry) + "\n")
        logger.info(f"Post-Mortem logged to JSONL: {outcome} for {memory_entry['symbol']}")
    except Exception as e:
        logger.error(f"Failed to log post-mortem: {e}")
        
    # 2. Ingest into LightRAG Engine for Deep Association (V6)
    try:
        rag_text = f"Trade Post-Mortem for {memory_entry['symbol']}. Direction: {memory_entry['predicted_direction']}. Outcome was {outcome}. Lesson learned: {lesson}"
        light_rag.ingest_message(
            text=rag_text,
            channel="SYSTEM_POST_MORTEM",
            timestamp=memory_entry['timestamp']
        )
        logger.info("Post-Mortem successfully ingested into LightRAG Knowledge Graph.")
    except Exception as e:
        logger.error(f"Failed to ingest post-mortem into RAG: {e}")

def retrieve_similar_memories(current_anomalies: list, symbol: str) -> str:
    """Retrieve past trade memories semantically similar to current market conditions.

    Primary: LightRAG vector search (semantic similarity — finds structurally similar
    past setups regardless of time). Post-mortems are stored with channel=SYSTEM_POST_MORTEM.
    Fallback: JSONL time-reverse scan (used when LightRAG/Milvus unavailable).
    """
    query_text = (
        f"past trade outcome {symbol} "
        f"anomalies: {', '.join(current_anomalies) if current_anomalies else 'none'}"
    )

    # ── Primary: LightRAG semantic search ──
    try:
        result = light_rag.query(query_text, top_k=10)
        semantic_hits = [
            r for r in result.get("semantic_results", [])
            if r.get("channel") == "SYSTEM_POST_MORTEM"
               and r.get("text", "").strip()
        ][:3]

        if semantic_hits:
            mem_str = "Similar Past Trades (Semantic Match via LightRAG):\n"
            for hit in semantic_hits:
                score = hit.get("score", 0)
                text = hit.get("text", "")[:180]
                mem_str += f"- [similarity={score:.2f}] {text}\n"
            return mem_str
    except Exception as e:
        logger.warning(f"LightRAG memory search failed, falling back to JSONL: {e}")

    # ── Fallback: JSONL time-reverse scan ──
    return _retrieve_from_jsonl(symbol)


def _retrieve_from_jsonl(symbol: str) -> str:
    """JSONL fallback: returns the 3 most recent post-mortems for this symbol."""
    if not os.path.exists(MEMORY_FILE_PATH):
        return "No memory database found."

    relevant_memories = []
    try:
        with open(MEMORY_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in reversed(lines[-50:]):
            mem = json.loads(line)
            if mem.get("symbol") == symbol:
                relevant_memories.append(mem)
            if len(relevant_memories) >= 3:
                break
    except Exception as e:
        logger.error(f"Failed to read JSONL memory: {e}")

    if not relevant_memories:
        return "System running in Bootstrap mode. No historical matches found yet."

    mem_str = "Recent Past Outcomes (JSONL fallback — time-ordered, not semantic):\n"
    for mem in relevant_memories:
        reasoning = mem.get("reasoning", "")
        if isinstance(reasoning, dict):
            reasoning = reasoning.get("final_logic", str(reasoning))
        mem_str += (
            f"- [{mem['outcome']}] {mem['predicted_direction']} | "
            f"{str(reasoning)[:100]}...\n"
        )
    return mem_str
