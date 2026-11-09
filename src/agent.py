import json
import os
import shutil
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from mock_llm import chat, reset
from rag import retrieve

DB_PATH = "data/instamart.db"
LEDGER_PATH = "outputs/refund_ledger.jsonl"


# ---------------------------------------------------------------------
# Task 5: the five tools
# ---------------------------------------------------------------------
def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
        (order_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1],
            "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    # Refund policy: delivered in MORE than 45 minutes -> full refund of the order amount.
    if delivery_minutes is None or delivery_minutes <= 45:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}


def _write_ledger(tool_name, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
             "tool": tool_name, "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _write_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _write_ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order",
     "description": "Look up an order in the database by its order id.",
     "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string", "description": "e.g. ORD-100048"}},
                    "required": ["order_id"]}},
    {"name": "search_policy",
     "description": "Search the company policy documents and return the best matching passages.",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string", "description": "what to look up"}},
                    "required": ["query"]}},
    {"name": "calculate_refund",
     "description": "Compute the refund in INR from the order amount and the delivery time in minutes.",
     "parameters": {"type": "object",
                    "properties": {"order_amount": {"type": "number"},
                                   "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund",
     "description": "Issue a refund of up to INR 1000 and record it in the refund ledger.",
     "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string"},
                                   "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human",
     "description": "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
     "parameters": {"type": "object",
                    "properties": {"order_id": {"type": "string"},
                                   "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# ---------------------------------------------------------------------
# Tasks 6, 7 and 8: the ReAct loop, three safety limits, checkpoints and resume
# ---------------------------------------------------------------------
AGENT_SYSTEM = ("You are a refund AGENT for Instamart. Use the tools to look up the order, "
                "check the policy, compute the refund, and then issue it or escalate it to a human. "
                "Finish with a one-line answer that starts with REFUND, ESCALATE, DENIED or NOT_FOUND, "
                "followed by a colon.")

STATE_DIR = "state"


class SimulatedCrash(Exception):
    """Raised on purpose to demonstrate that a run can be resumed."""


def _fallback(status, state):
    tools_done = ", ".join(s["tool"] for s in state["steps"]) or "no tools"
    final_answer = (f"{status}: stopped after {len(state['steps'])} step(s) ({tools_done}). "
                    "Handing this request over to a human agent with the steps completed so far.")
    return _result(status, final_answer, state)


def _result(status, final_answer, state):
    return {"run_id": state["run_id"], "status": status, "final_answer": final_answer,
            "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "tools_executed": state["tools_executed"]}


def _save_checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    checkpoint = {"run_id": state["run_id"], "scenario_id": state["scenario_id"],
                  "messages": state["messages"], "tokens_used": state["tokens_used"],
                  "llm_calls": state["llm_calls"], "step_number": len(state["steps"]),
                  "tools_executed": state["tools_executed"], "steps": state["steps"],
                  "max_iterations": state["max_iterations"], "token_limit": state["token_limit"],
                  "time_limit_seconds": state["time_limit_seconds"]}
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(checkpoint, f)


def _loop(state, crash_after_tools=None):
    start = time.monotonic()
    while state["llm_calls"] < state["max_iterations"]:
        # limit (c): wall-clock time, checked at the top of every iteration
        if time.monotonic() - start >= state["time_limit_seconds"]:
            return _fallback("FALLBACK_TIMEOUT", state)

        response = chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens

        # limit (b): tokens, checked BEFORE the requested tool is executed
        if state["tokens_used"] > state["token_limit"]:
            return _fallback("FALLBACK_TOKEN_LIMIT", state)

        message = response.choices[0].message
        if not message.tool_calls:                       # no tool requested: this is the final answer
            final_answer = message.content
            return _result(final_answer.split(":", 1)[0].strip(), final_answer, state)

        # the model asked for tool(s): record the request, run them, append the observations
        requested = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                     for tc in message.tool_calls]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": requested})
        for tc in requested:
            result = TOOL_MAP[tc["name"]](**tc["arguments"])
            state["messages"].append({"role": "tool", "tool_call_id": tc["id"],
                                      "name": tc["name"], "content": json.dumps(result)})
            state["steps"].append({"tool": tc["name"], "arguments": tc["arguments"],
                                   "observation": result})
            state["tools_executed"].append(tc["name"])
            _save_checkpoint(state)                      # Task 8: after every tool observation
            if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
                raise SimulatedCrash(state["run_id"])

    # limit (a): max_iterations model calls without a final answer
    return _fallback("FALLBACK_MAX_ITERATIONS", state)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, crash_after_tools=None):
    state = {"run_id": f"{scenario_id}-{uuid.uuid4().hex[:8]}", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": AGENT_SYSTEM},
                          {"role": "user", "content": user_message}],
             "steps": [], "llm_calls": 0, "tokens_used": 0, "tools_executed": [],
             "max_iterations": max_iterations, "token_limit": token_limit,
             "time_limit_seconds": time_limit_seconds}
    return _loop(state, crash_after_tools)


def resume_agent(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        checkpoint = json.load(f)
    state = {key: checkpoint[key] for key in
             ("run_id", "scenario_id", "messages", "steps", "llm_calls", "tokens_used",
              "tools_executed", "max_iterations", "token_limit", "time_limit_seconds")}
    return _loop(state)


def refund_amount(run):
    for step in reversed(run["steps"]):
        if step["tool"] in ("issue_refund", "escalate_to_human"):
            return step["observation"]["refund_inr"]
    return 0.0


def run_all_scenarios():
    for path in ("outputs/agent_runs.jsonl", LEDGER_PATH):      # start from a clean slate
        if os.path.exists(path):
            os.remove(path)
    os.makedirs("outputs", exist_ok=True)
    shutil.rmtree(STATE_DIR, ignore_errors=True)                 # clear old checkpoints
    reset()

    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    rows = []
    for _, s in scenarios.iterrows():
        run = run_agent(s["message"], s["scenario_id"])
        record = {"scenario_id": s["scenario_id"], "status": run["status"],
                  "final_answer": run["final_answer"], "llm_calls": run["llm_calls"],
                  "tokens_used": run["tokens_used"], "steps": run["steps"]}
        with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        amount = refund_amount(run)
        rows.append({"scenario_id": s["scenario_id"], "status": run["status"],
                     "refund_inr": amount, "llm_calls": run["llm_calls"],
                     "tokens_used": run["tokens_used"]})
        expected = s["expected_refund_inr"]
        assert abs(amount - expected) <= 0.01, f"{s['scenario_id']}: refund {amount} but expected {expected}"
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return table


def run_demos():
    global LEDGER_PATH
    s1 = pd.read_csv("data/raw/agent_scenarios.csv").set_index("scenario_id").loc["S1", "message"]

    # Task 7 demos (no tool is executed after the limit is hit, so the ledger is untouched)
    token_demo = run_agent(s1, "S1", token_limit=200)
    print("S1 with token_limit=200:", token_demo["status"],
          "| model calls:", token_demo["llm_calls"], "| tokens:", token_demo["tokens_used"],
          "| steps done:", len(token_demo["steps"]))
    time_demo = run_agent(s1, "S1", time_limit_seconds=0)
    print("S1 with time_limit_seconds=0:", time_demo["status"],
          "| model calls:", time_demo["llm_calls"], "| steps done:", len(time_demo["steps"]))

    # Task 8 demo: crash right after the second tool, then resume.
    # The demo has its own ledger file, so the main ledger never shows a second refund for S1.
    main_ledger = LEDGER_PATH
    LEDGER_PATH = "outputs/refund_ledger_resume_demo.jsonl"
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)
    try:
        run_id = None
        try:
            run_agent(s1, "S1", crash_after_tools=2)
        except SimulatedCrash as crash:
            run_id = str(crash)
            print("simulated crash after the second tool; run id:", run_id)
        with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
            saved = json.load(f)
        print("checkpoint has: tools_executed =", saved["tools_executed"],
              "| step_number =", saved["step_number"], "| tokens_used =", saved["tokens_used"])
        resumed = resume_agent(run_id)
    finally:
        LEDGER_PATH = main_ledger
    print("after resume:", resumed["status"], refund_amount(resumed), "| model calls:", resumed["llm_calls"])
    print("tools executed across the crash and the resume:", resumed["tools_executed"])
    assert resumed["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed["status"] == "REFUND" and abs(refund_amount(resumed) - 128.25) <= 0.01


if __name__ == "__main__":
    run_all_scenarios()
    run_demos()