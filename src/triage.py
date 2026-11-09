import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import re
import time

import pandas as pd

from mock_llm import est_tokens, chat, reset, TransientLLMError

# =====================================================================
# Task 1: load the tickets and apply the input guardrails
# =====================================================================
MAX_INPUT_TOKENS = 400

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
tickets["est_tokens"] = tickets["text"].map(est_tokens)

is_empty = tickets["text"].str.strip() == ""
is_long = tickets["est_tokens"] > MAX_INPUT_TOKENS

print(len(tickets), is_empty.sum(), is_long.sum(), tickets.loc[is_long, "est_tokens"].unique())
print((~is_empty & ~is_long).sum())

# =====================================================================
# Task 2: redact personal information (order matters: email, phone, order id)
# =====================================================================
PATTERNS = [
    ("EMAIL", r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
    ("PHONE", r"\b[6-9]\d{9}\b", "[PHONE]"),
    ("ORDER_ID", r"ORD-\d{6}", "[ORDER_ID]"),
]


def redact(text, counts=None):
    for name, pattern, placeholder in PATTERNS:
        text, n = re.subn(pattern, placeholder, text)
        if counts is not None:
            counts[name] += n
    return text


counts = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
tickets["redacted"] = [redact(t, counts) for t in tickets["text"]]
print(counts, sum(counts.values()))

left = sum(len(re.findall(p, t)) for t in tickets["redacted"] for _, p, _ in PATTERNS)
print(left)
assert left == 0, "personal information is still present"

# =====================================================================
# Task 3: the two prompts
# =====================================================================
ZERO_SHOT = ('CLASSIFY the support ticket into one category: billing, delivery, technical or fraud. '
             'Reply with JSON only: {"category": "...", "reason": "..."}')

FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged twice for one order. -> {"category": "billing", "reason": "double charge"}
Ticket: The rider has not arrived yet. -> {"category": "delivery", "reason": "late rider"}
Ticket: The app crashes when I log in. -> {"category": "technical", "reason": "app crash"}
Ticket: Someone used my card without permission. -> {"category": "fraud", "reason": "unauthorized use"}"""

CATEGORIES = ["billing", "delivery", "technical", "fraud"]
for prompt in (ZERO_SHOT, FEW_SHOT):
    assert "CLASSIFY" in prompt
    assert all(c in prompt for c in CATEGORIES)
assert "EXAMPLES:" not in ZERO_SHOT
assert "EXAMPLES:" in FEW_SHOT
print(est_tokens(ZERO_SHOT), est_tokens(FEW_SHOT))

# =====================================================================
# Task 4: resilient calls (retry with exponential backoff and jitter)
# =====================================================================
BASE_DELAY = 0.05
MAX_RETRIES = 3

retry_stats = {"errors": 0}
waits = []


def call_with_retry(messages, ticket_id):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return chat(messages)
        except TransientLLMError:
            retry_stats["errors"] += 1
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            low = BASE_DELAY * 2 ** attempt * 0.9
            high = BASE_DELAY * 2 ** attempt * 1.15
            assert low <= wait <= high, "wait outside the allowed jitter range"
            waits.append(wait)
            print(f"retry {attempt} for {ticket_id}: waiting {wait:.4f}s")
            time.sleep(wait)


# =====================================================================
# Task 5: validate every answer (never guess)
# =====================================================================
def validate_reply(text):
    """Return (status, category). Accept only what can be checked."""
    try:
        data = json.loads(text)
        category = data.get("category") if isinstance(data, dict) else None
        if category in CATEGORIES:
            return "classified_json", category
        return "flagged_invalid_output", None          # valid JSON, but not an allowed category
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", str(text).lower()))
        found = [c for c in CATEGORIES if c in words]
        if len(found) == 1:                              # exactly one allowed category word
            return "classified_keyword", found[0]
        return "flagged_invalid_output", None


# =====================================================================
# The pipeline: guardrails -> redaction -> model call (with retries) -> validation
# =====================================================================
def run_pipeline(system_prompt, token_budget=None):
    random.seed(42)
    reset()                                              # immediately before each full run
    retry_stats["errors"] = 0
    waits.clear()

    rows = []
    total_tokens = 0
    for i, row in tickets.iterrows():                    # file order
        status, category, tokens = None, None, 0

        if is_empty[i]:
            status = "skipped_empty"                     # never sent to the model
        elif is_long[i]:
            status = "flagged_long"                      # never sent to the model
        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"                  # no call is made
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": row["redacted"]}]
            # privacy check: no email, phone or order id in anything sent to the model
            assert not any(re.search(p, m["content"]) for m in messages for _, p, _ in PATTERNS)
            response = call_with_retry(messages, row["ticket_id"])
            tokens = response.usage.total_tokens
            total_tokens += tokens
            status, category = validate_reply(response.choices[0].message.content)

        rows.append({"ticket_id": row["ticket_id"], "customer_id": row["customer_id"],
                     "true_category": row["true_category"], "status": status,
                     "category": category, "tokens_used": tokens})
    return pd.DataFrame(rows), total_tokens


results, total = run_pipeline(ZERO_SHOT)
print(results["status"].value_counts())
print("retries:", retry_stats["errors"], "total tokens:", total)

def summarize(name, results, total):
    classified = results[results["status"].isin(["classified_json", "classified_keyword"])]
    correct = (classified["category"] == classified["true_category"]).sum()
    print(name, len(classified), correct, round(100 * correct / len(classified), 1),
          total, round(total / len(classified), 1))
    return classified


zero_results, zero_total = run_pipeline(ZERO_SHOT)
zero_classified = summarize("zero-shot", zero_results, zero_total)

few_results, few_total = run_pipeline(FEW_SHOT)
few_classified = summarize("few-shot", few_results, few_total)

print(pd.crosstab(zero_classified["true_category"], zero_classified["category"]))

segments = pd.read_csv("outputs/customer_segments.csv")
premium = dict(zip(segments["customer_id"], segments["is_premium"]))
segment_name = dict(zip(segments["customer_id"], segments["segment_name"]))

HUMAN_STATUSES = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}


def route_ticket(status, category, is_premium):
    if status in HUMAN_STATUSES:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def finish(results):
    out = results.copy()
    out["segment_name"] = out["customer_id"].map(segment_name)
    decisions = [route_ticket(s, c, bool(premium.get(cid, False)))
                 for s, c, cid in zip(out["status"], out["category"], out["customer_id"])]
    out["route"] = [d[0] for d in decisions]
    out["priority"] = [d[1] for d in decisions]
    return out[["ticket_id", "customer_id", "status", "category",
                "segment_name", "priority", "route", "tokens_used"]]

os.makedirs("outputs", exist_ok=True)

TOKEN_BUDGET_DEMO = 6000
demo_results, demo_total = run_pipeline(ZERO_SHOT, token_budget=TOKEN_BUDGET_DEMO)
demo = finish(demo_results)
demo.to_csv("outputs/triage_budget_demo.csv", index=False)
print("budget", TOKEN_BUDGET_DEMO, "tokens used", demo_total,
      "exhausted", (demo["status"] == "budget_exhausted").sum())

final_results, final_total = run_pipeline(FEW_SHOT, token_budget=100000)
final = finish(final_results)
final.to_csv("outputs/triage_results.csv", index=False)
print(len(final))
print(final["route"].value_counts())
print(final["status"].value_counts())

assert (final.loc[final["category"] == "fraud", "route"] == "security_team").all()
assert (final.loc[final["status"].isin(HUMAN_STATUSES), "route"] == "human_review").all()