"""Tries the BERT engine on a few sentences and prints what it returns.

Run from the project root:
    python -m backend.tests.try_bert_engine
The first run downloads about 440 MB, so it can take a while.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from backend.app import bert_engine  # noqa: E402

tests = [
    "I'm excited about the trip but also really nervous.",
    "Thank you so much, you really helped me.",
    "This is the third time they cancelled on me, I'm furious.",
    "I miss her so much and I can't stop crying.",
    "ok",
]

start = time.time()
try:
    results = bert_engine.predict(tests)
except bert_engine.EngineUnavailable as exc:
    print("Engine unavailable:", exc)
    raise SystemExit(1)

print(f"Loaded and scored {len(tests)} texts in {time.time() - start:.1f}s\n")
for text, result in zip(tests, results):
    print(text)
    print("  above threshold:", result["selected"] or "none")
    print("  top 3:", [(label, round(score, 2)) for label, score in result["scores"][:3]])
    print()