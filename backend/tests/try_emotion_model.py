"""Try the emotion model on its own (emotion project, Step 2).

This is an experiment, not a pass/fail test. It shows what the model
really predicts, so we can decide how to use it.

Run from the project root:
    python -m backend.tests.try_emotion_model
"""

import os
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

MODEL_NAME = "SamLowe/roberta-base-go_emotions"

# Labels we plan to rely on for the emotions we want to show.
PLANNED_LABELS = {
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "love",
    "excitement", "confusion", "embarrassment", "pride", "nervousness", "neutral",
}

SAMPLES = [
    ("Happiness", "I am so happy today! Everything is going wonderfully."),
    ("Sadness", "I feel very sad and disappointed today."),
    ("Anger", "I am furious that they ignored my request again."),
    ("Fear", "I am terrified of walking home alone in the dark."),
    ("Surprise", "Wow, I did not expect that at all! What a surprise."),
    ("Disgust", "That smell was disgusting, I nearly threw up."),
    ("Love", "I love my family so much, they mean everything to me."),
    ("Excitement", "I can't wait for the trip tomorrow, I'm so excited!"),
    ("Confusion", "I don't understand these instructions at all, what is going on?"),
    ("Embarrassment", "I tripped in front of the whole class and wanted to disappear."),
    ("Pride", "I am proud of myself for finishing my project on time."),
    ("Anxiety", "I'm nervous about tomorrow's exam and can't stop worrying."),
    ("Neutral", "The meeting starts at 10 AM tomorrow."),
    ("Negation", "I am not happy with the result."),
    ("Mixed", "I'm excited about the new job but also scared of failing."),
    ("Jealousy (not supported)", "I'm jealous of how easily she gets everything I want."),
    ("Shame (not supported)", "I am ashamed of what I did and I can't look anyone in the eye."),
    ("Loneliness (not supported)", "I feel so lonely, nobody has called me in weeks."),
    ("Contentment (not supported)", "I feel calm and content, everything is just right."),
]

print("Loading the model. The first run downloads it, which can take several minutes...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
labels = model.config.id2label
print(f"Loaded in {time.time() - start:.1f} seconds.\n")

print(f"The model has {len(labels)} labels:")
print(", ".join(sorted(labels.values())))
missing = PLANNED_LABELS - set(labels.values())
print(f"\nAll labels we plan to use are present: {not missing}")
if missing:
    print("Missing labels:", sorted(missing))


def top_emotions(text, count=3):
    """Return the highest-scoring emotions as (label, probability) pairs."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probabilities = torch.sigmoid(logits)  # multi-label: one probability per emotion
    values, indices = torch.topk(probabilities, count)
    return [(labels[int(i)], float(v)) for v, i in zip(values, indices)]


top_emotions("warm-up")  # the first call is slower, so do it once before timing

timings = []
for group, text in SAMPLES:
    began = time.time()
    results = top_emotions(text)
    timings.append(time.time() - began)
    print(f"\n=== {group} ===")
    print(f'"{text}"')
    for rank, (label, probability) in enumerate(results, start=1):
        print(f"  {rank}. {label:<15} {probability:.3f}")

print(f"\nAverage time per sentence: {1000 * sum(timings) / len(timings):.0f} ms")