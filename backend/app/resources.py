"""Fixed wellness resources: general self-care ideas and where to get help.

This module has no web code and no database. FastAPI (main.py) calls it.
The content is general information, not medical advice.
"""

DISCLAIMER = (
    "These are general self-care ideas, not medical advice or treatment. "
    "MoodMentor cannot diagnose anything. If you are worried about your "
    "well-being, please talk to someone you trust or a qualified professional."
)

RESOURCES = [
    {
        "id": "slow-breathing",
        "category": "Calm down",
        "title": "Slow breathing",
        "summary": "A short exercise that can help your body settle when you feel tense.",
        "steps": [
            "Sit or lie down somewhere comfortable.",
            "Breathe in slowly through your nose for about 4 counts.",
            "Breathe out slowly through your mouth for about 6 counts.",
            "Repeat for 2 to 5 minutes. Stop if you feel dizzy.",
        ],
    },
    {
        "id": "grounding-54321",
        "category": "Calm down",
        "title": "5-4-3-2-1 grounding",
        "summary": "A way to bring your attention back to the present when thoughts race.",
        "steps": [
            "Name 5 things you can see.",
            "Name 4 things you can touch.",
            "Name 3 things you can hear.",
            "Name 2 things you can smell.",
            "Name 1 thing you can taste.",
        ],
    },
    {
        "id": "write-it-down",
        "category": "Understand your feelings",
        "title": "Write it down",
        "summary": "Putting feelings into words can make them easier to look at.",
        "steps": [
            "Set a timer for 5 minutes.",
            "Write what happened and how you feel, without worrying about spelling.",
            "Add a Wellness Check-in entry in MoodMentor if you want to track it over time.",
        ],
    },
    {
        "id": "move-your-body",
        "category": "Look after yourself",
        "title": "Move your body",
        "summary": "Light movement is a simple way to change how a heavy day feels.",
        "steps": [
            "Go for a 10-minute walk, preferably outside.",
            "Stretch your neck, shoulders and back for a few minutes.",
            "Choose something you enjoy so it is easier to repeat.",
        ],
    },
    {
        "id": "sleep-routine",
        "category": "Look after yourself",
        "title": "A steadier sleep routine",
        "summary": "Regular sleep supports mood and concentration.",
        "steps": [
            "Try to go to bed and wake up at about the same time each day.",
            "Put screens away 30 to 60 minutes before bed.",
            "Avoid caffeine late in the day.",
            "Keep your room dark, quiet and cool.",
        ],
    },
    {
        "id": "reach-out",
        "category": "Get support",
        "title": "Reach out to someone",
        "summary": "Feelings are often easier to carry when someone else knows about them.",
        "steps": [
            "Pick one person you trust, such as a friend, family member or mentor.",
            "Send a short message or ask if they have time to talk.",
            "If the feelings last for weeks or affect daily life, consider a counsellor, "
            "your college counselling service or a doctor.",
        ],
    },
]

# Check these numbers are still correct before a public release.
HELP_CONTACTS = [
    {
        "name": "Emergency services (India)",
        "contact": "112",
        "note": "If you or someone else is in immediate danger.",
    },
    {
        "name": "Tele-MANAS (India)",
        "contact": "14416 or 1-800-891-4416",
        "note": "Free mental health support, 24 hours a day, 7 days a week.",
    },
]


def get_resources():
    """Everything the Resources page needs."""
    return {
        "disclaimer": DISCLAIMER,
        "items": RESOURCES,
        "help_contacts": HELP_CONTACTS,
    }