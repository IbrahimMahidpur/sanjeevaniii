"""
validate.py — Response validation and quality enhancement
Place at: agent/nodes/validate.py
"""

import re
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.response_cleaner import clean_response
from agent.state import MedicalState


EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "seizure", "unconscious",
    "not breathing", "severe bleeding", "overdose", "poisoning",
    "anaphylaxis", "allergic reaction", "difficulty breathing",
    "can't breathe", "cannot breathe", "choking", "drowning",
    "severe burn", "head injury", "spinal injury", "suicide",
    "self harm", "severe abdominal pain", "coughing blood",
    "vomiting blood", "loss of consciousness", "paralysis",
]

EMERGENCY_BANNER = (
    "## ⚠️ EMERGENCY — Seek Immediate Medical Help\n\n"
    "**Call emergency services (102 / 112) IMMEDIATELY.**\n\n"
    "Do not wait. Do not drive yourself. Call for help now.\n\n"
    "---\n\n"
)


def is_emergency(query: str, response: str) -> bool:
    """Check if this is an emergency situation."""
    text = (query + " " + response).lower()
    return any(kw in text for kw in EMERGENCY_KEYWORDS)


def validate_response_quality(response: str) -> tuple[bool, str]:
    """
    Check response quality.
    Returns (is_good, reason_if_bad)
    """
    if not response or len(response.strip()) < 50:
        return False, "Response too short"

    if response.count("I cannot") > 2 or response.count("I'm unable") > 2:
        return False, "Response is too restrictive"

    # Check for excessive word splitting (more than 5 occurrences of single letter + space + word)
    split_word_pattern = re.findall(r'\b[A-Z]\s+[a-z]{2,}\b', response)
    if len(split_word_pattern) > 5:
        return False, f"Too many split words: {split_word_pattern[:3]}"

    return True, "OK"


async def validate_node(state: MedicalState) -> MedicalState:
    """Validate and enhance the generated response."""

    response = state.get("generated_response", "")
    intent = state.get("intent", "general")
    query = state.get("query", "")

    # Step 1: Check quality
    is_good, reason = validate_response_quality(response)
    if not is_good:
        print(f"Warning: Response quality issue — {reason}")

    # Step 2: Run full cleaning pipeline
    response = clean_response(response, intent)

    # Step 3: Add emergency banner if needed
    if intent == "emergency" or is_emergency(query, response):
        if "EMERGENCY" not in response[:100]:
            response = EMERGENCY_BANNER + response

    # Step 4: Ensure response has proper title/header
    if not response.startswith("#"):
        # Extract topic from query
        topic = query.strip().rstrip("?").title()
        response = f"## {topic}\n\n{response}"

    # Step 5: Final length check — if too short, add note
    if len(response) < 200:
        response += (
            "\n\nFor more detailed information about your specific situation, "
            "please consult a qualified healthcare provider."
        )

    return {**state, "validated_response": response}
