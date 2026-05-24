"""MedicalState TypedDict definition placeholder.

Defines the fields used by the LangGraph state.
"""

from typing import TypedDict, List, Dict, Optional

class MedicalState(TypedDict, total=False):
    messages: List[Dict]
    query: str
    intent: str
    retrieved_docs: List[Dict]
    generated_response: str
    validated_response: str
    user_language: str
    multimodal_context: Optional[str]
    session_id: str
    error: Optional[str]
