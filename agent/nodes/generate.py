"""
generate.py — High quality medical response generation using Ollama
Place at: agent/nodes/generate.py
"""

import os
import json
import httpx
from agent.state import MedicalState
from rag.prompt_builder import build_medical_prompt

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "meditron")
TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.3"))   # Lower = more accurate
MAX_TOKENS      = int(os.getenv("LLM_MAX_TOKENS", "2048"))     # More tokens = fuller response
TIMEOUT         = 120


# ---------------------------------------------------------------------------
# System prompt — the most important part for quality
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Sanjeevani, an expert medical AI assistant trained on comprehensive medical literature.

## Your Identity
- You are a knowledgeable, compassionate medical information assistant
- You provide evidence-based, accurate, and well-structured medical information
- You communicate clearly and professionally

## Response Quality Rules — STRICTLY FOLLOW THESE

### Language & Writing
1. Write ALL words COMPLETELY — never split words with spaces (write "Fever" not "F ever", "Chills" not "Ch ills")
2. Use proper English spelling and grammar throughout
3. Never abbreviate mid-word or break words across tokens
4. Write numbers clearly: "500 mg" not "5 00mg"

### Structure — ALWAYS use this format for medical questions:
```
## [Topic Title]

Brief 1-2 sentence overview.

### [Section 1 - e.g., Common Symptoms / Overview / Causes]
- **Point 1:** Clear explanation
- **Point 2:** Clear explanation

### [Section 2 - e.g., Warning Signs / Diagnosis / Treatment]
- **Point 1:** Clear explanation

### When to Seek Immediate Medical Care
[Clear guidance on emergencies]
```

### Content Quality
1. Be SPECIFIC — include actual values, timeframes, percentages where relevant
   - Good: "Fever above 38.5°C (101.3°F) lasting more than 3 days"
   - Bad: "High fever for a long time"
2. Use BOLD for important terms: **malaria**, **Plasmodium falciparum**
3. Include BOTH common names and medical terms: "jaundice (yellowing of skin/eyes)"
4. Mention severity levels: mild, moderate, severe
5. Include relevant statistics when known: "affects 200+ million people annually"
6. For drug information: always include dosage ranges, contraindications
7. For emergencies: ALWAYS start with "⚠️ EMERGENCY" and advise calling emergency services

### What to NEVER do
- Never split words: "F ever", "Ch ills", "Ab dominal" are WRONG
- Never say "I cannot provide medical advice" — you CAN provide medical information
- Never give vague answers — be specific and educational
- Never omit the disclaimer

### Disclaimer — ALWAYS end with:
---
> ⚕️ **Medical Disclaimer:** This information is for educational purposes only. Always consult a qualified healthcare provider for diagnosis, treatment, and personalized medical advice. In emergencies, call emergency services immediately.
"""


# ---------------------------------------------------------------------------
# Intent-specific prompt templates
# ---------------------------------------------------------------------------

INTENT_PROMPTS = {
    "symptom_check": """
Provide a comprehensive, well-structured response about these symptoms covering:
1. **Overview** — What these symptoms may indicate
2. **Common Causes** — List with brief explanations
3. **Associated Symptoms** — What else to watch for
4. **Severity Indicators** — When it's mild vs serious
5. **When to Seek Care** — Specific red flags requiring immediate attention
6. **General Guidance** — Safe, general advice

Be thorough and specific. Use proper medical terminology with plain-language explanations.
""",

    "drug_info": """
Provide comprehensive drug information covering:
1. **Drug Overview** — Class, mechanism of action
2. **Indications** — What conditions it treats
3. **Dosage** — Standard adult/pediatric doses, frequency
4. **Side Effects** — Common (>10%), uncommon (1-10%), rare but serious (<1%)
5. **Contraindications** — Who should NOT take it
6. **Drug Interactions** — Major interactions to avoid
7. **Special Precautions** — Pregnancy, elderly, kidney/liver disease

Include specific numbers and evidence-based information.
""",

    "lab_report": """
Explain the lab results comprehensively:
1. **What This Test Measures** — Purpose and significance
2. **Normal Reference Ranges** — With units
3. **Interpretation** — What different values mean
4. **Clinical Significance** — What abnormal results may indicate
5. **Next Steps** — What typically follows abnormal results
6. **Limitations** — What can affect accuracy

Be precise with numbers and reference ranges.
""",

    "emergency": """
⚠️ THIS APPEARS TO BE AN EMERGENCY SITUATION.

Start your response with:
"⚠️ EMERGENCY: Call emergency services (102/112) IMMEDIATELY."

Then provide:
1. **Immediate Actions** — What to do RIGHT NOW while waiting for help
2. **What NOT to Do** — Common mistakes that worsen the situation
3. **Signs of Severity** — What to monitor
4. **Information for Emergency Responders** — What to tell them

Be direct, clear, and concise. Every second matters.
""",

    "general": """
Provide a thorough, well-structured medical response covering all relevant aspects.
Use headers, bullet points, and bold text for clarity.
Include specific details, not vague generalizations.
"""
}


async def generate_node(state: MedicalState) -> MedicalState:
    """Generate high-quality medical response using Ollama."""

    intent = state.get("intent", "general")

    # Build RAG-enhanced prompt
    rag_context = ""
    retrieved_docs = state.get("retrieved_docs", [])
    if retrieved_docs:
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):
            text = doc.get("text", doc.get("content", ""))
            if text and len(text) > 50:
                context_parts.append(f"[Medical Reference {i}]:\n{text[:600]}")
        if context_parts:
            rag_context = "\n\nRelevant Medical References:\n" + "\n\n".join(context_parts)

    multimodal = state.get("multimodal_context", "")
    multimodal_section = f"\n\nAnalysis of uploaded medical file:\n{multimodal}" if multimodal else ""

    # Get intent-specific instructions
    intent_instruction = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["general"])

    # Build the full user prompt
    user_prompt = f"""{rag_context}{multimodal_section}

Patient Question: {state['query']}

{intent_instruction}

Remember: Write all words completely without splitting them. Use proper markdown formatting."""

    # Build messages with conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add last 3 turns of history for context
    history = state.get("messages", [])[-6:]
    for msg in history:
        if hasattr(msg, "type") and hasattr(msg, "content"):
            role = "user" if msg.type == "human" else "assistant"
            messages.append({"role": role, "content": msg.content})

    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
            "top_p": 0.9,
            "repeat_penalty": 1.1,      # Reduces repetition
            "top_k": 40,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            generated = data["message"]["content"]

    except httpx.TimeoutException:
        generated = (
            "## Response Timeout\n\n"
            "The response is taking longer than expected. Please try again.\n\n"
            "If you have a medical emergency, call emergency services immediately."
        )
    except Exception as e:
        generated = (
            "## Unable to Generate Response\n\n"
            f"An error occurred. Please try again.\n\n"
            "For medical emergencies, call emergency services immediately."
        )

    return {**state, "generated_response": generated}


async def generate_streaming_node(state: dict):
    """Streaming version for SSE endpoint."""
    intent = state.get("intent", "general")

    rag_context = ""
    retrieved_docs = state.get("retrieved_docs", [])
    if retrieved_docs:
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):
            text = doc.get("text", doc.get("content", ""))
            if text and len(text) > 50:
                context_parts.append(f"[Medical Reference {i}]:\n{text[:600]}")
        if context_parts:
            rag_context = "\n\nRelevant Medical References:\n" + "\n\n".join(context_parts)

    intent_instruction = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["general"])

    user_prompt = f"""{rag_context}

Patient Question: {state['query']}

{intent_instruction}

Remember: Write all words completely without splitting them."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "top_k": 40,
        },
    }

    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
