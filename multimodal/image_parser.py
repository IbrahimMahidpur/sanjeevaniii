"""
image_parser.py — Medical image analysis using llava:7b
"""
import base64
import httpx
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "llava:7b")


async def parse_image(image_bytes: bytes, filename: str = "") -> str:
    """Analyze medical image using llava:7b."""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """You are a medical image analyst. Analyze this medical image carefully and describe:
1. Type of image (X-ray, MRI, CT scan, prescription, lab report, etc.)
2. Visible anatomical structures or content
3. Any abnormalities, findings, or notable observations
4. Any text, values, or measurements visible
5. Clinical significance

Use proper medical terminology. Note: Professional review always required."""

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
    except Exception as e:
        return f"Image analysis failed: {str(e)}"