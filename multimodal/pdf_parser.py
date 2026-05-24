"""
pdf_parser.py — PDF parsing using PyMuPDF + llava:7b for scanned pages
"""
import base64
import httpx
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "llava:7b")


async def parse_pdf(pdf_bytes: bytes) -> str:
    """Parse PDF — digital text direct, scanned pages via llava:7b."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text().strip()

            if len(text) > 50:
                # Digital text — extract directly
                all_text.append(f"**Page {page_num + 1}:**\n{text}")
            else:
                # Scanned page — use llava vision
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                payload = {
                    "model": VISION_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Extract all text and medical information from this document. Include all values, lab results, medications, diagnoses, and dates.",
                            "images": [img_b64],
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
                        data = response.json()
                        extracted = data["message"]["content"]
                        all_text.append(f"**Page {page_num + 1} (scanned):**\n{extracted}")
                except Exception as e:
                    all_text.append(f"**Page {page_num + 1}:** Vision failed: {e}")

        doc.close()
        return "\n\n".join(all_text) if all_text else "No content extracted."

    except ImportError:
        return "PyMuPDF not installed. Run: pip install PyMuPDF"
    except Exception as e:
        return f"PDF parsing failed: {str(e)}"