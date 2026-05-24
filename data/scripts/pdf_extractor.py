"""PDF extractor script.

Extracts text from PDF files, falling back to OCR for scanned pages.
Writes JSONL records with fields: id, text, source, filename.
"""

import argparse
import json
import hashlib
import sys
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF.
    Returns concatenated text from all pages.
    """
    doc = fitz.open(str(pdf_path))
    full_text = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text()
        if len(text.strip()) < 50:
            # Fallback to OCR for low‑text pages
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)
            full_text.append(ocr_text)
        else:
            full_text.append(text)
    return "\n".join(full_text)


def hash_content(content: str) -> str:
    """Return a stable hash for the given content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def process_pdfs(input_dir: Path, output_dir: Path) -> List[dict]:
    records = []
    for pdf_file in input_dir.rglob("*.pdf"):
        try:
            text = pdf_to_text(pdf_file)
            rec = {
                "id": hash_content(text)[:16],
                "text": text,
                "source": "pdf",
                "filename": pdf_file.name,
            }
            records.append(rec)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}", file=sys.stderr)
    # Write JSONL
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "pdf_extracted.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {out_path}", file=sys.stderr)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from PDFs in a directory.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing PDF files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Directory to write output JSONL.")
    args = parser.parse_args()
    process_pdfs(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
