"""PubMed scraper script.

Fetches abstracts for a list of predefined queries using the NCBI E-utilities API.
Outputs one JSONL file per query in the current working directory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import requests

# Base URL for the Entrez ESearch and ESummary utilities
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Default queries as per project spec
DEFAULT_QUERIES = [
    "diabetes treatment",
    "hypertension management",
    "drug interactions",
    "symptoms diagnosis",
]


def esearch(query: str, retmax: int = 20) -> List[str]:
    """Return a list of PubMed IDs for the given query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(retmax),
        "retmode": "json",
    }
    resp = requests.get(f"{BASE_URL}/esearch.fcgi", params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def esummary(pmid: str) -> dict:
    """Fetch summary information for a single PMID."""
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "json",
    }
    resp = requests.get(f"{BASE_URL}/esummary.fcgi", params=params)
    resp.raise_for_status()
    data = resp.json()
    result = data.get("result", {})
    return result.get(pmid, {})


def fetch_abstract(pmid: str) -> dict:
    """Fetch the abstract for a PMID using the efetch utility."""
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
    }
    resp = requests.get(f"{BASE_URL}/efetch.fcgi", params=params)
    resp.raise_for_status()
    # Very lightweight parsing – we look for <ArticleTitle> and <AbstractText>
    from xml.etree import ElementTree as ET
    root = ET.fromstring(resp.text)
    article = root.find('.//Article')
    if article is None:
        return {}
    title_elem = article.find('.//ArticleTitle')
    abstract_elem = article.find('.//Abstract/AbstractText')
    title = title_elem.text if title_elem is not None else ""
    abstract = abstract_elem.text if abstract_elem is not None else ""
    return {"title": title, "abstract": abstract}


def scrape_query(query: str, retmax: int = 20) -> List[dict]:
    ids = esearch(query, retmax=retmax)
    records = []
    for pmid in ids:
        summary = esummary(pmid)
        abstract_data = fetch_abstract(pmid)
        record = {
            "id": pmid,
            "title": abstract_data.get("title", summary.get("title", "")),
            "abstract": abstract_data.get("abstract", ""),
            "source": "pubmed",
        }
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts for predefined queries.")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=DEFAULT_QUERIES,
        help="Space‑separated list of queries to run.",
    )
    parser.add_argument(
        "--retmax",
        type=int,
        default=20,
        help="Maximum number of PubMed IDs per query.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where JSONL files will be written.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for query in args.queries:
        print(f"Scraping query: {query}", file=sys.stderr)
        records = scrape_query(query, retmax=args.retmax)
        out_path = args.output_dir / f"{query.replace(' ', '_')}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} records to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
