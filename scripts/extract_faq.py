"""
Extract FAQs from the HKPL "Ask a Librarian" page and save them as a clean CSV.

This script is used during data preparation, not during live chat.
The chatbot later uses the generated CSV for ingestion into LlamaIndex PGVectorStore.

Run from project root:
docker run --rm -it \
  -v $(pwd):/data \
  -w /data \
  python:3.11-slim \
  bash -c "
    pip install --quiet requests beautifulsoup4 &&
    python scripts/extract_faq.py
  "
Output:
data/hkpl_faq_clean.csv
"""

import csv
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


FAQ_URL = "https://www.hkpl.gov.hk/en/ask-a-librarian/faq.html"
SOURCE_TITLE = "HKPL Ask a Librarian FAQ"
OUTPUT_FILE = Path("data/hkpl_faq_clean.csv")


def classify_domain(question: str) -> str:
    """Classify each FAQ question into a simple domain label."""
    q_lower = question.lower()

    if any(
        word in q_lower
        for word in [
            "e-resource",
            "e-resources",
            "e-book",
            "e-books",
            "e-magazine",
            "mobile app",
            "digital",
            "online",
            "adobe digital editions",
            "libby",
        ]
    ):
        return "e_resources"

    if any(
        word in q_lower
        for word in [
            "annual report",
            "law",
            "gazette",
            "standard",
            "collection",
        ]
    ):
        return "collections"

    return "reference_services"


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-breaking spaces."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_snippet(answer: str, max_len: int = 160) -> str:
    """Create a short snippet from the answer."""
    if "." in answer:
        first_sentence = answer.split(".")[0].strip() + "."
        return first_sentence[:max_len]

    return answer[:max_len] + "..." if len(answer) > max_len else answer


def main() -> None:
    """Fetch HKPL FAQ page, extract Q/A pairs, and save them to CSV."""
    print(f"Fetching {FAQ_URL} ...")

    response = requests.get(
        FAQ_URL,
        timeout=30,
        headers={
            "User-Agent": "HKPL-RAG-Data-Extractor/1.0"
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    faq_areas = soup.select("ul.faq_area")

    if not faq_areas:
        raise RuntimeError("Could not find any FAQ containers: ul.faq_area")

    items = []

    for faq_area in faq_areas:
        items.extend(faq_area.find_all("li"))
    rows = []
    skipped = 0

    for i, item in enumerate(items):
        question_elem = item.select_one("p.quest")
        answer_elem = item.select_one("div.answ")

        if not question_elem or not answer_elem:
            skipped += 1
            continue

        question = clean_text(question_elem.get_text(" ", strip=True))
        answer = clean_text(answer_elem.get_text(" ", strip=True))

        if not question or not answer:
            skipped += 1
            continue

        domain = classify_domain(question)
        snippet = make_snippet(answer)

        rows.append(
            {
                "domain": domain,
                "query": question,
                "expected_answer_text": answer,
                "expected_bib_ids": "",
                "expected_context_snippet": snippet,
                "source_title": SOURCE_TITLE,
                "source_url": FAQ_URL,
                "source_type": "official_website",
                "source_row_id": str(len(rows)),
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "domain",
        "query",
        "expected_answer_text",
        "expected_bib_ids",
        "expected_context_snippet",
        "source_title",
        "source_url",
        "source_type",
        "source_row_id",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} FAQ items.")
    print(f"Skipped {skipped} non-FAQ items.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()