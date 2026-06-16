"""
This script extracts FAQs from the Hong Kong Public Library's "Ask a Librarian" page and saves them to a CSV file.
docker run --rm -it \
  -v $(pwd):/data \
  -w /data \
  python:3.11-slim \
  bash -c "
    pip install --quiet requests beautifulsoup4 &&
    python /data/extract_faq.py
  "

Extract FAQs from the HKPL "Ask a Librarian" page and save as a clean CSV.
Usage: python extract_faq.py
Output: hkpl_faq_clean.csv
"""

import requests
from bs4 import BeautifulSoup
import csv
import re

# ----------------------------------------------------------------------
# Domain classifier based on keywords in the question
# ----------------------------------------------------------------------
def classify_domain(question: str) -> str:
    q_lower = question.lower()
    if any(word in q_lower for word in ['e-resource', 'e-book', 'e-magazine', 'mobile app', 'digital', 'online']):
        return 'e_resources'
    elif any(word in q_lower for word in ['annual report', 'law', 'gazette', 'standard', 'collection']):
        return 'collections'
    else:
        return 'reference_services'

# ----------------------------------------------------------------------
# Main extraction
# ----------------------------------------------------------------------
def main():
    url = "https://www.hkpl.gov.hk/en/ask-a-librarian/faq.html"
    print(f"Fetching {url} ...")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all <li> that contain both .quest and .answ
    items = soup.find_all('li')
    data = []
    skipped = 0

    for item in items:
        question_elem = item.select_one('p.quest')
        answer_elem = item.select_one('div.answ')
        if question_elem and answer_elem:
            question = question_elem.text.strip()
            answer = answer_elem.text.strip()

            # Clean answer: replace multiple spaces/newlines with a single space
            answer = re.sub(r'\s+', ' ', answer).strip()
            # Remove invisible non-breaking spaces (common in web pages)
            answer = answer.replace('\u00a0', ' ').strip()

            # Context snippet: first sentence (up to first period)
            if '.' in answer:
                snippet = answer.split('.')[0] + '.'
            else:
                snippet = answer[:100] + '...' if len(answer) > 100 else answer

            domain = classify_domain(question)
            data.append([domain, question, answer, '', snippet])
        else:
            skipped += 1

    print(f"Extracted {len(data)} FAQ items (skipped {skipped} items without Q/A).")

    # Write to CSV
    output_file = 'hkpl_faq_clean.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'query', 'expected_answer_text', 'expected_bib_ids', 'expected_context_snippet'])
        writer.writerows(data)

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()