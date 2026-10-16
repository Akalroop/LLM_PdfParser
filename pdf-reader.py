import os
import pymupdf
from openai import OpenAI
from pypdf import PdfReader

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("OLLAMA_MODEL", "medgemma1.5")

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
)

def fix_pua(text: str) -> str:
    """Decodes characters shifted into Unicode Private Use Area (0xF000 - 0xF0FF)."""
    return "".join(
        chr(ord(c) - 0xF000) if 0xF000 <= ord(c) <= 0xF0FF else c
        for c in text
    )

def extract_pdf_text(filename: str) -> str:
    """Extracts and normalizes text from the PDF file using pymupdf with pypdf fallback."""
    text_parts = []
    try:
        doc = pymupdf.open(filename)
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text") or ""
            cleaned_text = fix_pua(raw_text)
            text_parts.append(f"--- Page {page_num} ---\n{cleaned_text}")
    except Exception:
        reader = PdfReader(filename)
        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned_text = fix_pua(raw_text)
            text_parts.append(f"--- Page {page_num} ---\n{cleaned_text}")
    return "\n\n".join(text_parts)


def ask_file(filename: str, question: str) -> str:
    """Answer questions related to the PDF files content"""
    content = extract_pdf_text(filename)

    if not content.strip():
        return "No extractable text was found in this PDF, it may be image based"
    # with open(filename, "r", encoding="utf-8") as f:
    #     content = f.read()

    response = client.chat.completions.create(
        model= MODEL,
        messages=[
            {
                "role" : "system",
                "content":( 
                "You are a medical document parser.Extract only data explicitly stated in the PDF text — never infer,"
                " diagnose, guess, or fill in typical values; missing fields must be null."
                " Preserve numbers, units, dates, and medication names/dosages exactly as written,"
                " with no reformatting or normalization. If text is unclear or garbled (e.g. from OCR),"
                " mark it `'unclear': true` instead of guessing."
                " Ignore headers, footers, and boilerplate unless clinically relevant."
                ),
            },
            {
                "role": "user",
                "content": f"Document content:\n\n{content}\n\nQuestion: {question}",
            },
        ],
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pdf-reader.py <path_to_pdf> <question>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    user_question = " ".join(sys.argv[2:])
    
    answer = ask_file(pdf_path, user_question)
    print(answer)