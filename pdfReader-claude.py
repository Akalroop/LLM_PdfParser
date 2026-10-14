"""
Simple local PDF reader/Q&A tool using a local LLM served by Ollama.

Requirements:
    pip install openai pypdf

Make sure Ollama is running locally and you've pulled the model, e.g.:
    ollama pull medgemma1.5
    ollama serve   (usually already running as a background service)
"""

import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("OLLAMA_MODEL", "medgemma1.5")

client = OpenAI(
    base_url=OLLAMA_BASE_URL,  # Ollama's OpenAI-compatible endpoint
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),  # required by the SDK but unused by Ollama
)


def extract_pdf_text(filename: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(filename)
    text_parts = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        text_parts.append(f"--- Page {page_num} ---\n{page_text}")
    return "\n\n".join(text_parts)


def ask_file(filename: str, question: str) -> str:
    """Answer a question about the contents of a PDF file using a local LLM."""
    content = extract_pdf_text(filename)

    if not content.strip():
        return "No extractable text was found in this PDF (it may be scanned/image-based)."

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer questions only using "
                    "the information contained in the provided document. "
                    "If the answer cannot be found in the document, say so clearly."
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
