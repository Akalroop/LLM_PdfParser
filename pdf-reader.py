from openai import OpenAI
from pypdf import PdfReader
from pymupdf4llm import pymupdf

client = OpenAI(
    base_url=("http://localhost:11434/v1"),
    api_key="ollama",
)

MODEL = "medgemma"

def extract_pdf_text(filename: str) -> str:
    """Extracts all the text from the PDF file"""
    reader = PdfReader(filename)
    text_parts = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        text_parts.append(f"--- Page {page_num} ---\n{page_text}")
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