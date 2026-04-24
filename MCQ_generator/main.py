from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import textwrap

import ollama
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Folders used by the script.
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "document"
OUTPUT_DIR = BASE_DIR / "output"


def ensure_folders():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def find_source_pdf():
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdf_files:
        return None
    return pdf_files[0]


def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(str(file_path))
        parts = []
        for page in reader.pages[:10]: 
            extracted = page.extract_text()
            if extracted:
                parts.append(extracted)
        return "\n".join(parts)
    except Exception as e:
        raise RuntimeError(f"Error reading the PDF: {e}") from e


def generate_quiz_text(document_text, num_mcqs, num_short_answers):
    num_mcqs = int(num_mcqs)
    num_short_answers = int(num_short_answers)

    print(f"Generating {num_mcqs} MCQs and {num_short_answers} short-answer questions with AI...")

    prompt = f"""
    You are an expert professor. I am going to give you a document. 
    Based ONLY on the information in this document, generate:
    1) Exactly {num_mcqs} multiple-choice questions.
    2) Exactly {num_short_answers} short-answer questions.

    Return your result in two sections with these exact headers:
    SECTION 1: MULTIPLE-CHOICE QUESTIONS
    SECTION 2: SHORT-ANSWER QUESTIONS
    
    In SECTION 1, for every question, you MUST follow this exact format:
    Question [Number]: [The Question]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [A, B, C, or D]
    Explanation: [One short sentence explaining why it is correct based on the text]

    In SECTION 2, for every question, you MUST follow this exact format:
    Question [Number]: [The Question]
    Answer: [A descriptive short answer in 2-4 sentences based on the document]

    DOCUMENT TEXT:
    {document_text}
    """

    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    except Exception as e:
        raise RuntimeError(f"AI Error: {e}") from e


def save_text_as_pdf(title, source_name, quiz_text, output_path):
    page_width, page_height = A4
    margin = 50
    text_width = page_width - (2 * margin)

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    pdf.setTitle(title)
    y = page_height - margin

    def draw_wrapped(text, font_name="Helvetica", font_size=11, indent=0, leading=15):
        nonlocal y
        max_chars = max(30, int(text_width / (font_size * 0.55)))
        paragraphs = text.splitlines() if text else [""]

        for paragraph in paragraphs:
            lines = [""] if not paragraph.strip() else textwrap.wrap(paragraph, width=max_chars)
            for line in lines:
                if y < margin + leading:
                    pdf.showPage()
                    y = page_height - margin
                pdf.setFont(font_name, font_size)
                pdf.drawString(margin + indent, y, line)
                y -= leading

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(margin, y, title)
    y -= 28
    pdf.setFont("Helvetica", 10)
    pdf.drawString(margin, y, f"Source: {source_name}")
    y -= 16
    pdf.drawString(margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 24

    draw_wrapped(quiz_text, font_name="Helvetica", font_size=11, leading=15)
    pdf.save()


def main():
    parser = ArgumentParser(description="Generate MCQs and short-answer questions from a PDF dropped into the document folder.")
    parser.add_argument("-n", "--num-questions", type=int, default=5, help="Number of MCQs to generate")
    parser.add_argument("-s", "--num-short-answers", type=int, default=5, help="Number of short-answer questions to generate")
    args = parser.parse_args()

    ensure_folders()
    source_pdf = find_source_pdf()
    if source_pdf is None:
        print(f"No PDF found in {INPUT_DIR}. Put your document there and run this script again.")
        return

    print(f"Reading PDF from: {source_pdf.name}")
    document_text = extract_text_from_pdf(source_pdf)
    if not document_text.strip():
        print("Could not find any readable text in the PDF. It might be an image-only document.")
        return

    quiz_text = generate_quiz_text(document_text, args.num_questions, args.num_short_answers)

    output_path = OUTPUT_DIR / f"quiz_{source_pdf.stem}.pdf"
    save_text_as_pdf(
        title="AI Generated MCQ + Short-Answer Quiz",
        source_name=source_pdf.name,
        quiz_text=quiz_text,
        output_path=output_path,
    )

    print(f"Quiz saved to: {output_path}")


if __name__ == "__main__":
    main()