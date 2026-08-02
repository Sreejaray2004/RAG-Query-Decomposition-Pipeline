from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF.
    """
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    return text.strip()