import pymupdf
import re

def extract_text_from_pdf(uploaded_file):
    """Reads a PDF file buffer or path and returns raw text."""
    if hasattr(uploaded_file, "read"):
        doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
    else:
        doc = pymupdf.open(uploaded_file)
    text = " ".join([page.get_text() for page in doc])
    return text

def preprocess_for_bm25(text):
    """Cleans text for exact keyword matching."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.split()

def preprocess_for_semantics(text):
    """Cleans text for vector embedding."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text