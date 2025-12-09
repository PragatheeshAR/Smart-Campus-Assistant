# summarizer.py
"""
Pipeline:
1) Extract text from PDF
2) Detect headings/topics (rule-based + spaCy helpers)
3) Adaptive semantic chunking (small / medium / large PDF)
4) Summarize each chunk safely (token truncation)
5) Merge into structured headings + bullet points
"""

import re
import io
from typing import List, Tuple, Optional
import PyPDF2
import spacy
import torch
from transformers import pipeline

# -------------------------
# Load spaCy
# -------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    from spacy.lang.en import English
    nlp = English()

# -------------------------
# Model selection
# -------------------------
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
_SUM_PIPE = None


def load_summarizer():
    """Load only once."""
    global _SUM_PIPE
    if _SUM_PIPE is None:
        device = 0 if torch.cuda.is_available() else -1
        _SUM_PIPE = pipeline(
            "summarization",
            model=SUMMARIZER_MODEL,
            tokenizer=SUMMARIZER_MODEL,
            device=device,
        )
    return _SUM_PIPE


# -------------------------
# PDF text extraction
# -------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    if hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(data))
    else:
        reader = PyPDF2.PdfReader(uploaded_file)

    pages = []
    for p in reader.pages:
        try:
            text = p.extract_text()
        except:
            text = ""
        if text:
            pages.append(text)

    return "\n\n".join(pages).strip()


# -------------------------
# Document size classification
# -------------------------
def classify_document_size(text: str):
    length = len(text)

    if length < 5000:
        return "small"
    elif length < 40000:
        return "medium"
    else:
        return "large"


# -------------------------
# Heading detection
# -------------------------
HEADING_PATTERNS = [
    r"^[A-Z][A-Z\s\-]{3,}$",
    r"^\d+[\.\)]\s+.+",
    r"^[A-Z][a-z]+(:)$",
    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}$",
]


def likely_heading(line: str) -> bool:
    if not line or len(line.strip()) < 3:
        return False

    s = line.strip()

    if s.endswith("."):
        return False

    if s.startswith(("-", "•", "*")):
        return False

    # Short, title-like lines
    if len(s.split()) <= 8:
        for pat in HEADING_PATTERNS:
            if re.match(pat, s):
                return True

    # Ratio of uppercase starting words
    words = s.split()
    cap = sum(1 for w in words if w[:1].isupper())
    if cap >= max(1, len(words) // 2) and len(words) <= 8:
        return True

    return False


def detect_headings_and_sections(text: str) -> List[Tuple[str, str]]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned = []

    # Remove double blank lines
    for ln in lines:
        if ln.strip() == "" and (len(cleaned) == 0 or cleaned[-1] == ""):
            continue
        cleaned.append(ln)

    sections = []
    current_heading = "Document"
    current_lines = []

    for ln in cleaned:
        if likely_heading(ln):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = ln.strip()
            current_lines = []
        else:
            current_lines.append(ln)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections if sections else [("Document", text)]


# -------------------------
# Adaptive chunking
# -------------------------
def semantic_chunk_section(text: str, doc_size="medium"):
    text = text.strip()
    if not text:
        return []

    if doc_size == "small":
        max_chars = 1800
        overlap = 80
    elif doc_size == "medium":
        max_chars = 2500
        overlap = 150
    else:
        max_chars = 3500
        overlap = 200

    # Sentence segmentation
    if nlp:
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    else:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current += (" " if current else "") + sent
        else:
            chunks.append(current.strip())
            seed = current[-overlap:] if len(current) > overlap else current
            current = seed + " " + sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


# -------------------------
# Remove instruction leaks
# -------------------------
def clean_instruction_leaks(txt: str) -> str:
    patterns = [
        r"You are an experienced teacher.*",
        r"Produce STUDENT-FRIENDLY.*",
        r"Explain key concepts.*",
        r"bullet points.*",
        r"Be exhaustive.*",
    ]
    for p in patterns:
        txt = re.sub(p, "", txt, flags=re.IGNORECASE)
    return txt.strip()


# -------------------------
# Safe summarization (prevents IndexError)
# -------------------------
def summarize_chunk_with_model(text: str, min_length=80, max_length=250):
    pipe = load_summarizer()
    text = clean_instruction_leaks(text)

    tokenizer = pipe.tokenizer

    # HARD TRUNCATION to stay within BART's limit
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=900,
        return_tensors="pt"
    )
    safe_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)

    try:
        out = pipe(
            safe_text,
            min_length=min_length,
            max_length=max_length,
            no_repeat_ngram_size=3,
            num_beams=4,
            early_stopping=True,
        )
        summary = out[0]["summary_text"]

    except Exception:
        # Last-resort fallback
        encoded_small = tokenizer(
            safe_text,
            truncation=True,
            max_length=600,
            return_tensors="pt"
        )
        safe_text2 = tokenizer.decode(encoded_small["input_ids"][0], skip_special_tokens=True)
        out = pipe(safe_text2, min_length=40, max_length=120)
        summary = out[0]["summary_text"]

    return clean_instruction_leaks(summary)


# -------------------------
# Summarize a section
# -------------------------
def summarize_section(title: str, content: str, doc_size="medium", progress=None):
    if not content or len(content) < 50:
        return f"## {title}\n\n- {content.strip()}\n"

    chunks = semantic_chunk_section(content, doc_size=doc_size)

    section_parts = []

    for idx, ch in enumerate(chunks, 1):
        if progress:
            progress(idx, len(chunks), f"Summarizing chunk {idx}/{len(chunks)} for {title}")

        summary = summarize_chunk_with_model(ch)

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
        bullets = "\n".join([f"- {s}" for s in sentences])

        section_parts.append(bullets)

    combined = "\n\n".join(section_parts)
    return f"## {title}\n\n{combined}\n"


# -------------------------
# Full document summarizer
# -------------------------
def summarize_document(text: str, progress_callback=None) -> str:
    doc_size = classify_document_size(text)

    sections = detect_headings_and_sections(text)
    final_output = []

    total = len(sections)

    for i, (title, content) in enumerate(sections, 1):

        if progress_callback:
            progress_callback(i, total, f"Processing section: {title}")

        def reporter(ch_idx, ch_total, msg):
            if progress_callback:
                progress_callback(
                    i - 1 + (ch_idx / max(1, ch_total)),
                    total,
                    msg
                )

        sec = summarize_section(title, content, doc_size, progress=reporter)
        final_output.append(sec)

    summary = "\n\n".join(final_output)
    summary = re.sub(r"\n{3,}", "\n\n", summary)
    return summary.strip()
