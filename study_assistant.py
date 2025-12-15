# study_assistant.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import spacy
import streamlit as st
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import random

# =========================
# LOAD MODELS
# =========================

try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.lang.en import English
    nlp = English()

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

QA_MODEL = "google/flan-t5-base"

@st.cache_resource(show_spinner=False)
def load_qa():
    tok = AutoTokenizer.from_pretrained(QA_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL).to("cpu")
    return tok, model

QG_MODEL = "mrm8488/t5-base-finetuned-question-generation-ap"

@st.cache_resource(show_spinner=False)
def load_qg():
    tok = AutoTokenizer.from_pretrained(QG_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL).to("cpu")
    return tok, model

embedder = load_embedder()
qa_tok, qa_model = load_qa()
qg_tok, qg_model = load_qg()

# =========================
# HELPERS
# =========================

def chunk_text(text, size=600):
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 30]
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) < size:
            cur += " " + s
        else:
            chunks.append(cur.strip())
            cur = s
    if cur:
        chunks.append(cur.strip())
    return chunks

def web_search(query):
    try:
        url = f"https://html.duckduckgo.com/html/?q={query.replace(' ','+')}"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        results = [d.get_text(" ", strip=True) for d in soup.select(".result__body")[:5]]
        return "\n".join(results)
    except:
        return ""

def generate_question(context):
    prompt = f"generate question: {context}"
    inp = qg_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = qg_model.generate(**inp, max_new_tokens=64)
    return qg_tok.decode(out[0], skip_special_tokens=True)

def extract_answer(context, question):
    prompt = f"""
Context:
{context}

Question:
{question}

Answer using only the context.
"""
    inp = qa_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = qa_model.generate(**inp, max_new_tokens=64)
    return qa_tok.decode(out[0], skip_special_tokens=True)

# =========================
# STUDY ASSISTANT
# =========================

class StudyAssistant:
    def __init__(self, full_text):
        self.text = full_text
        self.chunks = chunk_text(full_text)
        self.index = self.build_index()

    def build_index(self):
        if not self.chunks:
            return None
        emb = embedder.encode(self.chunks)
        idx = faiss.IndexFlatL2(emb.shape[1])
        idx.add(np.array(emb).astype("float32"))
        return idx

    def retrieve(self, query, k=4):
        if not self.index:
            return self.text
        q_emb = embedder.encode([query])
        _, I = self.index.search(np.array(q_emb).astype("float32"), k)
        return "\n".join([self.chunks[i] for i in I[0] if i < len(self.chunks)])

    # -------------------------
    # Q&A
    # -------------------------

    def answer_question_grounded(self, q):
        ctx = self.retrieve(q)
        return extract_answer(ctx, q)

    def answer_question_web(self, q):
        ctx = web_search(q)
        return extract_answer(ctx, q)

    # -------------------------
    # MCQ (QUESTION + ANSWER ONLY)
    # -------------------------

    def generate_mcqs(self, topic, num_questions, mode):
        topic = topic or "key concepts"
        context = self.retrieve(topic)

        if mode == "Web Only":
            context = web_search(topic)

        questions = []
        used = set()

        for chunk in self.chunks:
            if len(questions) >= num_questions:
                break

            q = generate_question(chunk)
            if not q or q in used:
                continue
            used.add(q)

            ans = extract_answer(chunk, q)
            if not ans or len(ans) < 3:
                continue

            # 🔥 OPTIONS REMOVED COMPLETELY
            questions.append({
                "question": q,
                "options": [],            # <-- NOTHING SHOWN
                "correct_answer": ans     # <-- SHOWN ONLY ON CLICK
            })

        return questions if questions else "MCQ generation failed."
