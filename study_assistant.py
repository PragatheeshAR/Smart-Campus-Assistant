# study_assistant.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import spacy
import streamlit as st # Imported for cache_resource

# --- Model Loading (Cached) ---

# Load spaCy for robust sentence/paragraph splitting
try:
    # Attempt to load small model
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # Fallback if model not downloaded
    from spacy.lang.en import English
    nlp = English()
    # It's recommended to run 'python -m spacy download en_core_web_sm' first

# 1. Retrieval Model (Embedding)
# Used for converting text chunks and questions into vectors
@st.cache_resource(show_spinner=False)
def load_embedder_model():
    # Model specified in requirements.txt (sentence-transformers)
    return SentenceTransformer('all-MiniLM-L6-v2') 

# 2. Generation Model (Q&A and MCQ generation)
# Flan-T5 is excellent for Instruction Tuning tasks
QA_MODEL = "google/flan-t5-base" 

@st.cache_resource(show_spinner=False)
def load_qa_model():
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL)
    return tokenizer, model

# Load models outside the class (using Streamlit cache)
embedder = load_embedder_model()
qa_tokenizer, qa_model = load_qa_model()

# --- Helper Functions (Chunking) ---

def _chunk_text_spacy(text: str):
    """Splits text into chunks, prioritizing paragraph boundaries for RAG."""
    doc = nlp(text)
    
    # Try to split by meaningful paragraphs first
    # FIX: Corrected from p.text.strip() to p.strip() as p is already a string
    paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""
    max_chunk_len = 1500 # Max characters for a RAG chunk

    for p in paragraphs:
        if len(p) > max_chunk_len:
            # If paragraph is too long, split into sentences
            sents = [s.text.strip() for s in nlp(p).sents if s.text.strip()]
            for s in sents:
                 if len(current_chunk) + len(s) + 10 < max_chunk_len:
                    current_chunk += " " + s
                 else:
                    if current_chunk: chunks.append(current_chunk.strip())
                    current_chunk = s
        else:
            # Add paragraph if it fits or start a new chunk with it
            if len(current_chunk) + len(p) + 10 < max_chunk_len:
                current_chunk += "\n\n" + p
            else:
                if current_chunk: chunks.append(current_chunk.strip())
                current_chunk = p
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short or empty chunks
    return [c for c in chunks if len(c) > 50] 


# --- Main Class (StudyAssistant) ---
class StudyAssistant:
    def __init__(self, full_text):
        self.chunks = _chunk_text_spacy(full_text)
        self.index = self._build_faiss_index()

    def _build_faiss_index(self):
        """Creates the FAISS vector index."""
        if not self.chunks:
            return None
            
        embeddings = embedder.encode(self.chunks)
        d = embeddings.shape[1] 
        # IndexFlatL2 for fast Euclidean distance similarity search
        index = faiss.IndexFlatL2(d) 
        # FAISS requires float32 numpy array
        index.add(np.array(embeddings).astype('float32'))
        return index

    def _retrieve_context(self, question, k=4):
        """Retrieves the top k chunks relevant to the question."""
        if not self.index:
            return "No document text indexed."
            
        # 1. Embed the query
        query_embedding = embedder.encode([question])
        # 2. Search the index
        D, I = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        # 3. Get the chunks based on the indices I
        context = [self.chunks[i] for i in I[0] if i < len(self.chunks)] 
        return "\n\n---\n\n".join(context)

    def _generate(self, prompt, max_new_tokens=150):
        """Handles the generation call to Flan-T5."""
        inputs = qa_tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, # Max context length for Flan-T5-base
            truncation=True
        )
        
        # Determine device for generation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        output = qa_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False
        )
        return qa_tokenizer.decode(output[0], skip_special_tokens=True)


    def answer_question(self, question):
        """RAG Q&A implementation."""
        context = self._retrieve_context(question, k=4)
        
        if len(context) < 100:
             return "I could not find enough relevant information in the document to answer that question."

        # Prompt forces the model to use ONLY the context
        prompt = f"""
        Context: {context}
        
        Using ONLY the context provided, answer the following question concisely and accurately. If the answer is not in the context, state that you cannot answer it based on the provided text.
        
        Question: {question}
        
        Answer:"""
        
        return self._generate(prompt, max_new_tokens=100)

    def generate_mcqs(self, topic: str = None, num_questions: int = 3):
        """Generates multiple choice questions."""
        
        # Retrieval query targets concepts relevant to testing
        retrieval_query = f"Key concepts and facts for a test on {topic}" if topic else "Key concepts and facts of the document for a practice test"
        context = self._retrieve_context(retrieval_query, k=5)
        
        if len(context) < 200:
            return "Could not retrieve enough document content to reliably generate MCQs."

        # Structured prompt for reliable MCQ output
        mcq_prompt = f"""
        Context: {context}
        
        Generate exactly {num_questions} multiple choice questions (MCQs) with 4 options (A, B, C, D) and clearly state the correct answer at the end of each question. Use ONLY information from the context.
        
        Example Format:
        1. What is the main characteristic of a T-cell?
        A) Produces antibodies
        B) Directly attacks infected cells (Correct Answer)
        C) Phagocytizes bacteria
        D) Releases histamine

        Begin the MCQs:"""
        
        # Allow more tokens for structured, longer MCQ response
        return self._generate(mcq_prompt, max_new_tokens=70 * num_questions)