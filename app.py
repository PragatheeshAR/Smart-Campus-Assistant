# app.py
import os
import tempfile
import traceback
import streamlit as st
import shutil
import re 
from typing import List, Dict, Any

# --- LOGIC IMPORTS ---

# 1. Core Utility: Text Extraction (from summarizer.py)
try:
    from summarizer import extract_text_from_pdf
    TEXT_UTILITY_AVAILABLE = True
except Exception:
    TEXT_UTILITY_AVAILABLE = False
    
# 2. Detailed Summarizer Model (from summarizer.py)
try:
    from summarizer import summarize_document
    HAS_DETAILED = TEXT_UTILITY_AVAILABLE and True
except Exception:
    HAS_DETAILED = False

# 3. Quick Summarizer (llmware logic)
try:
    from quick_summarizer import quick_summarize_file
    HAS_QUICK = True
except Exception:
    HAS_QUICK = False

# 4. Study Assistant (RAG logic)
try:
    from study_assistant import StudyAssistant
    HAS_ASSISTANT = TEXT_UTILITY_AVAILABLE and True 
except Exception as e:
    HAS_ASSISTANT = False
    print(f"Study Assistant loading error: {e}")


st.set_page_config(page_title="AI Study Assistant & Knowledge Builder", layout="centered")

# --- UI COMPONENTS & HELPERS ---

def save_uploaded_temp(uploaded) -> (str, str):
    """Saves uploaded_file to a temp directory."""
    fname = uploaded.name
    tmpdir = tempfile.mkdtemp(prefix="smartstudy_")
    out_path = os.path.join(tmpdir, fname)
    with open(out_path, "wb") as f:
        f.write(uploaded.getbuffer())
    return tmpdir, fname

def render_quick_points(points):
    """Render list of strings as a short revision summary."""
    if not points:
        st.warning("⚠️ No core revision points were generated.")
        return

    st.subheader("⚡ Core Revision Points")
    for i, p in enumerate(points, start=1):
        clean = p.replace("\n", " ").strip()
        clean = re.sub(r"<[^>]+>", "", clean)
        st.markdown(f"**{i}.** {clean}")

def render_detailed(doc_text):
    """Call and render detailed summary."""
    if not HAS_DETAILED:
        st.error("Detailed structured analysis module not found.")
        return
    with st.spinner("🧠 Generating detailed analysis..."):
        try:
            detailed = summarize_document(doc_text)
            st.subheader("📚 Comprehensive Document Analysis")
            st.text_area("Detailed analysis", value=detailed, height=400)
        except Exception as e:
            st.error(f"Failed to generate detailed analysis: {e}")
            st.exception(traceback.format_exc())


def render_assistant_ui(text_content):
    """UI for Dual-Mode Q&A and MCQ generation using StudyAssistant."""
    if not HAS_ASSISTANT:
        st.warning("Interactive Knowledge Assistant not available.")
        return

    st.markdown("---")
    st.header("Interactive Knowledge Assistant 🤖")
    
    @st.cache_resource
    def load_assistant(text):
        with st.spinner("Indexing document for knowledge retrieval..."):
            return StudyAssistant(text)

    assistant = load_assistant(text_content)
    
    # --- DUAL MODE Q&A SECTION ---
    st.subheader("❓ Ask Questions - Dual Mode")
    
    qa_mode = st.radio(
        "Select Answer Mode:",
        options=["📄 Grounded (Document Only)", "🌐 Web-Enhanced (Search Online)"],
        horizontal=True
    )
    
    q = st.text_input("Ask a question:", key="qa_query")
    
    if st.button("Get Answer", type="primary"):
        if q:
            if "Grounded" in qa_mode:
                with st.spinner("🔍 Searching document..."):
                    answer = assistant.answer_question_grounded(q)
                    st.markdown(answer)
            else:
                with st.spinner("🌐 Searching online..."):
                    answer = assistant.answer_question_web(q)
                    st.markdown(answer)
        else:
            st.warning("Please enter a question.")

    # --- MCQ GENERATION SECTION ---
    st.markdown("---")
    st.subheader("📝 Practice Quiz Generator")

    # 🔥 NEW MCQ MODE DROPDOWN ADDED
    mcq_mode = st.selectbox(
        "Select MCQ Generation Mode:",
        ["PDF Only", "Hybrid (PDF + Web)", "Web Only"]
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        mcq_topic = st.text_input("Optional Topic Filter:", key="mcq_topic")
    with col2:
        num_mcq = st.slider("Number of MCQs:", min_value=1, max_value=5, value=3)

    if st.button("Generate Quiz", type="secondary"):
        with st.spinner("🎯 Generating quiz questions..."):
            quizzes = assistant.generate_mcqs(
                topic=mcq_topic,
                num_questions=num_mcq,
                mode=mcq_mode   # <-- 🔥 NEW ARGUMENT
            )
            
            if isinstance(quizzes, str):
                st.error(quizzes)
                st.session_state.quiz_data = None
            elif quizzes:
                st.session_state.quiz_data = quizzes
                st.session_state.shown_answer = {}
                st.success(f"Generated {len(quizzes)} MCQs!")
            else:
                st.error("Quiz generation returned no questions.")

    # --- MCQ DISPLAY ---
    if "quiz_data" in st.session_state and st.session_state.quiz_data:
        st.markdown("---")
        st.markdown("### 🎯 Test Your Knowledge")

        if "shown_answer" not in st.session_state:
            st.session_state.shown_answer = {}

        for i, q_item in enumerate(st.session_state.quiz_data, 1):
            st.markdown(f"#### Question {i}")
            st.markdown(f"**{q_item['question']}**")
            st.markdown("\n".join(q_item["options"]))

            button_key = f"show_{i}"
            answer_key = f"ans_{i}"

            if st.button("Show Answer", key=button_key):
                st.session_state.shown_answer[answer_key] = not st.session_state.shown_answer.get(answer_key, False)
                st.rerun()

            if st.session_state.shown_answer.get(answer_key, False):
                st.success(f"Correct Answer: {q_item['correct_answer']}")

        if st.button("Clear Quiz"):
            del st.session_state.quiz_data
            st.session_state.shown_answer = {}
            st.rerun()


# --- MAIN EXECUTION ---

st.title("📘 AI Study Assistant & Knowledge Builder")
uploaded_file = st.file_uploader("📄 Upload Document", type=["pdf"])

if uploaded_file:
    
    summary_type = st.selectbox("Select Output Mode", [
        "Concise Revision Points", 
        "Detailed Structured Analysis", 
        "Both"
    ])
    
    max_batches = st.slider("Max Batches (Quick Summary)", 3, 40, 15)

    # FILE HANDLING
    if "tmpdir" not in st.session_state or st.session_state.fname != uploaded_file.name:
        try:
            if "tmpdir" in st.session_state and os.path.exists(st.session_state.tmpdir):
                shutil.rmtree(st.session_state.tmpdir)

            tmpdir, fname = save_uploaded_temp(uploaded_file)
            st.session_state.tmpdir = tmpdir
            st.session_state.fname = fname
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
            st.stop()
    else:
        tmpdir = st.session_state.tmpdir
        fname = st.session_state.fname
    
    full_path = os.path.join(tmpdir, fname)

    @st.cache_data(ttl=3600)
    def get_doc_text(path):
        return extract_text_from_pdf(path)

    doc_text = get_doc_text(full_path)

    # PROCESS DOCUMENT
    if st.button("✨ Process Document", type="primary"):
        if summary_type in ["Concise Revision Points", "Both"]:
            if HAS_QUICK:
                with st.spinner("Generating quick summary..."):
                    quick = quick_summarize_file(tmpdir, fname, max_batch_cap=max_batches)
                    render_quick_points(quick)

        if summary_type in ["Detailed Structured Analysis", "Both"]:
            render_detailed(doc_text)

    if doc_text and HAS_ASSISTANT:
        render_assistant_ui(doc_text)

else:
    st.info("Upload a PDF to begin.")
