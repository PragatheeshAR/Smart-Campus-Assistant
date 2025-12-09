# app.py
import os
import tempfile
import traceback
import streamlit as st
import shutil

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
    HAS_ASSISTANT = TEXT_UTILITY_AVAILABLE and True # Assistant requires extracted text
except Exception:
    HAS_ASSISTANT = False


st.set_page_config(page_title="Smart Study Summarizer", layout="centered")

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
        st.warning("⚠️ No quick points were returned by the summarizer.")
        return

    st.subheader("⚡ Quick Revision Summary (llmware)")
    for i, p in enumerate(points, start=1):
        clean = p.replace("\n", " ").strip()
        import re
        clean = re.sub(r"<[^>]+>", "", clean)
        st.markdown(f"**{i}.** {clean}")

def render_detailed(doc_text):
    """Call and render detailed summary."""
    if not HAS_DETAILED:
        st.error("Detailed summarizer not found.")
        return
    with st.spinner("🧠 Generating detailed summary..."):
        try:
            # summarize_document takes the extracted text
            detailed = summarize_document(doc_text)
            st.subheader("📚 Detailed Study Summary (BART-CNN)")
            st.text_area("Detailed summary", value=detailed, height=400)
        except Exception as e:
            st.error(f"Failed to generate detailed summary: {e}")
            st.exception(traceback.format_exc())

def render_assistant_ui(text_content):
    """UI for Q&A and MCQ generation using the StudyAssistant class."""
    if not HAS_ASSISTANT:
        st.warning("Study Assistant not available.")
        return

    st.markdown("---")
    st.header("Ask Your Document (RAG Assistant) 🤖")
    
    # Cache the assistant object itself
    @st.cache_resource
    def load_assistant(text):
        with st.spinner("Indexing document for Q&A..."):
            return StudyAssistant(text)

    assistant = load_assistant(text_content)
    
    # Q&A Section
    st.subheader("❓ Question & Answer")
    q = st.text_input("Ask a specific question about the document:", key="qa_query")
    if st.button("Get Answer"):
        if q:
            with st.spinner("Searching and generating answer..."):
                answer = assistant.answer_question(q)
                st.info(f"**Answer:** {answer}")
        else:
            st.warning("Please enter a question.")

    # MCQ Generation Section
    st.subheader("📝 Generate Practice MCQs")
    num_mcq = st.slider("Number of MCQs to generate:", min_value=1, max_value=5, value=3)
    mcq_topic = st.text_input("Focus MCQs on a topic (optional):", key="mcq_topic")
    
    if st.button("Generate MCQs"):
        with st.spinner("Generating practice questions..."):
            mcqs = assistant.generate_mcqs(mcq_topic, num_mcq)
            st.text_area("Generated Multiple Choice Questions", value=mcqs, height=400)


# --- MAIN UI LOGIC ---

st.title("📘 Smart Study Summarizer (llmware + Hugging Face RAG)")
st.write("Upload a PDF to generate summaries and enable the Q&A assistant.")

uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:
    
    # --- Configuration ---
    summary_type = st.selectbox("📝 Choose summary type", ["Quick Summary (llmware)", "Detailed Summary (existing)", "Both"])
    max_batches = st.slider("Max batches (llmware)", min_value=3, max_value=40, value=15, help="Higher -> cover more of the document (slower).")

    # --- File Management (using session_state for persistence) ---
    if "tmpdir" not in st.session_state or st.session_state.fname != uploaded_file.name:
        try:
            tmpdir, fname = save_uploaded_temp(uploaded_file)
            st.session_state.tmpdir = tmpdir
            st.session_state.fname = fname
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
            st.stop()
    else:
        tmpdir = st.session_state.tmpdir
        fname = st.session_state.fname

    full_file_path = os.path.join(tmpdir, fname)
    
    # CHECK: Ensure text extraction utility is available before proceeding
    if not TEXT_UTILITY_AVAILABLE:
        st.error("Text extraction utility not found. Please ensure summarizer.py is present and correct.")
        st.stop()


    @st.cache_data
    def get_document_text(path):
        return extract_text_from_pdf(path)
        
    doc_text = get_document_text(full_file_path)

    # --- Action Button ---
    if st.button("✨ Summarize and Index Document"):
        
        # 1. Quick Summary (llmware)
        if summary_type in ["Quick Summary (llmware)", "Both"]:
            if HAS_QUICK:
                try:
                    with st.spinner("🔎 Running quick summarizer (llmware)..."):
                        quick_points = quick_summarize_file(tmpdir, fname, max_batch_cap=max_batches)
                        render_quick_points(quick_points)
                except Exception as e:
                    st.error("❌ Quick summarizer failed.")
                    st.exception(traceback.format_exc())
            else:
                st.warning("Quick summarizer utility not available.")

        # 2. Detailed Summary (BART-CNN)
        if summary_type in ["Detailed Summary (existing)", "Both"]:
            if HAS_DETAILED and doc_text:
                render_detailed(doc_text)
            elif HAS_DETAILED:
                 st.error("Could not extract text for Detailed Summarizer.")
            else:
                st.warning("Detailed summarizer utility not available.")

    # 3. Interactive Assistant UI (Always render if possible after file upload)
    if doc_text and HAS_ASSISTANT:
        render_assistant_ui(doc_text)
    elif uploaded_file and not HAS_ASSISTANT:
        st.warning("Study Assistant not available. Please ensure study_assistant.py is present and dependencies are installed.")
        

else:
    # Cleanup temp directory when file is deselected/app is reloaded
    if "tmpdir" in st.session_state:
        try:
            shutil.rmtree(st.session_state.tmpdir)
        except:
            pass
        for key in ["tmpdir", "fname", "assistant_text_content"]:
            if key in st.session_state:
                del st.session_state[key]
        
    st.info("📄 Upload a PDF to get started.")