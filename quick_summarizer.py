# quick_summarizer.py
from llmware.prompts import Prompt
import traceback

def quick_summarize_file(file_path: str, file_name: str, max_batch_cap: int = 15):
    """
    Calls llmware Prompt.summarize_document_fc with a file path + file name.
    Returns a list of summary points (strings).
    """
    try:
        # text_only=True to get list of points (strings)
        kp = Prompt().summarize_document_fc(
            file_path,
            file_name,
            topic=None,
            query=None,
            text_only=True,
            max_batch_cap=max_batch_cap,
        )
        # Ensure list of strings
        if isinstance(kp, list):
            return [str(x).strip() for x in kp if x and str(x).strip()]
        # if single string returned, split into lines
        if isinstance(kp, str):
            return [line.strip() for line in kp.splitlines() if line.strip()]
        # fallback: return empty
        return []
    except Exception as e:
        # We re-raise the exception after logging for app.py to catch and display
        raise ValueError(f"llmware summarizer failed: {traceback.format_exc()}")