import streamlit as st
from src.summarizer.inference import SummarizerPipeline

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="centered",
)


@st.cache_resource
def load_summarizer():
    return SummarizerPipeline()


st.title("AI Abstractive Text Summarizer")
st.markdown(
    "Fine-tuned **BART** sequence-to-sequence model (`facebook/bart-base`) for summarizing dialogues and long text."
)
st.divider()

# Load cached model
with st.spinner("Loading fine-tuned BART model..."):
    summarizer = load_summarizer()

# Optional sidebar controls
with st.sidebar:
    st.header("Generation Settings")
    max_len = st.slider("Max Summary Length (tokens)", min_value=32, max_value=200, value=128, step=8)
    st.markdown("---")
    st.caption("**Evaluation Benchmark (SAMSum):**\n- ROUGE-1: `41.76`\n- ROUGE-2: `19.57`\n- ROUGE-L: `35.64`")

sample_text = (
    "Sarah: Hey Alex, are we still on for the client demo tomorrow at 3 PM?\n"
    "Alex: Actually, the client asked to move it to 4:30 PM because their VP had a conflict.\n"
    "Sarah: Got it. Does that give us enough time to test the new API endpoint before the call?\n"
    "Alex: Yes, Priya and I are running final load tests at 2 PM, so everything will be ready by 3:30.\n"
    "Sarah: Perfect, I'll update the calendar invite and send over the revised slide deck."
)

if st.button("Load Sample Dialogue"):
    st.session_state["input_text"] = sample_text

user_text = st.text_area(
    "Enter text or conversation to summarize:",
    value=st.session_state.get("input_text", ""),
    height=220,
    placeholder="Paste a conversation, email thread, or article here...",
)

if st.button("Generate Summary", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text before generating a summary.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarizer.summarize(user_text, max_length=max_len)

        st.subheader("Generated Summary")
        st.success(summary)

        # Word count & compression stats
        orig_words = len(user_text.split())
        summ_words = len(summary.split())
        compression = round((1 - summ_words / max(orig_words, 1)) * 100, 1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Original Words", orig_words)
        col2.metric("Summary Words", summ_words)
        col3.metric("Compression", f"{compression}%")