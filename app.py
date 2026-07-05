import os
import re
import traceback
import streamlit as st
import streamlit.components.v1 as components

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.txt_splitter import split_documents
from src.ingestion.embeddings import get_local_embeddings
from src.ingestion.retriever import retrieve
from src.vectorstore.ingestion.vector_db import create_vector_db, load_vector_db, vector_db_exists

from src.study_tools.summary_generator import generate_summary
from src.study_tools.flashcard_generator import generate_flashcards
from src.study_tools.notes_generator import generate_notes
from src.study_tools.chatbot import generate_chatbot_answer

APP_NAME = "Exam Prep AI Chatbot"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SESSION STATE ──
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Summary"
if "processed" not in st.session_state:
    st.session_state.processed = False
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []


# ── HELPER: parse raw "Front:/Back:" text into structured cards ──
def parse_flashcards(raw_text: str):
    cards = []
    blocks = re.split(r'\n?\d+\.\s*', raw_text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        front_match = re.search(r'Front:\s*(.+?)(?=Back:|$)', block, re.DOTALL | re.IGNORECASE)
        back_match = re.search(r'Back:\s*(.+?)$', block, re.DOTALL | re.IGNORECASE)
        if front_match and back_match:
            cards.append({
                "front": front_match.group(1).strip(),
                "back": back_match.group(1).strip(),
            })
    return cards


# ── HELPER: render flippable flashcards (beige/Playfair themed) ──
def render_flip_cards(cards):
    html = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Nunito:wght@400;600;700&display=swap');
    body { margin:0; background:transparent; font-family:'Nunito', sans-serif; }
    .fc-grid {
        display:grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 22px;
        padding: 8px 2px;
    }
    .fc-card { background:transparent; width:100%; height:220px; perspective:1200px; }
    .fc-inner {
        position:relative; width:100%; height:100%;
        transition: transform 0.7s; transform-style: preserve-3d;
        cursor:pointer;
    }
    .fc-card:hover .fc-inner { transform: rotateY(180deg); }
    .fc-face {
        position:absolute; width:100%; height:100%;
        backface-visibility:hidden; border-radius:16px;
        display:flex; align-items:center; justify-content:center;
        text-align:center; padding:24px; box-sizing:border-box;
        border: 1px solid #C8AD7F;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    }
    .fc-front {
        background: #D4B896;
        color:#3B2F1E;
        font-family:'Playfair Display', serif;
        font-weight:600;
        font-size:1.05rem;
    }
    .fc-back {
        background: #EDE0C4;
        color:#5C4827;
        transform: rotateY(180deg);
        font-size:0.95rem;
        overflow:auto;
    }
    .fc-tag {
        position:absolute; top:10px; left:14px;
        font-size:0.7rem; font-weight:700; letter-spacing:0.08em;
        color:#A0845C; text-transform:uppercase;
    }
    .fc-hint { text-align:center; color:#7A6040; font-family:'Nunito', sans-serif;
               font-size:0.85rem; margin-bottom:14px; }
    </style>
    <div class="fc-hint">🖱️ Hover a card to flip it and reveal the answer</div>
    <div class="fc-grid">
    """
    for i, card in enumerate(cards):
        front = card.get("front", "").replace("<", "&lt;").replace(">", "&gt;")
        back = card.get("back", "").replace("<", "&lt;").replace(">", "&gt;")
        html += f"""
        <div class="fc-card">
            <div class="fc-inner">
                <div class="fc-face fc-front">
                    <div class="fc-tag">Card {i+1}</div>
                    {front}
                </div>
                <div class="fc-face fc-back">{back}</div>
            </div>
        </div>
        """
    html += "</div>"

    rows = (len(cards) + 2) // 3  # ~3 cards per row
    height = 60 + rows * 250
    components.html(html, height=height, scrolling=True)

# ── GLOBAL BEIGE BACKGROUND + FONTS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Nunito:wght@400;600;700&display=swap');

    [data-testid="stAppViewContainer"] {
        background-color: #F2E8D9;
        font-family: 'Nunito', sans-serif;
    }
    [data-testid="stHeader"] {
        background-color: #F2E8D9;
    }
    [data-testid="stSidebar"] {
        background-color: #E8D5B7;
        font-family: 'Playfair Display', serif !important;
    }

    /* Apply Playfair only to text elements, NOT icons */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Playfair Display', serif !important;
        color: #3B2F1E;
    }

    /* Keep icon fonts untouched so they render as icons, not text */
    [data-testid="stSidebar"] [data-testid="stIconMaterial"],
    [data-testid="stSidebar"] .material-icons,
    [data-testid="stSidebar"] [class*="icon"] {
        font-family: 'Material Symbols Rounded', sans-serif !important;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #3B2F1E !important;
    }
    p, div, label {
        font-family: 'Nunito', sans-serif !important;
    }
    .stButton > button {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.1rem !important;
        background-color: #D4B896 !important;
        color: #3B2F1E !important;
        border: 1px solid #C8AD7F !important;
        border-radius: 10px !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #C8AD7F !important;
        color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

# ── HERO SECTION ──
st.markdown(f"""
    <div style="
        background-color: #D4B896;
        padding: 50px 40px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 30px;
        border: 2px solid #C8AD7F;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    ">
        <h1 style="
            color: #3B2F1E;
            font-size: 2.8rem;
            font-family: 'Playfair Display', serif;
            margin-bottom: 10px;
        ">
            📚 {APP_NAME}
        </h1>
        <p style="
            color: #3B2F1E;
            font-size: 1.1rem;
            font-family: 'Nunito', sans-serif;
            margin-bottom: 6px;
        ">
            Transform your PDFs into:
        </p>
        <p style="
            color: #5C4827;
            font-size: 1rem;
            font-family: 'Nunito', sans-serif;
        ">
            ✅ Smart Summaries &nbsp;|&nbsp; 🃏 Flashcards &nbsp;|&nbsp; 📝 Study Notes &nbsp;|&nbsp; 🤖 AI Q&A
        </p>
    </div>
""", unsafe_allow_html=True)


# ── PIPELINE HELPER ──
def process_document(uploaded_file):
    """Save the uploaded PDF, chunk it, embed it, and build the FAISS store."""

    save_dir = "saved_files"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    documents = load_pdf(file_path)
    if not documents:
        raise ValueError("No content could be extracted from this PDF.")

    chunks = split_documents(documents)
    embeddings = get_local_embeddings()
    vector_db = create_vector_db(chunks, embeddings)

    return vector_db, len(documents), len(chunks)


def get_context(topic, top_k=5, fallback_chars=8000):
    """Retrieve relevant chunks for a topic, or fall back to the start of the doc."""

    vector_db = st.session_state.vector_db

    if topic and topic.strip():
        docs = retrieve(topic, vector_db, top_k=top_k)
    else:
        docs = retrieve("overview summary main topics", vector_db, top_k=top_k)

    context = "\n\n".join(doc.page_content for doc in docs)
    return context[:fallback_chars] if not topic else context


# ── SIDEBAR ──
with st.sidebar:
    st.markdown(f"""
        <div style="
            text-align: center;
            padding: 24px 0 20px 0;
            border-bottom: 2px solid #C8AD7F;
            margin-bottom: 20px;
        ">
            <div style="
                font-family: 'Playfair Display', serif;
                font-size: 2.4rem;
                font-weight: 700;
                color: #3B2F1E;
                line-height: 1.1;
            ">
                📚 {APP_NAME}
            </div>
            <div style="
                font-family: 'Playfair Display', serif;
                font-size: 1rem;
                color: #7A6040;
                margin-top: 4px;
            ">
                ~ AI Exam Assistant ~
            </div>
            <div style="
                margin-top: 10px;
                background: linear-gradient(90deg, #C8AD7F, #D4B896, #C8AD7F);
                height: 3px;
                border-radius: 10px;
            "></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Upload your PDF")
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        is_new_file = st.session_state.file_name != uploaded_file.name

        if is_new_file:
            st.session_state.processed = False

        if st.session_state.processed:
            st.success(f"✅ {uploaded_file.name} processed and ready!")
        else:
            st.info(f"👆 {uploaded_file.name} uploaded — click below to process it")
            if st.button("⚙️ Process Document", use_container_width=True):
                with st.spinner(
                    "Reading PDF, chunking and embedding… this can take a minute or two "
                    "on Voyage's free tier (rate-limited to 3 requests/min if you haven't "
                    "added a payment method) — please don't refresh."
                ):
                    try:
                        vector_db, doc_count, chunk_count = process_document(uploaded_file)
                        st.session_state.vector_db = vector_db
                        st.session_state.processed = True
                        st.session_state.file_name = uploaded_file.name
                        st.session_state.doc_count = doc_count
                        st.session_state.chunk_count = chunk_count
                        st.session_state.chat_history = []
                        st.success("Document processed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
                        st.code(traceback.format_exc())
    else:
        st.info("👆 Upload a PDF to get started")

    if st.session_state.processed:
        st.markdown("---")
        st.caption(
            f"📄 {st.session_state.get('doc_count', 0)} pages · "
            f"🧩 {st.session_state.get('chunk_count', 0)} chunks"
        )

    st.markdown("---")
    st.markdown("### 🗂️ Navigate")

    if st.button("📄 Summary"):
        st.session_state.active_tab = "Summary"

    if st.button("🃏 Flashcards"):
        st.session_state.active_tab = "Flashcards"

    if st.button("📝 Notes"):
        st.session_state.active_tab = "Notes"

    if st.button("🤖 Ask Questions"):
        st.session_state.active_tab = "QA"


# ── CONTENT AREA ──
ready = st.session_state.processed

if st.session_state.active_tab == "Summary":
    st.markdown("### 📄 Summary")
    if not ready:
        st.info("Upload and process a PDF in the sidebar, then click Generate Summary to see results here.")
    topic = st.text_input("Focus topic (optional)", key="summary_topic", placeholder="Leave blank to summarize the whole document")
    if st.button("✨ Generate Summary", disabled=not ready):
        with st.spinner("Generating summary…"):
            context = get_context(topic)
            result = generate_summary(context)
        st.markdown(result)

elif st.session_state.active_tab == "Flashcards":
    st.markdown("### 🃏 Flashcards")
    if not ready:
        st.info("Upload and process a PDF in the sidebar, then click Generate Flashcards to see results here.")
    topic = st.text_input("Topic (optional)", key="flash_topic", placeholder="Leave blank to cover the whole document")
    num_cards = st.number_input("Number of flashcards", min_value=1, max_value=30, value=10)
    if st.button("✨ Generate Flashcards", disabled=not ready):
        with st.spinner("Generating flashcards…"):
            context = get_context(topic)
            result = generate_flashcards(context, count=num_cards)
        parsed = parse_flashcards(result)
        if parsed:
            st.session_state.flashcards = parsed
        else:
            st.session_state.flashcards = []
            st.warning("Couldn't parse the flashcards into front/back pairs — showing raw output instead.")
            st.markdown(result)

    if st.session_state.flashcards:
        st.markdown(f"**{len(st.session_state.flashcards)} flashcards generated**")
        render_flip_cards(st.session_state.flashcards)

elif st.session_state.active_tab == "Notes":
    st.markdown("### 📝 Notes")
    if not ready:
        st.info("Upload and process a PDF in the sidebar, then click Generate Notes to see results here.")
    topic = st.text_input("Topic", key="notes_topic", placeholder="e.g. Thermodynamics, World War II…")
    if st.button("✨ Generate Notes", disabled=not ready):
        with st.spinner("Generating notes…"):
            context = get_context(topic)
            result = generate_notes(context, topic=topic or "the given content")
        st.markdown(result)

elif st.session_state.active_tab == "QA":
    st.markdown("### 🤖 Ask Questions")
    if not ready:
        st.info("Upload and process a PDF in the sidebar, then type your question below.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Type your question here...", disabled=not ready)

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-10:]
        )

        with st.spinner("Thinking…"):
            docs = retrieve(user_question, st.session_state.vector_db, top_k=5)
            context = "\n\n".join(d.page_content for d in docs)
            answer = generate_chatbot_answer(user_question, context, history_text)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)