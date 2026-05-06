from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.embeddings import DEFAULT_EMBEDDING_MODEL
from rag.generator import DEFAULT_GENERATION_MODEL
from rag.pipeline import RagChatbot


TOP_K = 4
MIN_SCORE = 0.25
MAX_NEW_TOKENS = 220
TEMPERATURE = 0.1
TOP_P = 0.9


# Streamlit owns the visual layer; all RAG logic lives in src/rag.
st.set_page_config(
    page_title="Colombia Comparte",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    .main .block-container {
        max-width: 1180px;
        padding: 1.4rem 2rem 2.2rem;
    }
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.9rem;
        margin-bottom: 2.2rem;
    }
    .brand {
        font-size: 1.05rem;
        font-weight: 800;
        color: #111827;
    }
    .nav-note {
        color: #6b7280;
        font-size: 0.92rem;
    }
    .hero {
        min-height: 58vh;
        display: grid;
        align-items: center;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 2.6rem;
    }
    .eyebrow {
        color: #0f766e;
        font-weight: 750;
        margin-bottom: 0.7rem;
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 0;
    }
    .hero h1 {
        font-size: clamp(2.1rem, 5vw, 4.8rem);
        line-height: 1.02;
        margin: 0 0 1rem;
        color: #111827;
        letter-spacing: 0;
        max-width: 860px;
    }
    .hero p {
        font-size: 1.12rem;
        color: #4b5563;
        max-width: 680px;
        line-height: 1.65;
        margin-bottom: 1.4rem;
    }
    .trust-row {
        display: flex;
        gap: 1.4rem;
        flex-wrap: wrap;
        margin-top: 1.6rem;
        color: #374151;
    }
    .trust-item strong {
        display: block;
        font-size: 1.35rem;
        color: #111827;
    }
    .chat-shell {
        border: 1px solid #d1d5db;
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 18px 45px rgba(17, 24, 39, 0.12);
        margin-top: 1rem;
    }
    .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.8rem;
        margin-bottom: 0.9rem;
    }
    .chat-title {
        font-weight: 800;
        color: #111827;
    }
    .chat-status {
        color: #0f766e;
        font-size: 0.88rem;
        font-weight: 650;
    }
    .hint {
        color: #6b7280;
        font-size: 0.92rem;
        margin-top: 0.6rem;
    }
    .stButton > button {
        border-radius: 999px;
        border: 1px solid #0f766e;
        background: #0f766e;
        color: white;
        font-weight: 700;
        min-height: 2.7rem;
    }
    .stButton > button:hover {
        border: 1px solid #115e59;
        background: #115e59;
        color: white;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 8px;
    }
    @media (max-width: 760px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .topbar {
            align-items: flex-start;
            gap: 0.5rem;
            flex-direction: column;
        }
        .hero {
            min-height: 52vh;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    _ensure_state()
    _render_landing()

    if st.session_state.chat_open:
        _render_chat()


def _render_landing() -> None:
    st.markdown(
        """
        <div class="topbar">
            <div class="brand">Colombia Comparte</div>
            <div class="nav-note">Emprendimiento, propósito y reconstrucción productiva</div>
        </div>
        <section class="hero">
            <div>
                <div class="eyebrow">Chatbot inteligente</div>
                <h1>Respuestas claras sobre Colombia Comparte, en segundos.</h1>
                <p>
                    Conoce la historia, los programas y el impacto de Colombia Comparte mediante un asistente
                    que consulta la base documental del proyecto antes de responder.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    action_col, example_col_1, example_col_2, example_col_3 = st.columns([1.25, 1, 1, 1])
    with action_col:
        if st.button("💬 Abrir chat", use_container_width=True):
            st.session_state.chat_open = True
            st.rerun()
    with example_col_1:
        if st.button("Qué es EDIFICA", use_container_width=True):
            _open_with_question("¿Qué es EDIFICA?")
    with example_col_2:
        if st.button("Impacto", use_container_width=True):
            _open_with_question("¿Cuál ha sido el impacto de Colombia Comparte?")
    with example_col_3:
        if st.button("Historia", use_container_width=True):
            _open_with_question("¿Cómo nació Colombia Comparte?")

    st.markdown(
        """
        <div class="trust-row">
            <div class="trust-item"><strong>RAG</strong>Busca primero en documentos reales.</div>
            <div class="trust-item"><strong>Qwen2.5</strong>Redacta respuestas naturales.</div>
            <div class="trust-item"><strong>FAISS</strong>Recupera contexto por similitud.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chat() -> None:
    st.markdown(
        """
        <div class="chat-shell">
            <div class="chat-header">
                <div class="chat-title">Asistente Colombia Comparte</div>
                <div class="chat-status">Disponible</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chat_area = st.container()
    with chat_area:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    pending_question = st.session_state.pop("pending_question", None)
    typed_question = st.chat_input("Pregúntame sobre Colombia Comparte")
    question = pending_question or typed_question
    if not question:
        st.markdown('<div class="hint">También puedes preguntar por EDIFICA, impacto, historia o programas.</div>', unsafe_allow_html=True)
        return

    _answer_question(question)


def _answer_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Pensando en darte muy buenas ideas..."):
            # The cached pipeline keeps FAISS and models loaded between chat turns.
            chatbot = load_chatbot()
            response = chatbot.ask(
                question,
                top_k=TOP_K,
                min_score=MIN_SCORE,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        st.markdown(response["answer"])

    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})


@st.cache_resource(show_spinner="Preparando el asistente de Colombia Comparte...")
def load_chatbot() -> RagChatbot:
    # local_files_only avoids accidental downloads during the live demo.
    return RagChatbot(
        chunks_path=PROJECT_ROOT / "data/processed/chunks.jsonl",
        index_path=PROJECT_ROOT / "indexes/faiss.index",
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        generation_model_name=DEFAULT_GENERATION_MODEL,
        local_files_only=True,
    )


def _ensure_state() -> None:
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hola, soy el asistente de Colombia Comparte. Estoy listo para ayudarte con preguntas sobre EDIFICA, impacto, historia y programas.",
            }
        ]


def _open_with_question(question: str) -> None:
    st.session_state.chat_open = True
    st.session_state.pending_question = question
    st.rerun()


if __name__ == "__main__":
    main()
