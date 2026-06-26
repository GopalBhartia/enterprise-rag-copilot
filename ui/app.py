import requests
import streamlit as st

API_URL = "http://backend:8000/rag/ask"
FEEDBACK_URL = "http://backend:8000/feedback"

st.set_page_config(
    page_title="Enterprise RAG Copilot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Enterprise RAG Copilot")

# -----------------------------
# STATE INIT
# -----------------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "feedback_state" not in st.session_state:
    st.session_state.feedback_state = {}

if "pending_feedback" not in st.session_state:
    st.session_state.pending_feedback = None

# -----------------------------
# INPUT
# -----------------------------
example_questions = [
    "What is FastAPI?",
    "What is dependency injection in FastAPI?",
    "How does FastAPI handle startup and shutdown?",
    "What is Pydantic validation?",
    "What is a Docker container?",
]

selected = st.selectbox("Example Questions", [""] + example_questions)
query = st.text_input("Ask a question", value=selected)

ask_clicked = st.button("Ask")

# -----------------------------
# ASK HANDLER
# -----------------------------
if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    response = requests.post(
        API_URL,
        json={"query": query},
        timeout=120,
    )

    if response.status_code != 200:
        st.error("Backend failed")
        st.stop()

    st.session_state.result = response.json()

# -----------------------------
# HANDLE FEEDBACK FIRST (CRITICAL FIX)
# -----------------------------
result = st.session_state.result

if result:
    trace_id = result.get("trace_id", "unknown")

    # Handle pending feedback BEFORE rendering UI
    if st.session_state.pending_feedback:
        rating = st.session_state.pending_feedback

        try:
            requests.post(
                FEEDBACK_URL,
                json={
                    "trace_id": trace_id,
                    "query": query,
                    "rating": rating,
                },
                timeout=10,
            )
        except Exception:
            st.error("Feedback failed")
        else:
            st.session_state.feedback_state[trace_id] = rating
            st.toast("Feedback submitted 👍")

        st.session_state.pending_feedback = None
        st.rerun()

# -----------------------------
# DISPLAY RESULT
# -----------------------------
if result:
    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    for citation in result["citations"]:
        with st.expander(citation["source"]):
            st.write(f"Score: {citation['score']:.3f}")

    # -----------------------------
    # FEEDBACK UI (NOW RELIABLE)
    # -----------------------------
    st.divider()
    st.subheader("Was this answer helpful?")

    trace_id = result.get("trace_id", "unknown")

    if st.session_state.feedback_state.get(trace_id):
        st.success("✅ Feedback submitted. Thank you!")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Helpful", key=f"up_{trace_id}"):
                st.session_state.pending_feedback = "up"
                st.rerun()

        with col2:
            if st.button("👎 Not Helpful", key=f"down_{trace_id}"):
                st.session_state.pending_feedback = "down"
                st.rerun()
