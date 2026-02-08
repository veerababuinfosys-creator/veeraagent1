import streamlit as st
import requests, os, time, faiss, numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

# ======================
# AUTH
# ======================

USERS = {"admin": "1234", "veera": "ai2026"}

def login():
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if USERS.get(u) == p:
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid credentials")

if "user" not in st.session_state:
    login()
    st.stop()

# ======================
# CONFIG
# ======================

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    st.error("Missing OPENROUTER_API_KEY")
    st.stop()

URL = "https://openrouter.ai/api/v1/chat/completions"

# Auto-fallback models
MODEL_LIST = [
    "mistralai/mistral-7b-instruct",
    "openai/gpt-3.5-turbo",
    "meta-llama/llama-3-8b-instruct",
    "nousresearch/nous-hermes-2-mistral-7b"
]

# ======================
# SESSION STATE (CODE LAB)
# ======================

if "current_step" not in st.session_state:
    st.session_state.current_step = 0

if "generated_files" not in st.session_state:
    st.session_state.generated_files = {}

if "project_architecture" not in st.session_state:
    st.session_state.project_architecture = None

# ======================
# EMBEDDINGS
# ======================

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================
# VECTOR MEMORY
# ======================

DIM = 384
index = faiss.IndexFlatIP(DIM)
doc_chunks = []

def chunk_text(text, size=400, overlap=60):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-overlap)]

def add_to_memory(text):
    chunks = chunk_text(text)
    vecs = embedder.encode(chunks, normalize_embeddings=True)
    index.add(vecs.astype("float32"))
    doc_chunks.extend(chunks)

def search_memory(q, k=4):
    if index.ntotal == 0:
        return ""
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
    _, ids = index.search(qv, k)
    return "\n".join(doc_chunks[i] for i in ids[0])

# ======================
# SAFE AI CALL (AUTO FALLBACK)
# ======================

def call_ai(messages, max_tokens=800, retries=2):

    # Trim very long prompts
    if len(messages[-1]["content"]) > 4000:
        messages[-1]["content"] = messages[-1]["content"][:4000]

    last_error = "Unknown error"

    for model in MODEL_LIST:
        for _ in range(retries):
            try:
                r = requests.post(
                    URL,
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "Veera Enterprise AI"
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens
                    },
                    timeout=60
                )

                data = r.json()

                if "choices" in data:
                    reply = data["choices"][0]["message"]["content"]
                    return f"🧠 ({model})\n{reply}"

                if "error" in data:
                    last_error = data["error"].get("message", str(data))
                    break

            except Exception as e:
                last_error = str(e)
                time.sleep(1)

    return f"❌ All AI models failed.\nLast error: {last_error}"

# ======================
# MULTI AGENT WORKFLOW
# ======================

AGENTS = {
    "planner": "Break into steps",
    "researcher": "Provide technical insights",
    "architect": "Design scalable system",
    "qa": "Find risks and gaps"
}

def agent_workflow(q, context):
    outputs = []
    for role in AGENTS.values():
        outputs.append(call_ai([
            {"role": "system", "content": role},
            {"role": "user", "content": context + "\n" + q}
        ], max_tokens=400))

    return call_ai([
        {"role": "system", "content": "Create enterprise-grade final solution"},
        {"role": "user", "content": "\n".join(outputs)}
    ], max_tokens=700)

# ======================
# VOICE
# ======================

def speak(text):
    components.html(f"""
    <script>
    const msg = new SpeechSynthesisUtterance({repr(text)});
    speechSynthesis.speak(msg);
    </script>
    """, height=0)

# ======================
# CODE LAB HELPERS
# ======================

FILE_ORDER = [
    "database.py",
    "auth.py",
    "booking_engine.py",
    "seat_management.py",
    "payment.py",
    "admin.py",
    "app.py"
]

def next_file():
    idx = st.session_state.current_step - 1
    return FILE_ORDER[idx] if idx < len(FILE_ORDER) else None

# ======================
# UI
# ======================

st.set_page_config("Veera Enterprise AI Platform", layout="wide")
st.title("🚀 Veera Enterprise AI Platform")

tabs = st.tabs([
    "💬 AI Chat",
    "📄 Document Memory",
    "🎤 Voice Reply",
    "⚙️ Automation",
    "💻 Code Lab"
])

# ---------- AI CHAT ----------

with tabs[0]:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask anything...")
    if q:
        mem = search_memory(q)
        ans = agent_workflow(q, mem)
        st.session_state.chat += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans}
        ]
        st.rerun()

# ---------- DOCUMENT MEMORY ----------

with tabs[1]:
    f = st.file_uploader("Upload PDF", type="pdf")
    if f:
        text = "".join(p.extract_text() for p in PdfReader(f).pages if p.extract_text())
        add_to_memory(text)
        st.success("📚 Document stored in AI memory")

# ---------- VOICE ----------

with tabs[2]:
    vq = st.text_input("Ask for voice reply")
    if vq:
        mem = search_memory(vq)
        ans = agent_workflow(vq, mem)
        st.markdown(ans)
        speak(ans)

# ---------- AUTOMATION ----------

with tabs[3]:
    task = st.text_area("Describe task for AI agents")
    if st.button("Run Automation"):
        mem = search_memory(task)
        res = agent_workflow(task, mem)
        st.markdown(res)

# ---------- CODE LAB ----------

with tabs[4]:
    st.subheader("💻 Token-Safe Multi-Agent Code Lab")

    project = st.text_area(
        "Describe project once",
        placeholder="Build production-ready bus booking system like RedBus"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if st.button("🧱 Architecture"):
            if st.session_state.current_step > 0:
                st.info("Architecture already generated")
            else:
                arch = call_ai([
                    {"role": "system", "content": "You are a senior software architect"},
                    {"role": "user", "content": project + "\nCreate folder structure only"}
                ], max_tokens=600)
                st.session_state.project_architecture = arch
                st.session_state.current_step = 1
                st.code(arch)

    with c2:
        if st.button("📄 Next File"):
            if st.session_state.current_step == 0:
                st.warning("Generate architecture first")
            else:
                file = next_file()
                if not file:
                    st.info("All files generated")
                elif file in st.session_state.generated_files:
                    st.info(f"{file} already exists")
                else:
                    code = call_ai([
                        {"role": "system", "content": "Write clean production-ready Python code"},
                        {"role": "user", "content": f"Generate ONLY {file} for:\n{project}"}
                    ], max_tokens=1200)
                    st.session_state.generated_files[file] = code
                    st.session_state.current_step += 1
                    st.code(code, language="python")

# ---------- LOGOUT ----------

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
