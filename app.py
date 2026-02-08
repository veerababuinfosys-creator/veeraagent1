import streamlit as st
import requests, os, time, faiss, numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

# ======================
# AUTH (Simple demo only)
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

MODEL_LIST = [
    "mistralai/mistral-7b-instruct",
    "openai/gpt-3.5-turbo",
    "meta-llama/llama-3-8b-instruct",
    "nousresearch/nous-hermes-2-mistral-7b"
]

# ======================
# PRODUCTION RULES
# ======================

PRODUCTION_RULES = """
You are generating enterprise-grade production code.

Mandatory rules:
- Use PostgreSQL (never SQLite)
- Use JWT or OAuth2 (never HTTPBasic)
- Use modular architecture
- Use environment variables for secrets
- Add logging
- Add error handling
- Use async where possible
- Follow clean architecture
Return clean, production-ready code only.
"""

# ======================
# SESSION STATE
# ======================

if "generated_files" not in st.session_state:
    st.session_state.generated_files = {}

if "project_architecture" not in st.session_state:
    st.session_state.project_architecture = None

if "project_name" not in st.session_state:
    st.session_state.project_name = ""

# ======================
# EMBEDDINGS
# ======================

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

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
# SAFE AI CALL
# ======================

def call_ai(messages, max_tokens=800, retries=2):
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
                        "Content-Type": "application/json"
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
                    return data["choices"][0]["message"]["content"]

                if "error" in data:
                    last_error = data["error"].get("message", str(data))
                    break

            except Exception as e:
                last_error = str(e)
                time.sleep(1)

    return f"❌ All AI models failed.\nLast error: {last_error}"

# ======================
# AGENTS
# ======================

def architect_agent(project):
    prompt = f"""
Design an enterprise-grade architecture.

Project:
{project}

Return:
1. Folder structure
2. Tech stack
3. Main components
"""
    return call_ai([
        {"role": "system", "content": "You are a senior software architect."},
        {"role": "user", "content": prompt}
    ], max_tokens=700)


def developer_agent(project, architecture):
    prompt = f"""
Project:
{project}

Architecture:
{architecture}

Generate full production-ready code.
Return files with clear separators:

### filename.py
<code>
"""
    return call_ai([
        {"role": "system", "content": PRODUCTION_RULES},
        {"role": "user", "content": prompt}
    ], max_tokens=2000)


def audit_project(code_text):
    audit_prompt = f"""
You are a senior production auditor.

Evaluate:
- Security
- Scalability
- Database choice
- Authentication
- Architecture

Give:
1. Score out of 10
2. Issues
3. Fix instructions

Project code:
{code_text}
"""
    return call_ai([
        {"role": "system", "content": "You are a strict enterprise auditor."},
        {"role": "user", "content": audit_prompt}
    ], max_tokens=800)


def auto_fix_project(code_text, audit_text):
    fix_prompt = f"""
Fix the following issues.

Audit:
{audit_text}

Project:
{code_text}

Return improved production-grade code.
"""
    return call_ai([
        {"role": "system", "content": PRODUCTION_RULES},
        {"role": "user", "content": fix_prompt}
    ], max_tokens=2000)

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
    q = st.text_input("Ask anything")
    if q:
        mem = search_memory(q)
        ans = call_ai([
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": mem + "\n" + q}
        ])
        st.markdown(ans)

# ---------- DOCUMENT MEMORY ----------
with tabs[1]:
    f = st.file_uploader("Upload PDF", type="pdf")
    if f:
        text = "".join(p.extract_text() for p in PdfReader(f).pages if p.extract_text())
        add_to_memory(text)
        st.success("Document stored")

# ---------- VOICE ----------
with tabs[2]:
    vq = st.text_input("Ask for voice reply")
    if vq:
        ans = call_ai([
            {"role": "user", "content": vq}
        ])
        st.markdown(ans)
        speak(ans)

# ---------- AUTOMATION ----------
with tabs[3]:
    task = st.text_area("Describe task")
    if st.button("Run"):
        res = call_ai([
            {"role": "system", "content": "Multi-agent automation."},
            {"role": "user", "content": task}
        ])
        st.markdown(res)

# ---------- CODE LAB ----------
with tabs[4]:
    st.subheader("💻 Enterprise Code Lab")

    project = st.text_area("Describe your project")

    if st.button("1️⃣ Generate Architecture"):
        arch = architect_agent(project)
        st.session_state.project_architecture = arch
        st.markdown(arch)

    if st.button("2️⃣ Generate Code"):
        if not st.session_state.project_architecture:
            st.warning("Generate architecture first")
        else:
            code = developer_agent(project, st.session_state.project_architecture)
            st.session_state.generated_files["project"] = code
            st.code(code)

    if st.button("3️⃣ Audit Project"):
        if "project" in st.session_state.generated_files:
            code = st.session_state.generated_files["project"]
            audit = audit_project(code)
            st.session_state.audit = audit
            st.markdown(audit)

    if st.button("4️⃣ Auto Fix to 100%"):
        if "audit" in st.session_state:
            code = st.session_state.generated_files["project"]
            fixed = auto_fix_project(code, st.session_state.audit)
            st.code(fixed)

# ---------- LOGOUT ----------
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
