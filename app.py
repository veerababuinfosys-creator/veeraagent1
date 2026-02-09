import streamlit as st
import requests, os, time, faiss, numpy as np, logging
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

# ======================
# LOGGING
# ======================
logging.basicConfig(level=logging.INFO)

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

# Stable model fallback list (no :free suffix)
MODEL_LIST = [
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "mistralai/mistral-7b-instruct"
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
    return "\n".join(doc_chunks[i] for i in ids[0] if i < len(doc_chunks))

# ======================
# SAFE AI CALL WITH FALLBACK
# ======================
def call_ai(messages, max_tokens=800, retries=2):
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
                    logging.warning(f"{model} failed: {last_error}")
                    break

            except Exception as e:
                last_error = str(e)
                logging.warning(f"{model} exception: {last_error}")
                time.sleep(1)

    return f"❌ All AI models failed.\nLast error: {last_error}"

# ======================
# AGENTS (4-AGENT SYSTEM)
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
# UI
# ======================
st.set_page_config("VeeraAgent1", layout="wide")
st.title("🚀 VeeraAgent1 – 4-Agent AI Code Platform")

project = st.text_area("Describe your project")

if st.button("1️⃣ Generate Architecture"):
    arch = architect_agent(project)
    st.session_state.arch = arch
    st.markdown(arch)

if st.button("2️⃣ Generate Code"):
    if "arch" not in st.session_state:
        st.warning("Generate architecture first")
    else:
        code = developer_agent(project, st.session_state.arch)
        st.session_state.code = code
        st.code(code)

if st.button("3️⃣ Audit Project"):
    if "code" in st.session_state:
        audit = audit_project(st.session_state.code)
        st.session_state.audit = audit
        st.markdown(audit)

if st.button("4️⃣ Auto Fix"):
    if "audit" in st.session_state:
        fixed = auto_fix_project(st.session_state.code, st.session_state.audit)
        st.code(fixed)

# LOGOUT
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
