import streamlit as st
import requests, os, time, faiss, numpy as np, logging, tempfile, zipfile
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

# ======================
# LOGGING
# ======================
logging.basicConfig(level=logging.INFO)

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
# CONFIG (GROK API)
# ======================
API_KEY = os.getenv("GROK_API_KEY")
if not API_KEY:
    st.error("Missing GROK_API_KEY")
    st.stop()

URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-2-mini"

# ======================
# EMBEDDINGS
# ======================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

DIM = 384
if "index" not in st.session_state:
    st.session_state.index = faiss.IndexFlatIP(DIM)
    st.session_state.doc_chunks = []

def chunk_text(text, size=400, overlap=60):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-overlap)]

def add_to_memory(text):
    chunks = chunk_text(text)
    vecs = embedder.encode(chunks, normalize_embeddings=True)
    st.session_state.index.add(vecs.astype("float32"))
    st.session_state.doc_chunks.extend(chunks)

def search_memory(q, k=4):
    if st.session_state.index.ntotal == 0:
        return ""
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
    _, ids = st.session_state.index.search(qv, k)
    return "\n".join(
        st.session_state.doc_chunks[i]
        for i in ids[0]
        if i < len(st.session_state.doc_chunks)
    )

# ======================
# AI CALL (GROK)
# ======================
def call_ai(messages, max_tokens=800):
    try:
        r = requests.post(
            URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens
            },
            timeout=60
        )

        data = r.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        if "error" in data:
            return f"❌ AI Error: {data['error'].get('message')}"

    except Exception as e:
        return f"❌ Request failed: {str(e)}"

    return "❌ Unknown AI error"

# ======================
# PRODUCTION RULES
# ======================
PRODUCTION_RULES = """
You are generating enterprise-grade production code.
- Use PostgreSQL
- Use JWT/OAuth2
- Modular architecture
- Env variables for secrets
- Logging and error handling
"""

# ======================
# 4 AGENTS
# ======================
def architect_agent(project, memory=""):
    prompt = f"Context:\n{memory}\n\nProject:\n{project}\n\nDesign architecture."
    return call_ai([
        {"role": "system", "content": "You are a senior architect."},
        {"role": "user", "content": prompt}
    ], 700)

def developer_agent(project, architecture):
    prompt = f"Project:\n{project}\nArchitecture:\n{architecture}\nGenerate code."
    return call_ai([
        {"role": "system", "content": PRODUCTION_RULES},
        {"role": "user", "content": prompt}
    ], 2000)

def audit_project(code_text):
    prompt = f"Audit this project and score it:\n{code_text}"
    return call_ai([
        {"role": "system", "content": "You are a strict auditor."},
        {"role": "user", "content": prompt}
    ], 800)

def auto_fix_project(code_text, audit_text):
    prompt = f"Fix issues.\nAudit:\n{audit_text}\nProject:\n{code_text}"
    return call_ai([
        {"role": "system", "content": PRODUCTION_RULES},
        {"role": "user", "content": prompt}
    ], 2000)

# ======================
# PROJECT ZIP
# ======================
def create_project_zip(code):
    temp_dir = tempfile.mkdtemp()
    app_path = os.path.join(temp_dir, "app.py")
    with open(app_path, "w") as f:
        f.write(code)

    zip_path = os.path.join(temp_dir, "project.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(app_path, "app.py")
    return zip_path

# ======================
# VOICE
# ======================
def speak_browser(text):
    components.html(f"""
    <script>
    const msg = new SpeechSynthesisUtterance({repr(text)});
    window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

# ======================
# UI
# ======================
st.set_page_config("Veera Enterprise AI", layout="wide")
st.title("🚀 Veera Enterprise AI Platform")

tabs = st.tabs([
    "💬 AI Chat",
    "🧠 Project Builder",
    "📄 Document Memory",
    "⚙️ Automation",
    "🧠 Project Generator"
])

# ======================
# AI CHAT
# ======================
with tabs[0]:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask anything...")
    if q:
        mem = search_memory(q)
        ans = call_ai([
            {"role": "system", "content": "You are a helpful enterprise AI."},
            {"role": "user", "content": mem + "\n" + q}
        ])
        st.session_state.chat += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans}
        ]
        st.rerun()

# ======================
# PROJECT BUILDER
# ======================
with tabs[1]:
    project = st.text_area("Describe your project")

    if st.button("1️⃣ Generate Architecture"):
        memory = search_memory(project)
        arch = architect_agent(project, memory)
        st.session_state.arch = arch
        st.markdown(arch)

    if st.button("2️⃣ Generate Code"):
        if "arch" in st.session_state:
            code = developer_agent(project, st.session_state.arch)
            st.session_state.code = code
            st.code(code)

    if st.button("3️⃣ Audit"):
        if "code" in st.session_state:
            audit = audit_project(st.session_state.code)
            st.session_state.audit = audit
            st.markdown(audit)

    if st.button("4️⃣ Auto Fix"):
        if "audit" in st.session_state:
            fixed = auto_fix_project(st.session_state.code, st.session_state.audit)
            st.code(fixed)

# ======================
# MEMORY
# ======================
with tabs[2]:
    f = st.file_uploader("Upload PDF", type="pdf")
    if f:
        text = "".join(
            p.extract_text() for p in PdfReader(f).pages if p.extract_text()
        )
        add_to_memory(text)
        st.success("Document stored")

# ======================
# AUTOMATION
# ======================
with tabs[3]:
    task = st.text_area("Describe task")
    if st.button("Run Automation"):
        mem = search_memory(task)
        result = call_ai([
            {"role": "system", "content": "You are an automation AI."},
            {"role": "user", "content": mem + "\n" + task}
        ])
        st.markdown(result)

# ======================
# PROJECT GENERATOR
# ======================
with tabs[4]:
    idea = st.text_area("Describe project idea")
    if st.button("Generate Project"):
        code = call_ai([
            {"role": "system", "content": PRODUCTION_RULES},
            {"role": "user", "content": idea}
        ], 2000)

        zip_path = create_project_zip(code)
        st.code(code)

        with open(zip_path, "rb") as f:
            st.download_button(
                "⬇️ Download Project",
                data=f,
                file_name="project.zip"
            )

# ======================
# LOGOUT
# ======================
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
