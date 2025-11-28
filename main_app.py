import streamlit as st

st.set_page_config(page_title="GenAI Assignment 4", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .task-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🤖 Generative AI Assignment 4</h1><h3>Retrieval Augmented Generation (RAG) Systems</h3><p>Powered by LangChain, FAISS & Google Gemini API</p></div>', unsafe_allow_html=True)

st.markdown("""
<div class="task-card">
<h4>📋 Project Overview</h4>
<p>This project implements two production-ready RAG systems demonstrating advanced document retrieval and AI-powered question answering capabilities.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="task-card">
    <h4>🏥 Task 1: Medical RAG QA System</h4>
    <ul>
        <li>Dataset: 5000+ clinical transcriptions</li>
        <li>Technology: FAISS vector store</li>
        <li>Features: Semantic search, context-aware responses</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="task-card">
    <h4>🛡️ Task 2: Policy Compliance Checker</h4>
    <ul>
        <li>Dataset: Company policy documents</li>
        <li>Technology: Rule-based compliance engine</li>
        <li>Features: Automated auditing, Q&A agent</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("# 🧭 Navigation")
task = st.sidebar.radio("Select Application", ["🏠 Home", "🏥 Medical QA", "🛡️ Compliance Checker"])

if task == "🏠 Home":
    st.info("👆 Please select a task from the sidebar to begin.")

elif task == "🏥 Medical QA":
    # We can run the other script or import its main function. 
    # Importing is better but streamlit scripts are weird. 
    # Let's just use os.system or similar if we want to keep them separate, 
    # OR better, refactor them into functions.
    # For simplicity in this environment, I'll just point the user to run the specific files 
    # or I can try to inline the code.
    
    # Let's try to execute the file content.
    with open("streamlit_app.py", "r", encoding='utf-8') as f:
        code = f.read()
        # Remove set_page_config from the code as it can only be called once
        code = code.replace("st.set_page_config", "# st.set_page_config")
        exec(code, {"__name__": "__main__"})

elif task == "🛡️ Compliance Checker":
    with open("task2_app.py", "r", encoding='utf-8') as f:
        code = f.read()
        code = code.replace("st.set_page_config", "# st.set_page_config")
        exec(code, {"__name__": "__main__"})
