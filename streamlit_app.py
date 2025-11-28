import streamlit as st
import os
from dotenv import load_dotenv
from src.search import RAGSearch

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Medical RAG Assistant", page_icon="🏥", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #667eea;
        color: white;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🏥 Medical RAG Assistant</h1><p>Ask medical questions based on clinical transcriptions dataset</p></div>', unsafe_allow_html=True)

# Initialize RAG Search
@st.cache_resource
def get_rag_search():
    return RAGSearch()

col1, col2 = st.columns([3, 1])

with col1:
    try:
        rag = get_rag_search()
        st.success("✅ System initialized successfully!")
    except Exception as e:
        st.error(f"❌ Error initializing system: {e}")
        st.stop()

with col2:
    st.metric("Vector Store", "FAISS", "Active")

st.markdown("### 💬 Ask Your Question")

# Sample questions
with st.expander("📋 Sample Medical Questions"):
    sample_qs = [
        "What are the symptoms of allergic rhinitis?",
        "What is sleep apnea?",
        "Describe laparoscopic gastric bypass procedure",
        "What is a 2-D Echocardiogram used for?"
    ]
    for q in sample_qs:
        st.markdown(f"• {q}")

query = st.text_input("Enter your medical question:", placeholder="e.g., What are the symptoms of pneumonia?")

if query:
    with st.spinner("🔍 Searching medical records and generating answer..."):
        try:
            response = rag.search_and_summarize(query)
            
            st.markdown("### 📝 Answer")
            st.markdown(f'<div class="answer-box">{response}</div>', unsafe_allow_html=True)
            
            with st.expander("📚 View Retrieved Context (Top 3 Sources)"):
                results = rag.vectorstore.query(query, top_k=3)
                for i, res in enumerate(results):
                    st.markdown(f"**Source {i+1}** (Similarity Score: {1 - res['distance']:.4f})")
                    text_preview = res['metadata'].get('text', '')[:500]
                    st.info(text_preview + "...")
                    st.markdown("---")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

st.markdown("---")
st.markdown('<div class="warning-box">⚠️ <strong>Medical Disclaimer:</strong> This is an AI assistant for informational purposes only. Always consult a qualified healthcare professional for medical advice.</div>', unsafe_allow_html=True)
