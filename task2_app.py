import streamlit as st
import pandas as pd
from src.compliance import ComplianceChecker

st.set_page_config(page_title="Policy Compliance Checker", page_icon="🛡️", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #11998e;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .compliant {
        color: #28a745;
        font-weight: bold;
    }
    .non-compliant {
        color: #dc3545;
        font-weight: bold;
    }
    .missing {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🛡️ Policy Compliance Checker & Assistant</h1><p>Automated policy auditing and intelligent Q&A system</p></div>', unsafe_allow_html=True)

@st.cache_resource
def get_checker():
    return ComplianceChecker()

try:
    checker = get_checker()
    st.success("Compliance System Initialized")
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()

tab1, tab2 = st.tabs(["📊 Compliance Audit", "💬 Policy Chat Agent"])

with tab1:
    st.header("🔍 Automated Compliance Audit")
    st.markdown("Evaluate policy documents against 15 predefined security and compliance rules.")
    
    if st.button("🚀 Run Compliance Check", use_container_width=True):
        with st.spinner("🔄 Analyzing policies against rules..."):
            df = checker.run_audit()
            
            # Metrics
            compliant_count = df[df['Status'] == 'Compliant'].shape[0]
            non_compliant_count = df[df['Status'] == 'Non-Compliant'].shape[0]
            missing_count = df[df['Status'] == 'Missing'].shape[0]
            total_rules = len(df)
            compliance_rate = (compliant_count / total_rules * 100) if total_rules > 0 else 0
            
            st.markdown("### 📊 Audit Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("✅ Compliant", compliant_count, f"{compliant_count/total_rules*100:.1f}%")
            col2.metric("❌ Non-Compliant", non_compliant_count, f"{non_compliant_count/total_rules*100:.1f}%")
            col3.metric("⚠️ Missing", missing_count, f"{missing_count/total_rules*100:.1f}%")
            col4.metric("📈 Compliance Rate", f"{compliance_rate:.1f}%")
            
            st.markdown("### 📋 Detailed Results")
            st.dataframe(df, use_container_width=True, height=400)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Report CSV", csv, "compliance_report.csv", "text/csv", use_container_width=True)

    st.subheader("Defined Rules")
    st.json(checker.rules)

with tab2:
    st.header("💬 Policy Q&A Agent")
    st.markdown("Get instant answers from your policy documents using AI-powered search.")
    
    # Initialize session state for query
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ""

    sample_questions = [
        "What is the policy on remote work?",
        "How often should passwords be changed?",
        "Are we allowed to use personal devices for work?",
        "What is the procedure for reporting a security incident?",
        "Can I forward work emails to my personal account?"
    ]
    
    st.markdown("**💡 Sample Questions (Click to use):**")
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        if cols[i % 2].button(f"🔹 {q}", key=f"btn_{i}", use_container_width=True):
            st.session_state["query_input"] = q
            st.rerun()

    query = st.text_input("Ask a question:", key="query_input")
    
    if query:
        with st.spinner("Searching policies..."):
            # Reuse the vector store from the checker
            results = checker.vectorstore.query(query, top_k=5)
            texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
            context = "\n\n".join(texts)
            
            if context:
                prompt = f"""You are a Policy Expert. Answer the user's question based ONLY on the following policy context.
                
Context:
{context}

Question: {query}

Answer:"""
                try:
                    response = checker.model.generate_content(prompt)
                    st.markdown("### 📝 Answer")
                    st.info(response.text)
                    
                    with st.expander("View Source Context"):
                        # Clean up context for display: replace newlines with spaces within chunks
                        # but keep separation between chunks
                        cleaned_texts = [t.replace("\n", " ").strip() for t in texts]
                        formatted_context = "\n\n---\n\n".join(cleaned_texts)
                        st.markdown(formatted_context)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("No relevant policy information found.")
