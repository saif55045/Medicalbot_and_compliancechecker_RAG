# 🤖 Advanced RAG Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This project demonstrates the implementation of two production-ready **Retrieval Augmented Generation (RAG)** systems using state-of-the-art technologies including LangChain, FAISS vector store, and Google's Gemini API. The systems showcase advanced document retrieval, semantic search, and AI-powered question answering capabilities.

### 🎯 Key Features

- **Dual RAG Applications**: Medical Q&A and Policy Compliance systems
- **Advanced Embeddings**: Sentence Transformers (all-MiniLM-L6-v2) for semantic understanding
- **Efficient Vector Search**: FAISS for lightning-fast similarity search
- **Modern UI**: Beautiful Streamlit interfaces with custom styling
- **Production-Ready**: Comprehensive error handling and logging
- **Scalable Architecture**: Modular design for easy extension

## Project Structure

```
├── data/                   # Dataset storage (CSV and PDF)
├── src/                    # Source code for RAG pipelines
│   ├── compliance.py       # Task 2 logic
│   ├── data_loader.py      # Document loading utilities
│   ├── embedding.py        # Embedding generation
│   ├── search.py           # Task 1 RAG logic
│   └── vectorstore.py      # FAISS vector store management
├── app.py                  # (Deprecated)
├── compliance_rules.json   # Rules for Task 2
├── evaluate.py             # Evaluation script for Task 1
├── main_app.py             # Main Streamlit entry point
├── requirements.txt        # Python dependencies
├── streamlit_app.py        # Task 1 UI
└── task2_app.py            # Task 2 UI
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐              ┌──────────────────┐   │
│  │   Medical QA     │              │   Compliance     │   │
│  │   (Task 1)       │              │   Checker        │   │
│  └────────┬─────────┘              └────────┬─────────┘   │
│           │                                 │             │
│  ┌────────▼─────────┐              ┌───────▼──────────┐   │
│  │  RAGSearch       │              │ ComplianceChecker│   │
│  │  (search.py)     │              │ (compliance.py)  │   │
│  └────────┬─────────┘              └───────┬──────────┘   │
│           │                                 │             │
│  ┌────────▼──────────────────────────────┬─┘             │
│  │      FAISS Vector Store               │               │
│  │      (vectorstore.py)                 │               │
│  └────────┬──────────────────────────────┘               │
│           │                                               │
│  ┌────────▼──────────────────────────────┐               │
│  │   Sentence Transformers Embeddings    │               │
│  │   (embedding.py)                      │               │
│  └───────────────────────────────────────┘               │
│                                                           │
│  ┌───────────────────────────────────────┐               │
│  │      Google Gemini API               │               │
│  │      (LLM Response Generation)        │               │
│  └───────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|----------|
| **LLM** | Google Gemini 2.0 Flash | Natural language generation |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Text vectorization |
| **Vector DB** | FAISS | Similarity search & retrieval |
| **Framework** | LangChain | RAG orchestration |
| **UI** | Streamlit | Web interface |
| **Data Processing** | Pandas, PyPDF | Document loading & processing |

## 📦 Prerequisites

1.  **Python 3.10+** - Required for latest dependencies
2.  **Google Gemini API Key** - Free tier available at [Google AI Studio](https://aistudio.google.com/)
3.  **4GB+ RAM** - For vector store operations
4.  **2GB+ Storage** - For datasets and embeddings

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd RAG
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `langchain` - RAG framework
- `langchain-community` - Document loaders
- `google-generativeai` - Gemini API
- `sentence-transformers` - Embeddings
- `faiss-cpu` - Vector store
- `streamlit` - UI framework
- `pandas` - Data processing

### 4. Configure Environment
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_api_key_here
```

**Getting your API key:**
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API Key"
4. Copy and paste into `.env` file

### 5. Prepare Data
Ensure the following files are in the `data/` directory:
- `Task1_data.csv` - Medical transcriptions dataset
- `Task2_data.pdf` - Policy document

### 6. Run the Application
```bash
streamlit run main_app.py
```
The app will open at `http://localhost:8501`

## 🏥 Task 1: Medical RAG QA System

### Overview
An intelligent medical assistant that provides evidence-based answers to clinical questions by retrieving relevant information from a database of 5,000+ medical transcriptions.

### Features
- ✅ **Semantic Search**: Uses embeddings to find contextually similar medical records
- ✅ **Context-Aware Responses**: Retrieves top-K relevant documents before generating answers
- ✅ **Source Attribution**: Shows which medical records were used to generate the answer
- ✅ **Safety First**: Includes disclaimers and advises consulting medical professionals

### Dataset
- **Source**: [Medical Transcriptions - Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
- **Size**: 4,999 clinical documents
- **Content**: Medical histories, procedures, diagnoses across 40+ specialties
- **Processing**: Chunked into 29,894 segments for optimal retrieval

### Running the Application
```bash
streamlit run main_app.py
# Select "🏥 Medical QA" from the sidebar
```

### Sample Questions
- "What are the symptoms of allergic rhinitis?"
- "Describe the procedure for laparoscopic gastric bypass"
- "What is a 2-D Echocardiogram used for?"
- "What medications are used for high cholesterol?"

### Evaluation
Run comprehensive evaluation on 30 medical queries:
```bash
python evaluate.py
```

**Output:** `evaluation_results.csv` containing:
- Query text
- Generated response
- Response time
- Source documents used

### Performance Metrics
- **Average Response Time**: ~2-3 seconds
- **Vector Store Size**: 29,894 chunks
- **Embedding Dimension**: 384
- **Top-K Retrieval**: 5 documents

## 🛡️ Task 2: Policy Compliance Checker

### Overview
An automated compliance auditing system that evaluates company policies against predefined security and governance rules, providing evidence-based compliance reports.

### Features
- ✅ **Automated Auditing**: Evaluates 15 compliance rules automatically
- ✅ **Evidence-Based Results**: Provides direct quotes from policy documents
- ✅ **Remediation Suggestions**: AI-generated recommendations for non-compliant policies
- ✅ **Interactive Q&A**: Chat interface for policy exploration
- ✅ **Compliance Dashboard**: Visual metrics and downloadable reports

### Dataset
- **Source**: [CUAD - Atticus Project](https://www.atticusprojectai.org/cuad)
- **Format**: PDF policy documents
- **Processing**: Extracted and chunked into 52 segments

### Compliance Rules
The system checks against 15 rules across categories:

| Category | Examples |
|----------|----------|
| **Data Security** | Encryption standards, data protection |
| **Access Control** | MFA requirements, authentication |
| **Password Policy** | Complexity, rotation frequency |
| **Remote Work** | VPN usage, secure connections |
| **Incident Response** | Reporting procedures, timelines |
| **Vendor Management** | NDA requirements, third-party access |

### Running the Application
```bash
streamlit run main_app.py
# Select "🛡️ Compliance Checker" from the sidebar
```

### Usage

#### 1. Automated Audit
1. Click "🚀 Run Compliance Check"
2. View compliance metrics dashboard
3. Review detailed findings table
4. Download CSV report

#### 2. Policy Q&A
1. Type a question or click sample questions
2. Get AI-powered answers with source references
3. View retrieved context for verification

### Sample Questions
- "What is the policy on remote work?"
- "How often should passwords be changed?"
- "Are personal devices allowed for work?"
- "What is the security incident reporting procedure?"

### Output Format
Compliance report includes:
- **Rule ID**: Unique identifier
- **Category**: Policy domain
- **Status**: Compliant / Non-Compliant / Missing
- **Evidence**: Direct quotes from policy
- **Remediation**: Suggested fixes
- **Severity**: High / Medium / Low

## 📊 Project Statistics

### Task 1: Medical QA
- **Documents**: 4,999 medical transcriptions
- **Chunks**: 29,894 text segments
- **Vector Store**: 11.4 MB
- **Specialties**: 40+ medical fields
- **Evaluation Queries**: 30

### Task 2: Compliance
- **Documents**: 11 PDF pages
- **Chunks**: 52 text segments
- **Rules**: 15 compliance checks
- **Categories**: 10 policy domains
- **Severity Levels**: 3 (High/Medium/Low)

## 🎨 UI Features

### Design Elements
- **Gradient Headers**: Eye-catching color schemes
- **Responsive Layout**: Optimized for all screen sizes
- **Interactive Components**: Buttons, expanders, metrics
- **Custom Styling**: CSS-enhanced aesthetics
- **Sample Questions**: Quick-start templates
- **Progress Indicators**: Real-time feedback

### User Experience
- **Intuitive Navigation**: Clear sidebar menu
- **Fast Loading**: Cached resources for speed
- **Error Handling**: Graceful failure messages
- **Download Options**: Export results as CSV
- **Source Attribution**: Transparency in AI responses

## 🔬 Technical Implementation

### Document Processing Pipeline
```python
1. Load Documents (CSV/PDF/TXT/etc.)
   ↓
2. Split into Chunks (RecursiveCharacterTextSplitter)
   ↓
3. Generate Embeddings (Sentence Transformers)
   ↓
4. Store in FAISS (Vector Database)
   ↓
5. Query & Retrieve (Semantic Search)
   ↓
6. Generate Response (Gemini API)
```

### Key Components

**`src/data_loader.py`**
- Supports multiple formats: PDF, CSV, TXT, DOCX, JSON
- Automatic encoding detection
- Metadata preservation

**`src/embedding.py`**
- Sentence Transformers integration
- Configurable chunk sizes
- Batch processing for efficiency

**`src/vectorstore.py`**
- FAISS index management
- Persistent storage
- Fast similarity search

**`src/search.py`** (Task 1)
- RAG pipeline orchestration
- Context retrieval
- Gemini integration

**`src/compliance.py`** (Task 2)
- Rule-based evaluation
- Evidence extraction
- Remediation generation

## 🧪 Testing & Evaluation

### Task 1 Evaluation
```bash
python evaluate.py
```

Tests 30 diverse medical queries across:
- Symptoms & diagnosis
- Procedures & treatments
- Medical equipment
- Chronic conditions

### Metrics Collected
- Response accuracy
- Retrieval relevance
- Generation time
- Source quality

## 📝 File Structure Explained

```
RAG/
│
├── data/                          # Dataset storage
│   ├── Task1_data.csv            # Medical transcriptions
│   └── Task2_data.pdf            # Policy documents
│
├── src/                           # Core application logic
│   ├── data_loader.py            # Multi-format document loader
│   ├── embedding.py              # Text chunking & embeddings
│   ├── vectorstore.py            # FAISS vector database
│   ├── search.py                 # Task 1 RAG pipeline
│   └── compliance.py             # Task 2 compliance engine
│
├── faiss_store/                   # Task 1 vector store
│   ├── faiss.index               # FAISS index file
│   └── metadata.pkl              # Document metadata
│
├── faiss_store_policy/            # Task 2 vector store
│   ├── faiss.index
│   └── metadata.pkl
│
├── main_app.py                    # Main Streamlit entry point
├── streamlit_app.py              # Task 1 UI
├── task2_app.py                  # Task 2 UI
├── evaluate.py                    # Task 1 evaluation script
├── compliance_rules.json          # Task 2 rule definitions
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (API keys)
└── README.md                     # This file
```

## 🚀 Deployment Guide

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add `GOOGLE_API_KEY` to secrets
5. Deploy!

### Hugging Face Spaces
1. Create new Space on [huggingface.co](https://huggingface.co)
2. Select Streamlit SDK
3. Upload code files
4. Configure secrets
5. Launch

### Local Network Deployment
```bash
streamlit run main_app.py --server.port 8080 --server.address 0.0.0.0
```

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `GOOGLE_API_KEY not found`
- **Solution**: Ensure `.env` file exists with valid API key

**Issue**: `No module named 'faiss'`
- **Solution**: `pip install faiss-cpu`

**Issue**: Vector store takes too long to build
- **Solution**: Reduce chunk size or use pre-built store

**Issue**: Out of memory errors
- **Solution**: Process documents in batches

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Gemini API Docs](https://ai.google.dev/docs)
- [Streamlit Docs](https://docs.streamlit.io)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✉️ Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using LangChain, FAISS, and Google Gemini**
