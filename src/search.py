import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
import google.generativeai as genai

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemma2-9b-it"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        
        # Initialize Gemini
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("[WARNING] GOOGLE_API_KEY not found in environment variables.")
        else:
            genai.configure(api_key=google_api_key)
        
        self.model_name = "gemini-2.0-flash"
        self.model = genai.GenerativeModel(self.model_name)
        print(f"[INFO] Gemini LLM initialized: {self.model_name}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        
        if not context:
            return "No relevant documents found."
            
        prompt = f"""You are a helpful and safe medical assistant. Use the following context to answer the user's question.
If the answer is not in the context, say you don't know. Do not make up medical information.
Always advise the user to consult a doctor for professional advice.

Context:
{context}

Question: {query}

Answer:"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)