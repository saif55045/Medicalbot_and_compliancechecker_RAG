import json
import os
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents

load_dotenv()

class ComplianceChecker:
    def __init__(self, 
                 rules_path: str = "compliance_rules.json", 
                 data_dir: str = "data", 
                 persist_dir: str = "faiss_store_policy",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 model_name: str = "gemini-2.0-flash"):
        
        self.rules_path = rules_path
        self.persist_dir = persist_dir
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        
        # Initialize Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Load Rules
        with open(rules_path, 'r') as f:
            self.rules = json.load(f)
            
        # Build/Load Vector Store
        self._initialize_vectorstore(data_dir)

    def _initialize_vectorstore(self, data_dir):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print(f"[INFO] Building vector store for Policy Compliance from {data_dir}...")
            
            # Directly load only the Task 2 PDF
            pdf_path = os.path.join(data_dir, "Task2_data.pdf")
            if os.path.exists(pdf_path):
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                policy_docs = loader.load()
                print(f"[INFO] Loaded {len(policy_docs)} pages from {pdf_path}")
            else:
                print(f"[WARNING] {pdf_path} not found. Falling back to loading all docs.")
                all_docs = load_all_documents(data_dir)
                policy_docs = [d for d in all_docs if "Task2_data.pdf" in d.metadata.get("source", "")]

            if not policy_docs:
                 raise ValueError("No documents found to build vector store.")

            self.vectorstore.build_from_documents(policy_docs)
        else:
            print(f"[INFO] Loading existing vector store from {self.persist_dir}...")
            self.vectorstore.load()

    def check_compliance(self, rule: Dict) -> Dict:
        query = f"policy regarding {rule['category']} {rule['rule']}"
        results = self.vectorstore.query(query, top_k=3)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        
        prompt = f"""You are a strict Compliance Officer. Evaluate if the company policy text provided below complies with the following rule.

Rule Category: {rule['category']}
Rule: "{rule['rule']}"

Policy Context Retrieved:
{context}

Task:
1. Determine if the policy is "Compliant", "Non-Compliant", or "Missing" (if not mentioned).
2. Provide a brief "Evidence" quote from the context if found.
3. Suggest "Remediation" if non-compliant or missing.

Output Format (JSON):
{{
  "status": "Compliant/Non-Compliant/Missing",
  "evidence": "...",
  "remediation": "..."
}}
"""
        try:
            response = self.model.generate_content(prompt)
            # Clean up json block if present
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:-3]
            elif text.startswith("```"):
                text = text[3:-3]
            
            result = json.loads(text)
            return result
        except Exception as e:
            return {"status": "Error", "evidence": str(e), "remediation": "Check logs"}

    def run_audit(self):
        audit_results = []
        print(f"Starting audit on {len(self.rules)} rules...")
        for rule in self.rules:
            print(f"Checking Rule {rule['id']}...")
            compliance = self.check_compliance(rule)
            audit_results.append({
                "Rule ID": rule['id'],
                "Category": rule['category'],
                "Rule": rule['rule'],
                "Severity": rule['severity'],
                "Status": compliance.get("status", "Unknown"),
                "Evidence": compliance.get("evidence", ""),
                "Remediation": compliance.get("remediation", "")
            })
        
        return pd.DataFrame(audit_results)

if __name__ == "__main__":
    checker = ComplianceChecker()
    df = checker.run_audit()
    print(df)
    df.to_csv("compliance_report.csv", index=False)
