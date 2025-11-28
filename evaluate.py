import time
import pandas as pd
from src.search import RAGSearch

def evaluate():
    print("Initializing RAG System for Evaluation...")
    rag = RAGSearch()
    
    queries = [
        "What are the symptoms of allergic rhinitis?",
        "Describe the procedure for laparoscopic gastric bypass.",
        "What are the risks of gastric bypass surgery?",
        "What is a 2-D Echocardiogram used for?",
        "Symptoms of mitral regurgitation?",
        "Treatment for chronic back pain?",
        "What is sleep apnea?",
        "Medications for high cholesterol?",
        "Signs of a heart attack?",
        "What is a colonoscopy?",
        "Treatment for carpal tunnel syndrome?",
        "Symptoms of pneumonia?",
        "What is degenerative disc disease?",
        "Management of type 2 diabetes?",
        "What is a hysterectomy?",
        "Symptoms of anxiety disorder?",
        "Treatment for rotator cuff tear?",
        "What is a CT scan used for?",
        "Symptoms of kidney stones?",
        "What is cataract surgery?",
        "Treatment for migraine headaches?",
        "What is a hernia repair?",
        "Symptoms of hypothyroidism?",
        "What is a knee replacement?",
        "Treatment for asthma?",
        "What is a biopsy?",
        "Symptoms of anemia?",
        "What is a lumbar puncture?",
        "Treatment for depression?",
        "What is an MRI used for?"
    ]
    
    results = []
    
    print(f"Starting evaluation on {len(queries)} queries...")
    
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: {query}")
        start_time = time.time()
        try:
            response = rag.search_and_summarize(query)
            elapsed_time = time.time() - start_time
            results.append({
                "query": query,
                "response": response,
                "time_taken": elapsed_time
            })
        except Exception as e:
            print(f"Error on query '{query}': {e}")
            results.append({
                "query": query,
                "response": f"ERROR: {e}",
                "time_taken": 0
            })
            
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation complete. Results saved to 'evaluation_results.csv'.")

if __name__ == "__main__":
    evaluate()
