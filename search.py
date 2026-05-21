import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
try:
     from langchain_community.vectorstores import FAISS as FaissVectorStore
except Exception:
     # Fallback: try local import if available
     local_dir = os.path.dirname(os.path.abspath(__file__))
     if local_dir not in sys.path:
          sys.path.append(local_dir)
     try:
        from langchain_community.vectorstores import FAISS as FaissVectorStore
     except Exception:
          FaissVectorStore = None

load_dotenv()


class RAGSearch:
    def __init__(self,persist_dir:str = 'faiss_store', embedding_model :str ="all-MiniLM-L6-v2", llm_model: str = "gemma2-9b-it"):
         self.vectorstore  = FaissVectorStore(persist_dir,embedding_model, allow_dangerous_deserialization=True)

         faiss_path = os.path.join(persist_dir,"faiss.index")
         meta_path = os.path.join(persist_dir,"metadata.pkl")
         if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
              from data_loader import load_all_documents
              docs = load_all_documents("data")
              self.vectorstore.build_from_documents(docs)
         else:
              self.vectorstore.load()
         self.llm = ChatGroq(groq_api_key="groq_api_key", model_name="openai/gpt-oss-120b")
         print(f"[INFO] Groq LLm intialized: {llm_model}")

    def search_and_summarize(self, query:str , top_k :int = 5 ) -> str:
         results = self.vectorstore.query(query,top_k=top_k)
         texts=[r["metadata"].get("text","") for r in results if r["metadata"]]
         context = "\n\n".join(texts)
         if not context :
              return "No relevant documents found"
         prompt = f"""Summarise the following context for the query :'{query}'\n\n Context :\n {context} \n\n Summary:"""
         response = self.llm.invoke([prompt])
         return response.content
    
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is cloud computing?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)