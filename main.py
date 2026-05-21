from src.data_loader import load_all_documents
from src.vectorbase import FaissVectorStore
from src.search import RAGSearch

if __name__  == "__main__":
    docs= load_all_documents("data")
    store= FaissVectorStore("faiss_store")
    store.load()
    rag_search = RAGSearch()
    query  = "what  is cloud computing "
    summary = rag_search.search_and_summarise(query,top=3)
    print("summary:",summary)