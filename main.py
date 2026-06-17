from data_loader import load_all_documents
from vectorbase import FaissVectorStore
from search import RAGSearch

if __name__  == "__main__":
    docs= load_all_documents("data")
    store= FaissVectorStore("faiss_store")
    store.load()
    rag_search = RAGSearch()
    query  = "what is newton thrid law wit formula "
    summary = rag_search.search_and_summarize(query,top_k=3)
    print("summary:",summary)