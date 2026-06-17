import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from vectorbase import FaissVectorStore

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b"
    ):

        # Initialize custom FAISS vector store
        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model
        )

        # Check whether index already exists
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            print("[INFO] Loading existing vector store...")
            self.vectorstore.load()
        else:
            print("[INFO] Building new vector store...")
            from data_loader import load_all_documents

            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)

        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=llm_model
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        results = self.vectorstore.query(
            query_text=query,
            top_k=top_k
        )

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""
You are a helpful assistant.

Question:
{query}

Context:
{context}

Provide a concise answer based only on the context.
"""

        response = self.llm.invoke(prompt)

        return response.content


if __name__ == "__main__":
    rag_search = RAGSearch()

    query = "What is cloud computing?"

    summary = rag_search.search_and_summarize(
        query=query,
        top_k=3
    )

    print("\nSummary:\n")
    print(summary)