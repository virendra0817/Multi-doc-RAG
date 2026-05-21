from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: list[Any]):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        print(
            f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks."
        )

        return chunks

    def build_vectorstore(
        self,
        documents: list[Any],
        persist_dir: str = "faiss_store",
    ):

        chunks = self.chunk_documents(documents)

        vectorstore = FAISS.from_documents(
            chunks,
            self.embeddings
        )

        vectorstore.save_local(persist_dir)

        print(
            f"[INFO] Vector store saved successfully to '{persist_dir}'"
        )

        return vectorstore


if __name__ == "__main__":

    docs = load_all_documents("data")

    pipeline = EmbeddingPipeline()

    pipeline.build_vectorstore(
        docs,
        persist_dir="faiss_store"
    )