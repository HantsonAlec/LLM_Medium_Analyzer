import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data"

TEXT_SPLITTER = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
EMBEDDING_MODEL = OllamaEmbeddings(model="llama3")

if __name__ == "__main__":
    loader = TextLoader(DATA_PATH / "mediumblog1.txt")
    document = loader.load()

    print("splitting")
    texts = TEXT_SPLITTER.split_documents(document)
    print(f"Created {len(texts)} chunks")

    print("ingesting")
    PineconeVectorStore.from_documents(
        documents=texts, embedding=EMBEDDING_MODEL, index_name=os.getenv("INDEX_NAME")
    )
    print("Done")
