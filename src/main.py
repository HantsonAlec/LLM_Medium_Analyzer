import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain import hub
from langchain_community.chat_models import ChatOllama

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data"

EMBEDDING_MODEL = OllamaEmbeddings(model="llama3")
LLM_MODEL = ChatOllama(model="llama3")
VECTOR_STORE = PineconeVectorStore(
    embedding=EMBEDDING_MODEL, index_name=os.getenv("INDEX_NAME")
)
RETRIEVAL_QA_PROMPT = hub.pull("langchain-ai/retrieval-qa-chat")

CUSTOM_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know, don't try to make up the answer.
Use three sentences maximum and try to keep the answer as concise as possible.
Always say "Thanks for asking" at the end of the answer.

{context}

Question: {question}

Helpful answer:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving..")
    query = "How does netflix make its operations such as error handling and production rollout efficient?"
    combine_docs_chain = create_stuff_documents_chain(LLM_MODEL, RETRIEVAL_QA_PROMPT)
    retrieval_chain = create_retrieval_chain(
        retriever=VECTOR_STORE.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result = retrieval_chain.invoke(input={"input": query})
    print(result)

    custom_rag_prompt = PromptTemplate.from_template(CUSTOM_TEMPLATE)
    rag_chain = (
        {"context": VECTOR_STORE.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | LLM_MODEL
    )

    result = rag_chain.invoke(query)
    print(result)
