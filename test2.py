import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<your-api-key>"
### LLM

local_llm = "mistral:v0.1"
### Index

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings


docs = [
    Document(page_content="This is a demo document"),
    Document(page_content="This is another demo document"),
]

vectorstore = Chroma.from_documents(
    documents=docs,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model="nomic-embed-text:latest"),
)
retriever = vectorstore.as_retriever()
print(retriever)