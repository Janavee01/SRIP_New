import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from IPython.display import display, Markdown

from google.colab import files
uploaded = files.upload()

pdf_path = next(iter(uploaded))

GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant = QdrantClient(":memory:")
collection_name = "rag_gemini"

qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vector_store = Qdrant.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    location=":memory:",
    collection_name=collection_name,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the context below to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""
)
