import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import pdfplumber
from pinecone import Pinecone, ServerlessSpec
from add_documents import initialize_vectorstore

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
vector_store = initialize_vectorstore()
retriever = vector_store.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-4o",
            temperature=0
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


