import os, re, sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
# from langchain.vectorstores import Pinecone
# import pinecone
from pinecone import Pinecone 
from pinecone import Pinecone,ServerlessSpec
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


# 環境変数をロード
load_dotenv()

def initialize_vectorstore():

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if "construction-laws" not in pc.list_indexes().names():
        pc.create_index(
            name="construction-laws",
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    embeddings = OpenAIEmbeddings()
    return PineconeVectorStore.from_existing_index("construction-laws", embeddings)

def read_pdf(folder_path):
    text = {}
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            with open(f"{folder_path}/{file}", "rb") as f:
                pdf = PdfReader(f)
                text[file] = ""
                for page in pdf.pages:
                    text[file] += page.extract_text().replace("\n", "")
    return text

if __name__ == "__main__":
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=64)
    documents = []
    text = read_pdf("pdfs")
    for filename, content in text.items():
      chunks = text_splitter.split_text(content.strip())
      for chunk in chunks:
          documents.append(Document(page_content=chunk,metadata={"filename":filename}))

    vectorstore = initialize_vectorstore()
    vectorstore.add_documents(documents)