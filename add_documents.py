import os, re, sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
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
    texts = []  # 辞書からリストに変更
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            with open(f"{folder_path}/{file}", "rb") as f:
                pdf = PdfReader(f)
                # ページごとに情報を保持
                for page_num, page in enumerate(pdf.pages, 1):  # 1からページ番号を開始
                    texts.append({
                        "filename": file,
                        "page": page_num,
                        "content": page.extract_text().replace("\n", "")
                    })
    return texts

if __name__ == "__main__":
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=128)
    documents = []
    texts = read_pdf("pdfs")
    
    # 各ページのテキストをチャンクに分割し、メタデータを付与
    for text_info in texts:
        chunks = text_splitter.split_text(text_info["content"].strip())
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "filename": text_info["filename"],
                    "page": text_info["page"]
                }
            ))

    vectorstore = initialize_vectorstore()
    vectorstore.add_documents(documents)