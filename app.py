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

# .envファイルを読み込む
load_dotenv()

# 環境変数を取得
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Pineconeの初期化
pc = Pinecone(api_key=pinecone_api_key)

index_name = "construction_laws"

vector_store = initialize_vectorstore()
retriever = vector_store.as_retriever()
# Streamlit UI
st.title("RAG Chatbot")
st.write("質問を入力してください。")

# ユーザーからの入力を取得
user_input = st.text_input("質問を入力してください:")


# チャットボットの設定
if user_input:
    # QAチェーンの作成
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-4o",
            temperature=0
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # 質問を実行
    response = qa_chain.invoke({"query": user_input})
    
    # 回答を表示
    st.write("回答:", response['result'])
    
    # 参照された文書を表示
    st.write("\n参照文書:")
    for doc in response['source_documents']:
        st.write("---")
        st.write(doc.page_content)