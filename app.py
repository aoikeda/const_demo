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

# ページ設定
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Streamlit UI
st.title("RAG Chatbot")
st.write("質問を入力してください。")

# ユーザーからの入力を取得
user_input = st.text_input("質問を入力してください:")

# チャットボットの設定
if user_input:
    with st.spinner('回答を生成中...'):
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
        st.markdown("### 回答")
        st.info(response['result'])
        
        # 参照された文書を表示
        st.markdown("### 参照文書")
        for i, doc in enumerate(response['source_documents'], 1):
            with st.expander(f"参照文書 {i}"):
                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

# サイドバーに使い方の説明を追加
with st.sidebar:
    st.markdown("### 使い方")
    st.write("1. 質問を入力欄に入力してください")
    st.write("2. 自動的に関連する文書を検索し、回答を生成します")
    st.write("3. 参照文書は折りたたみ形式で表示されます")