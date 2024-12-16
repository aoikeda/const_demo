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
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# .envファイルを読み込む
load_dotenv()

# 環境変数を取得
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Streamlit用のカスタムコールバックハンドラー
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Pineconeの初期化
pc = Pinecone(api_key=pinecone_api_key)

index_name = "construction_laws"

vector_store = initialize_vectorstore()
retriever = vector_store.as_retriever()

# ページ設定
st.set_page_config(page_title="耐火基準エキスパート", layout="wide")

# Streamlit UI
st.title("RAG Chatbot")
st.write("質問を入力してください。")

# ユーザーからの入力を取得
user_input = st.text_input("質問を入力してください:")

# チャットボットの設定
if user_input:
    # 回答用のプレースホルダーを作成
    answer_placeholder = st.empty()
    
    # ストリーミングハンドラーの設定
    stream_handler = StreamHandler(answer_placeholder)
    
    # QAチェーンの作成
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            streaming=True,
            callbacks=[stream_handler]
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # 質問を実行
    response = qa_chain.invoke({"query": user_input})
    
    # 参照された文書を表示
    st.markdown("### 参照文書")
    for i, doc in enumerate(response['source_documents'], 1):
        with st.expander(f"参照文書 {i}"):
            st.markdown(f"""
            ```text
            {doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content}
            ```
            """)

# サイドバーに使い方の説明を追加
with st.sidebar:
    st.markdown("### 使い方")
    st.markdown("""
    #### 基本的な使い方
    1. 入力欄に知りたいことを質問として入力してください
    2. Enterキーを押すと、AIが回答を生成します
    3. AIの回答が文章として表示されます
    4. 回答の下に、参考にした法令や規定が表示されます
    
    #### 質問のコツ
    - 具体的に聞きたいことを明確に書くと、より正確な回答が得られます
    - 複数の質問は分けて聞くと、より分かりやすい回答になります
    
    #### 例えば...
    - 「避難階段の必要な条件を教えてください」
    - 「最上階から数えた階数が12の階の床は炎にどれだけ耐えればいいですか？」
    
    """)