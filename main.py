import os
import streamlit as st
import pickle
import time
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

st.title("News Research Tool 📰✨")
st.sidebar.title("News Article URLs 🔗")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}") 
    if url:
        urls.append(url)

process_url_click = st.sidebar.button("Process URLs")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Devstral-Small-2505",
    temperature=0.7,
    max_new_tokens= 500,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

main_placeholder = st.empty()

if process_url_click:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.info("🔄 Loading data...")
    data = loader.load()

    # split data
    main_placeholder.info("✂️ Text Splitter...started...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ".",""],
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(data)

    # create embeddings and save it to FAISS index
    main_placeholder.info("🧠 Embedding vector started Building...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorindex_huggingface = FAISS.from_documents(docs, embeddings)

    # save vector index
    file_path="vector_index.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_huggingface, f)



query = main_placeholder.text_input("❓ Question : ")

if query:
    # load vector index
    if os.path.exists("vector_index.pkl"):
        with open("vector_index.pkl", "rb") as f:
            vectorStore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.subheader("💡 Answer")
            st.write(result["answer"])
            st.subheader("📚 Sources")
            st.write(result["sources"])
    else:
        st.error("❗ Vector index not found")

