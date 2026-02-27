import streamlit as st
import tempfile

# NEW imports (IMPORTANT)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Resume RAG Chatbot")
st.title("📄 Resume RAG Chatbot")

st.write("Upload bulk resume PDFs and ask about specific candidates.")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

query = st.text_input("Ask about a candidate:")

# ---------------------------
# Embeddings Model
# ---------------------------

# Option 1: OpenAI Embeddings
# embeddings = OpenAIEmbeddings(openai_api_key="YOUR_API_KEY")

# Option 2: Free HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# Process PDFs
# ---------------------------

def process_pdfs(files):
    documents = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Add metadata (Candidate Name from file name)
        for doc in docs:
            doc.metadata["candidate_name"] = file.name

        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(documents)

    return split_docs

# ---------------------------
# Main Logic
# ---------------------------

if uploaded_files:

    with st.spinner("Processing resumes..."):
        docs = process_pdfs(uploaded_files)

        vectorstore = FAISS.from_documents(docs, embeddings)

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        llm = ChatOpenAI(
            temperature=0,
            openai_api_key="YOUR_API_KEY",
            model_name="gpt-3.5-turbo"
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )

    st.success("Resumes processed successfully!")

    if query:
        with st.spinner("Searching candidate..."):
            response = qa_chain(query)

            st.subheader("Answer:")
            st.write(response["result"])

            st.subheader("Matched Candidate Sources:")
            for doc in response["source_documents"]:
                st.write("📌", doc.metadata["candidate_name"])