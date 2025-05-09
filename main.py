import streamlit as st
from PyPDF2 import PdfReader  # Fixed import (capital P and R)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Fixed spelling
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# genai.configure(api_key=os.getenv("Google_API_KEY"))

google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("Google_API_KEY")  ## this woll read the secrets from .tomal file and locally
genai.configure(api_key=google_api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Fixed class name
        for page in pdf_reader.pages:  # Fixed property name (.pages not .page)
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Fixed class name and parameter case (chunk_size not chunk_Size)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Reduced from 10000 for better performance
        chunk_overlap=200  # Reduced from 1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Fixed FAISS usage for newer versions
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():  # Fixed spelling of function name
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Helpful Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Fixed FAISS loading for newer versions
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()  # Fixed function name
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("E-JUST (Research method & Data Analysis) Chat with MY RAG APP (Eng / Omar Adel)")
    
    user_question = st.text_input("Ask a question about your PDFs:")
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDFs",  # Fixed typo
            type=["pdf"],
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):  # Fixed button text
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")

if __name__ == "__main__":
    main()