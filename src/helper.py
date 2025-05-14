
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_pdf_text(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        temp_path = "./temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)
    return documents

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return splitter.split_documents(documents)

def create_embeddings(splits):
    return FAISS.from_documents(documents=splits, embedding=embeddings)

def create_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just reformulate it if needed."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an assistant for question-answering tasks. Use the following retrieved context "
            "to answer the question. If you don't know the answer, say you don't know. "
            "Use three sentences max.\n\n{context}"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)
