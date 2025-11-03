import streamlit as st
import os
from openai import OpenAI
from os import environ

#Extra Import
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

#Server and Website
client = OpenAI(
	api_key="os.environ["API_KEY"]",
	base_url="https://api.ai.it.cornell.edu",
)

print(f"DEBUG: API Key value is: {os.environ.get('API_KEY', 'Key Not Found')}")

#UI Setup
st.set_page_config(page_title="File Q&A with OpenAI", page_icon="📝")
st.title("📝 File Q&A with AI!")
uploaded_files = st.file_uploader("Upload documents", type=("txt", "md", "pdf"), accept_multiple_files=True)

question = st.chat_input(
    "Please ask something about the Uploaded File.",
    disabled=not uploaded_files,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#Load and Preprocess
if uploaded_files:
    os.makedirs("./data", exist_ok=True)
    documents = []

    for uploaded_file in uploaded_files:
        temp_path = f"./data/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            match uploaded_file.name.lower().split(".")[-1]:
                case "pdf":
                    loader = PyPDFLoader(temp_path)
                case "txt":
                    loader = TextLoader(temp_path)
                case "md":
                    loader = TextLoader(temp_path)
                case _:
                    st.warning(f"⚠️ Unsupported file format: {uploaded_file.name}")
                    continue; 

        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = uploaded_file.name
        documents.extend(loaded_docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    st.success(f"Successfully processed {len(uploaded_files)} document(s).")

#Promt Template
template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}

    Context:{context}

    Answer:
"""
prompt = PromptTemplate.from_template(template)

#Original Q&A Logic
# if question and uploaded_files:
#     # Read the content of the uploaded file
#     file_content = uploaded_file.read().decode("utf-8")
#     print(file_content)

#     # Append the user's question to the messages
#     st.session_state.messages.append({"role": "user", "content": question})
#     st.chat_message("user").write(question)

#     with st.chat_message("assistant"):
#         stream = client.chat.completions.create(
#             model="gpt-4o",  # Change this to a valid model name
#             messages=[
#                 {"role": "system", "content": f"Here's the content of the file:\n\n{file_content}"},
#                 *st.session_state.messages
#             ],
#             stream=True
#         )
#         response = st.write_stream(stream)

#     # Append the assistant's response to the messages
#     st.session_state.messages.append({"role": "assistant", "content": response})

# Main Q&A Logic
if question and uploaded_files:

    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    retrieved_docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(
        [f"From {doc.metadata.get('source', 'unknown')}: {doc.page_content}" for doc in retrieved_docs]
    )

    formatted_prompt = prompt.format(question=question, context=context)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": formatted_prompt },
            ],
            stream=True
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
