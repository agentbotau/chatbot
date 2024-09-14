from langchain.text_splitter import RecursiveCharacterTextSplitter

# To interact with OpenAI's large language models (LLMs) in a conversational manner
from langchain.chat_models import ChatOpenAI
import streamlit as st
# To create prompt templates
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
# To combine the Retriever with the QA chain
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
# To generate embeddings using OpenAI's LLM


# To interact with OpenAI's large language models (LLMs) in a conversational manner


from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import ChromaDB
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
import chromadb

from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import UnstructuredRTFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# To tidy up print output
import pprint
import faiss
import pypandoc
# pypandoc.download_pandoc()

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import nltk
nltk.data.path.append('C:/Users/HP/AppData/Roaming/nltk_data')

import os

# Define the path
base_path = os.path.expanduser('~\\AppData\\Roaming\\nltk_data\\tokenizers\\punkt')
directory_path = os.path.join(base_path, 'PY3')
file_path = os.path.join(directory_path, 'PY3_tab')

# Create the directory and file
os.makedirs(directory_path, exist_ok=True)
if not os.path.isfile(file_path):
    open(file_path, 'a').close()

print(f"File created or already exists at {file_path}")

# Optional: Verify the existence of the file
if os.path.isfile(file_path):
    print("File exists.")
else:
    print("File does not exist.")







import os
os.environ["OPENAI_API_KEY"] =  st.secrets["openai_api_key"]


def load_rtf_files_in_batch(directory_path):
    documents = []
    # Traverse all directories and subdirectories
    for root, dirs, files in os.walk(directory_path):
        # Filter out the .rtf files and ignore macOS metadata files
        rtf_files = [f for f in files if f.endswith('.rtf') and not f.startswith('._')]
        for file in rtf_files:
            file_path = os.path.join(root, file)
            try:
                # Attempt to load the RTF file
                loader = UnstructuredRTFLoader(file_path, mode="elements", strategy="fast")
                docs = loader.load()
                documents.extend(docs)
            except RuntimeError as e:
                # If there's an encoding issue, skip the file or log the error
                print(f"Error processing file {file_path}: {e}")
                continue
    return documents



def create_and_save_vectorstore(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local("faiss_index")







def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever, llm):
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain



def invoke_rag_chain(user_input, rag_chain, temperature, max_t, top_p, fp, pp):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature, max_tokens=max_t, model_kwargs={"top_p": top_p, "frequency_penalty":fp, "presence_penalty":pp})
    return rag_chain.invoke(user_input)
