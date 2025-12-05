#import libraries 
import openai
import langchain
import pinecone
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Pinecone

from langchain_openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()

#Read the document
def read_documents(directory_path):
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    return documents
doc=read_documents("C:\\Users\\Atul Bhosale\\OneDrive\\Desktop\\LLM_project\\Documents")
# print(len(doc))

def chunk_data(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    return chunks
chunks = chunk_data(doc)
# print(len(chunks))


#Embeddings Technique of OpenAI --converts the text into vectors
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

#testing if my API key is working
vectors = embeddings.embed_query("How are you?")

print(len(vectors))