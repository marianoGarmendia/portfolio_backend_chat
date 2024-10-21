from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from astrapy import DataAPIClient
from astrapy.info import CollectionVectorServiceOptions

import os
import json
from dotenv import load_dotenv

load_dotenv()
# client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Connect to a database by database endpoint
# Default keyspace, long form
# database = client.get_database("API_ENDPOINT")
# Default keyspace, short form
# database = client["API_ENDPOINT"]
# Explicit keyspace
# database = client.get_database(os.getenv("ASTRA_DB_API_ENDPOINT"), keyspace="chat_cv_space")


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY,
)

def doc_load(file_path:str):
    file_path_src = file_path #Ruta al archivo PDF
    loader = PyPDFLoader(file_path_src)

    docs = loader.load()

    return docs 

def text_splitter(docs:str):
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n" , "true"]) # type: ignore
    documents_splitters = text_splitter.split_documents(docs)
    return documents_splitters

def create_vector_store(documents , path:str):

    vector = Chroma.from_documents(documents = documents,embedding= embeddings,persist_directory= path)
    if vector is None:
        raise Exception("No se pudo crear el vector")
    else:
        return True
    
def get_vector_store_retriever(vector_path:str):
    vector_local = Chroma(persist_directory=vector_path, embedding_function=embeddings)
    retriever = vector_local.as_retriever()
    return retriever



# VERSIÓN ASTRADB
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")


    



def connect_to_astra_vstore(embeddings):
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    desired_namespace = os.getenv("ASTRA_DB_KEYSPACE")

#     openai_vectorize_options = CollectionVectorServiceOptions(
#     provider="openai",
#     model_name="text-embedding-3-small",
#     authentication={
#         "providerKey": OPENAI_API_KEY,
#     },
# )
    
    if desired_namespace:
        ASTRA_DB_KEYSPACE = desired_namespace
    else:
        ASTRA_DB_KEYSPACE = None
    
    vstore_astra = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="chat_cv_collection_test",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
    )

      
    return vstore_astra


def add_docs_astra_and_get_retriever(vstore_astra , documents):
    vstore_astra.delete_collection()
    vstore_astra.add_documents(documents=documents)
    retriever_astra = vstore_astra.as_retriever(search_kwargs={"k":5})
    return retriever_astra
    
    
