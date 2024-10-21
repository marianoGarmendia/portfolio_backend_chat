from langchain.chains import ConversationalRetrievalChain, RetrievalQA
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage

from utils import doc_load , text_splitter , create_vector_store ,get_vector_store_retriever , connect_to_astra_vstore , add_docs_astra_and_get_retriever , embeddings

import os
from dotenv import load_dotenv
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

   
llm_openai = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=1500,
    
)

chat_history = []



def get_history_aware_retriever(retriever):
    contextualize_q_system_prompt = """Obten el chat history y la ultima pregunta del usuario \n
    que podría hacer referencia al contexto en el historial de chat, formular una pregunta independiente \
    que se puede entender sin el historial de chat. si no sabes la respuesta  di que no lo sabes, \
    simplemente reformúlelo si es necesario y de lo contrario devuélvalo como está."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm_openai, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def create_question_answer_chain():
    qa_system_prompt = """Tu eres Mariano Garmendia que responde preguntas de un usuario que puede ser un reclutador de una empresa. \n
    Estás buscando una oportunidad laboral, responde amablemente y en tono cálido. \n
    Responde con tono argentino, hacé que la charla sea amena \n
    Usa las siguientes piezas del contexto recuperado para responder la pregunta. \n
    Si tu no sabes la respuesta, di que no lo sabes. \n
    Usa pocas oraciones y muestra predisposición para trabajar.\n

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm_openai, qa_prompt)
    return question_answer_chain

def create_rag_chain(history_aware_retriever, question_answer_chain):
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def memory_retrieval_conversation(rag_chain, query:str, chat_history):
    response = rag_chain.invoke({"input": query, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=query), response["answer"]])
    return { response["answer"]}


def get_rag_chain(path_vector:str , local):
    print("Cargando documentos...")
    if local:
        retriever = get_vector_store_retriever(path_vector)
        history_aware_retriever = get_history_aware_retriever(retriever)
        question_answer_chain = create_question_answer_chain()
        rag_chain = create_rag_chain(history_aware_retriever, question_answer_chain)
    else:
        docs = doc_load("./Mariano_G_CV_oct_24.pdf")
        docs_splitters = text_splitter(docs)
        vector = create_vector_store(docs_splitters , path_vector)
        retriever = get_vector_store_retriever(path_vector)
        history_aware_retriever = get_history_aware_retriever(retriever)
        question_answer_chain = create_question_answer_chain()
        rag_chain = create_rag_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

# VERSIÓN ASTRADB

def get_rag_chain_astradb(astraDB , embeddings):
    if astraDB is False:
        docs = doc_load("./Mariano_G_CV_oct_24.pdf")
        docs_splitters = text_splitter(docs)
        vstore_astra = connect_to_astra_vstore(embeddings=embeddings)
        retriever_astra = add_docs_astra_and_get_retriever(vstore_astra, docs_splitters)
        history_aware_retriever = get_history_aware_retriever(retriever_astra)
        question_answer_chain = create_question_answer_chain()
        rag_chain_astra = create_rag_chain(history_aware_retriever, question_answer_chain)
        return rag_chain_astra
    else:
        vstore_astra = connect_to_astra_vstore(embeddings=embeddings)
        retriever_astra = vstore_astra.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.5},
        )
        # retriever_astra = add_docs_astra_and_get_retriever(vstore_astra, docs_splitters)
        history_aware_retriever = get_history_aware_retriever(retriever_astra)
        question_answer_chain = create_question_answer_chain()
        rag_chain_astra = create_rag_chain(history_aware_retriever, question_answer_chain)
        return rag_chain_astra
    
      



