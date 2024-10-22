from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Request
import sys
import os

# from pydantic import BaseModel
# from langchain_core.prompts.base import BasePromptTemplate

# class MyModel(BaseModel):
#     prompt_template: BasePromptTemplate  # Usando el tipo que genera el error

#     class Config:
#         arbitrary_types_allowed = True  # Permitir tipos arbitrarios

# Ejemplo de uso
# my_instance = MyModel(prompt_template=BasePromptTemplate(...))  # I


from utils import connect_to_astra_vstore, embeddings , doc_load , text_splitter
from chat import get_rag_chain , memory_retrieval_conversation , chat_history , get_rag_chain_astradb

app = FastAPI()

# Define un modelo para la entrada
class Pregunta(BaseModel):
    pregunta: str

# Añade la ruta del directorio raíz al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Lista de orígenes permitidos
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "https://www.marianodev.site/#/chat-cv",
    "https://www.marianodev.site"
    # Agrega aquí otros dominios que necesiten acceder al backend
]

# Configuración del middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permitir orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los headers
)

# INICIA UN RAG CHAIN
# rag_chain = get_rag_chain("./vecotr_cv_enlaces" , local=True)
rag_chain = get_rag_chain_astradb(astraDB=False, embeddings=embeddings)
# //Esta linea que pasa?
# vstore_astra = connect_to_astra_vstore(embeddings=embeddings)
# docs = doc_load("./Mariano_Garmendia_cv_enlaces.pdf")
# docs_splitters = text_splitter(docs)
# vstore_astra.add_documents(documents=docs_splitters)
# results = vstore_astra.similarity_search(
#     "Dime si tienes experiencia en trabajo en equipo",
#     k=2,
   
# )

# retriever = vstore_astra.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.5},
# )
# print(results)

# consulta = input("Haceme tu consulta:")
# result = retriever.invoke(consulta)
# print(result)

# -------------  INICIA EL SERVIDOR -------------
@app.post("/chatbot")
async def chatbot( request: Request):

    body = await request.json()  # Obtén el cuerpo de la solicitud

    results = memory_retrieval_conversation(rag_chain=rag_chain, query=body['pregunta'], chat_history=chat_history)
    return results

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Si no encuentra el puerto, usará 5000 por defecto
    app.run( host="0.0.0.0", port=port)  # Inicia la aplicación en el puerto especificado
