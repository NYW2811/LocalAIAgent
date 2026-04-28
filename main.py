from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from vector import imdb_retriever, tv_retriever, games_retriever
from utils import ejecutar_rag as ejecutar_rag_utils
from prompts import system_template, user_turn_template
import os

try:
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "phi3:latest"),  #<-- Aquí se configura el nombre del modelo
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("BASE_URL", "http://localhost:11434/v1")
    )
except Exception as e:
    print(f"Error al cargar el modelo. Cambie la ruta del modelo e intente nuevamente.: {e}")
    exit(1)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", user_turn_template),
])
chain = prompt | model


def ejecutar_rag(question):
    return ejecutar_rag_utils(question, tv_retriever, chain)


while True:
    print("\n\n---------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    result = ejecutar_rag(question)
    print(result["answer"])