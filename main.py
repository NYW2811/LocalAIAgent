from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from vector import imdb_retriever, tv_retriever, games_retriever
from utils import ejecutar_rag as ejecutar_rag_utils
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

system_template = """"
Ejemplo xd
"""
#Mover los templates a un archivo a parte para que el main quede más limpio     
user_template = """
You are "Jhon Tv Show" an expert in answering questions about entretainment focusing on TV and movies.
Your task is to answer only using the provide context. If the answer is not in the given context, say: "I'm sorry. I have no information to answet that."

Here are some relevant shows rankings: {shows_rankings}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", user_template),
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