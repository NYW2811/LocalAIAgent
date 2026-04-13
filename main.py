"""
Este programa es una reestructuración del main anterior. Para que funcione en cada máquina, el modelo debe ser cambiado por
el usuario. 
"""


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from vector import retriever

model = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen3:4b")   #Este es el modelo que deben cambiar para que funcione.

template = """
You are "Jhon Tv Show" an expert in answering questions about entretainment focusing on TV and movies

Here are some relevant shows rankings: {shows_rankings}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n---------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    shows_rankings = retriever.invoke(question)
    result = chain.invoke({"shows_rankings":shows_rankings, "question": question})
    print(result.content)

