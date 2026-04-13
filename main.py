from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

try:
    model = OllamaLLM(model="llama3.2")
except Exception as e:
    print(f"no sé, error: {e}")
    exit(1)

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

    try:
        shows_rankings = retriever.invoke(question)
    except Exception as e:
        print(f"no sé, error: {e}")
        continue

    try:
        result = chain.invoke({"shows_rankings":shows_rankings, "question": question})
        print(result)
    except:
        print(f"Error con el modelo {e}")
        continue

