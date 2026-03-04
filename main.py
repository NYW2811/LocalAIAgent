from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

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
    print(result)

