"""
Modificaciones: 
-Eliminaicón del ciclo while para interacción con usuario.
-Implememtación de nueva función ejecutar_rag (revisar abajo)
"""


from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are "Jhon Tv Show" an expert in answering questions about entretainment focusing on TV and movies.
Your task is to answer only using the provide context. If the answer is not in the given context, say: "I'm sorry. I have no information to answet that."

Here are some relevant shows rankings: {shows_rankings}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

"""
while True:
    print("\n\n---------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    shows_rankings = retriever.invoke(question)
    result = chain.invoke({"shows_rankings":shows_rankings, "question": question})
    print(result)
"""


#Función final que es útil para ragas.
#NOTA: RECIBE DESCRIPCIÓN, TÍTULO Y METADATA. 
#Quité el while xd

def ejecutar_rag(question):
    docs = retriever.invoke(question)

    contexts = [
    f"""
     Title: {doc.metadata['title']}
     Metascore: {doc.metadata['metascore']}
     Userscore: {doc.metadata['userscore']}
     Date: {doc.metadata['date']}

    Description:
    {doc.page_content}
    """
    for doc in docs
]
    contexts_text = "\n\n".join(contexts)

    result = chain.invoke({
        "shows_rankings": contexts_text,
        "question" : question
    })
    
    answer = result.content

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }