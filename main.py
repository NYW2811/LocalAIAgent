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
    docs = []
    if hasattr(retriever, "_get_relevant_documents"):
        docs = retriever._get_relevant_documents(question, run_manager=None)
    elif hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(question)
    elif hasattr(retriever, "retrieve"):
        docs = retriever.retrieve(question)
    elif hasattr(retriever, "invoke"):
        docs = retriever.invoke(question)
    else:
        raise RuntimeError("El retriever no soporta métodos de búsqueda conocidos.")

    if isinstance(docs, str):
        print(f"[WARNING] El retriever devolvió una cadena en lugar de documentos para: {question}")
        docs = []

    if not docs:
        print(f"[WARNING] No se encontraron documentos para la pregunta: {question}")

    contexts = contexts = [doc.page_content for doc in docs]
    contexts_text = "\n\n".join(contexts)

    print(f"[DEBUG] Documentos encontrados: {len(docs)} para la pregunta: {question}")
    if contexts_text:
        print(f"[DEBUG] Contexto enviado al modelo (primeros 300 caracteres): {contexts_text[:300].replace('\n', ' ')}")
    else:
        print("[DEBUG] No hay contexto para enviar al modelo.")

    result = chain.invoke({
        "shows_rankings": contexts_text,
        "question" : question
    })
    
    if hasattr(result, "content"):
        answer = result.content
    else:
        answer = str(result)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }