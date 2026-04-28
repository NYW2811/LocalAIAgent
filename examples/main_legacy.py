# CÓDIGO LEGADO - Función ejecutar_rag antes de refactor
# Este archivo muestra la implementación original que ahora está en utils.py

# La función fue diseñada para manejar diferentes tipos de retrievers
# y ejecutar consultas RAG de manera robusta, capturando contextos y respuestas

def ejecutar_rag(question, retriever, chain):
    """
    Función central para ejecutar el pipeline RAG.
    
    Recibe:
    - question: pregunta del usuario
    - retriever: objeto retriever de LangChain (puede tener múltiples APIs)
    - chain: cadena de prompt + modelo LLM
    
    Retorna:
    - Dict con:
      - question: la pregunta original
      - answer: respuesta generada por el modelo
      - contexts: documentos recuperados relevantes
    
    Complejidad: Maneja múltiples tipos de retrievers (algunos tienen diferentes métodos)
    """
    
    # Intenta diferentes métodos de llamada según qué retriever se use
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
        raise RuntimeError("El retriever no soporta metodos de busqueda conocidos.")


    # Validaciones: a veces el retriever devuelve string en lugar de objetos
    if isinstance(docs, str):
        print(f"[WARNING] El retriever devolvió una cadena en lugar de documentos para: {question}")
        docs = []
    
    if not docs:
        print(f"[WARNING] No se encontraron documentos para la pregunta: {question}")
        contexts = []
    else:
        # Extrae contenido de cada documento
        contexts = [doc.page_content for doc in docs]

    # Une todos los contextos en una cadena para pasar al modelo
    contexts_text = "\n\n".join(contexts)

    # Debug info
    print(f"[DEBUG] Documentos encontrados: {len(docs)} para la pregunta: {question}")
    if contexts_text:
        print(f"[DEBUG] Contexto enviado al modelo (primeros 300 caracteres): {contexts_text[:300].replace('\n', ' ')}")
    else:
        print("[DEBUG] No hay contexto para enviar al modelo.")

    # Invoca la cadena LLM
    result = chain.invoke({
        "shows_rankings": contexts_text,
        "question": question
    })
    
    # Maneja dos formatos de salida del modelo
    if hasattr(result, "content"):
        answer = result.content
    else:
        answer = str(result)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }


# NOTA: Esta función ahora está en utils.py/ejecutar_rag()
# y mantiene toda la lógica robusta de manejo de diferentes retrievers
