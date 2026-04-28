from difflib import SequenceMatcher

from langchain_chroma import Chroma
from langchain_core.documents import Document


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _get_retriever_documents(retriever, question):
    """Obtiene documentos desde diferentes APIs de retriever compatibles."""
    if hasattr(retriever, "_get_relevant_documents"):
        return retriever._get_relevant_documents(question, run_manager=None)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(question)
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(question)
    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)
    raise RuntimeError("El retriever no soporta metodos de busqueda conocidos.")


def ejecutar_rag(question, retriever, chain):
    docs = _get_retriever_documents(retriever, question)

    if isinstance(docs, str):
        docs = []
    if not docs:
        contexts = []
    else:
        contexts = [doc.page_content for doc in docs]

    contexts_text = "\n\n".join(contexts)
    result = chain.invoke({
        "shows_rankings": contexts_text,
        "question": question,
    })

    answer = result.content if hasattr(result, "content") else str(result)
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }


def build_vectordb(df, collection_name, persist_dir, embeddings, text_splitter):
    add_documents = len(
        Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        ).get().get("ids", [])
    ) == 0

    documents = []
    ids = []

    if add_documents:
        for i, row in df.iterrows():
            full_text = "\n".join(
                [f"{col}: {str(row[col])}" for col in df.columns if row[col] == row[col]]
            )

            chunks = text_splitter.split_text(full_text)

            for j, chunk in enumerate(chunks):
                doc_id = f"{collection_name}_{i}_{j}"
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata=row.to_dict(),
                        id=doc_id,
                    )
                )
                ids.append(doc_id)

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    if add_documents:
        for i in range(0, len(documents), 500):
            vector_store.add_documents(
                documents=documents[i : i + 500],
                ids=ids[i : i + 500],
            )

    return vector_store


#Función final que es útil para ragas.
#NOTA: RECIBE DESCRIPCIÓN, TÍTULO Y METADATA. 


def ejecutar_rag_to_RAGAS(question):
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