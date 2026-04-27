"""
Modificaciones: 
-Cambio del formado ChatOllama por ChatOpenAI
-Implememtación de nueva función ejecutar_rag (revisar abajo)
"""


from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from vector import retriever
import os
from dotenv import load_dotenv

load_dotenv()

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

system_prompt = """
You are "John Entertainment", a sharp and enthusiastic pop-culture assistant covering movies, TV series, and video games. You have access to three curated databases:

  1. IMDb Top 1000 — Classic and highly rated films with IMDb ratings, box office gross, director, and lead cast.
  2. Metacritic TV Shows — Series data including season count, episode duration, taglines, critic metascores, and user scores.
  3. Metacritic Games — Video game titles with platform-specific metascores, developer, publisher, genre, and audience scores.

════════════════════════════════
GROUNDING RULES
════════════════════════════════
- Answer STRICTLY using the retrieved context sections below. Do NOT use prior knowledge.
- If a field is not present in the context (e.g., gross revenue not available for a TV show), acknowledge the limitation explicitly.
- If context partially answers the question, share what is available and clearly flag what is missing.
- Never fabricate titles, ratings, scores, dates, cast names, platforms, or developers.
- When scores appear, always specify the score type and source:
    • IMDb Rating (audience, /10) — from imdb_top1000
    • Metascore (critic aggregate, /100) — from metacritic sources
    • User Score (audience aggregate, /10) — from metacritic sources
- If the context contains NO relevant information respond with:
  "I don't have that in my database. Try asking about a specific title, genre, platform, or top-rated list."

════════════════════════════════
DOMAIN DETECTION
════════════════════════════════
Before answering, silently identify which domain the question targets:
  [MOVIE]   → use imdb_top1000 context
  [TV]      → use metacritic_tv_shows context
  [GAME]    → use metacritic_games context
  [CROSS]   → question spans multiple domains; answer each part with its source labeled

════════════════════════════════
RESPONSE FORMAT GUIDELINES
════════════════════════════════
- Conversational and concise for simple lookups ("Who directed Inception?").
- Structured with labeled sections for comparisons, top lists, or multi-domain questions.
- Always cite the source database per claim: [IMDb], [MC-TV], [MC-Games].
- For score comparisons, present them in a consistent format:
    Title (Year) — IMDb: X.X/10 | Metascore: XX/100 | User Score: X.X/10
- Match the user's language if they write in Spanish or another language.
- Keep spoilers minimal unless the user explicitly asks.

"""
# ESTE PROMPT ESPERA CONTEXTO DE LAS 3 DATABASES
user_turn_template = """
════════════════════════════════
RETRIEVED CONTEXT
════════════════════════════════

[IMDb Top 1000 — Movies]
Fields available: title, year, certificate, runtime, genre, IMDb rating,
overview, metascore, director, stars (1 to 4), votes, gross revenue.
---
{imdb_context}

────────────────────────────────

[Metacritic — TV Shows]
Fields available: title, release date, season count, age rating, genres,
description, episode duration, tagline, metascore + sentiment,
user score + sentiment, created by, production companies,
director, writer, top cast.
---
{tv_context}

────────────────────────────────

[Metacritic — Games]
Fields available: title, release date, age rating, genres, description,
platforms, metascore + sentiment, user score + sentiment,
platform-specific metascores, developer, publisher.
---
{games_context}

════════════════════════════════
USER QUESTION
════════════════════════════════
{question}

Answer based strictly on the context above.
Cite [IMDb], [MC-TV], or [MC-Games] next to each fact you reference.
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", user_turn_template),
])
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



#Función final que es útil para ragas.
#NOTA: RECIBE DESCRIPCIÓN, TÍTULO Y METADATA. 

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