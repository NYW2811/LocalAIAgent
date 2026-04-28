"""
En este programa se llevan a cabo las evaluaciones del sistema RAG.
Para la primera prueba, se establecieron 15 preguntas para verificar (no tuve oportunidad de ejecutar pq no tengo ni GPU ni internet)
So this code may explode. Be careful. 
Al ejecutarse, el programa: 
- Definirá las listas con preguntas y verdades estructuradas medinate un proceso mixto (manual, LLM Local) sobre la base de datos metric.tv_shows.csv
- Mediante un ciclo for, llamará a la función "ejecutar_rag" del main.py.
- Recuperará las repsuestas y contextos del RAG, questions and ground_truths y formará un dataset from Dataset.
- Con el Dataset formado, se pasará a evaluación con ragas, usando faithfulness y asnwer_relevancy. 
- Solo Dios sabe que pasará cuando se ejecute.
"""
from datasets import Dataset
from main import ejecutar_rag
import time
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_ollama import ChatOllama, OllamaEmbeddings
from utils import similarity


questions = [
    "¿De qué trata The Last of Us?",
    "¿De qué genero es The last of Us?",
    "¿Cuántas temporadas tiene la serie de Cosmos: A space-time Odyssey?",
    "¿En qué está basado el documental de Cosmos?",
    "¿En qué está basada Arcane?",
    "¿De qué géneros es la serie Arcane?",
    "¿Cuál es la puntiación de The Wire?",
    "¿Quién dirigió Bo Burnham: Inside?",
    "¿Quiénes son los directores de The Vietnam War?",
    "¿Cuál es la puntiación de The Last Dance?",
    "¿Cúal es el puntaje de Bleak House?",
    "¿Quién dirigío The Wire Season 2?",
    "¿A qué género pertence The Americans?",
    "Dame el usersocre del documental que habla sobre Michael Jordan",
    "¿Cuál es el nombre de la serie ambientada en personajes de League of Legends?"
]

ground_truths = [
    "Set 20 years after the destruction of civilization, Joel (Pedro Pascal) is hired to smuggle 14-year-old Ellie (Bella Ramsey) out of a quarantine zone in this drama series based on the PlayStation video game of the same name.",
    "Action,Adventure,Drama,Horror,Sci-Fi,Thriller",
    "1 temporada.",
    "Based on the Carl Sagan's original Cosmos series, astrophysicist Neil deGrasse Tyson hosts this new version.",
    "The animated series based on the League of Legends follows the origin of two League champions in the prosperous city of Piltover and the underground district of Zaun.",
    "Animation,Action,Adventure,Drama,Fantasy,Sci-Fi",
    "92",
    "Bo Burnham",
    "Ken Burns, Lynn Novick",
    "88",
    "78",
    "No disponible en la base de datos.",
    "No disponible en la base de datos.",
    "The Last Dance, 88.",
    "Arcane"
]

dataset = []

print("Iniciando evaluación RAG...")
print(f"Total de preguntas a evaluar: {len(questions)}")
print("--------------------------------------------")

global_start = time.perf_counter()

for index, (q, g_t) in enumerate(zip(questions, ground_truths), start=1):
    print(f"Pregunta {index}: {q}")

    start_time = time.perf_counter()
    result = ejecutar_rag(q)
    end_time = time.perf_counter()

    elapsed = end_time - start_time

    print(f"  ✔ Respuesta recibida en {elapsed:.2f} segundos")
    print(f"  Respuesta: {result.get('answer')}")
    print(f"  Contextos: {len(result.get('contexts', []))} elemento(s)")
    print("--------------------------------------------")

    dataset.append({
        "question": q,
        "answer": result["answer"],
        "contexts": list(result["contexts"]),
        "ground_truth": g_t
    })

global_end = time.perf_counter()

print(f"Tiempo total: {global_end - global_start:.2f} segundos")

# Dataset para RAGAS
dataset_evaluation = Dataset.from_list(dataset)

# =========================
# MÉTRICAS BÁSICAS
# =========================

exact_matches = sum(
    1
    for item in dataset
    if item["answer"].strip().lower() == item["ground_truth"].strip().lower()
)

avg_similarity = sum(
    similarity(item["answer"], item["ground_truth"])
    for item in dataset
) / len(dataset)

print("\nMétricas básicas:")
print(f"Exact matches: {exact_matches} de {len(dataset)}")
print(f"Similitud promedio: {avg_similarity:.2f}")

# =========================
# RAGAS
# =========================

print("\nIniciando evaluación con RAGAS...")

# LLM y embeddings 
llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

resultados = evaluate(
    dataset_evaluation,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=llm,
    embeddings=embeddings
)

print("\nResultados RAGAS:")
print(resultados)

#Detalle por pregunta 
df = resultados.to_pandas()

print("\nDetalle por pregunta:")
print(df)