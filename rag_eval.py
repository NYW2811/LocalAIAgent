"""
Este programa ejecuta las evaluaciones de faithfulness y answer_relevancy usando la librería ragas. 
Necesita:
- Ejecutar las preguntas (question) en el sistema RAG
- Colocar las salidas del RAG en orden, tomando solo el context de la función modificada en el archivo main.py
"""

import ragas
from datasets import Dataset
from ragas.metrics.collections import faithfulness, answer_relevancy
from ragas import evaluate

evaluation_data_faith ={
    "question": [
        "¿De qué trata The Last of Us?",
        "¿De qué genero es The last of Us?",
        "¿Cuántas temporadas tiene la serie de Cosmos: A space-time Odyssey?",
        "¿En qué está basado el documental de Cosmos?",
        "¿En qué está basada Arcane?",
        "¿De qué géneros es la serie Arcane?"


    ],
    "context": [ #Faltaría agregar las respuestas que arroje el RAG 

    ],
    "answer": [ #Faltaría agregar las respuestas que arroge el RAG

    ],

    "ground_truth":[
        "Set 20 years after the destruction of civilization, Joel (Pedro Pascal) is hired to smuggle 14-year-old Ellie (Bella Ramsey) out of a quarantine zone in this drama series based on the PlayStation video game of the same name.",
        "Action,Adventure,Drama,Horror,Sci-Fi,Thriller",
        "1 temporada.",
        "Based on the Carl Sagan's original Cosmos series, astrophysicist Neil deGrasse Tyson hosts this new version.",
        "The animated series based on the League of Legends follows the origin of two League champions in the prosperous city of Piltover and the underground district of Zaun.",
        "Animation,Action,Adventure,Drama,Fantasy,Sci-Fi"
    ]
}

dataset_faith = Dataset.from_dict(evaluation_data_faith)
print(dataset_faith)

results = evaluate(dataset=dataset_faith, metrics=[faithfulness])