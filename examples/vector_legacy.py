# CÓDIGO LEGADO - Construcción manual de vector stores antes de refactor
# Este archivo muestra el enfoque anterior antes de migrar a la función build_vectordb() en utils.py

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_ollama import OllamaEmbeddings


# ENFOQUE ANTERIOR 1: Usando Ollama embeddings (comentado, reemplazado por HuggingFace)
# Cargaba datos de un CSV de TV shows y construía vectores usando Ollama localmente
# Problema: requería Ollama corriendo, menos portable que HuggingFace
"""
df_1 = pd.read_csv("data/metacritic_tv_shows.csv")
embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")  #<-- Aquí se modifica el modelo de embeddings

db_location = "./chrome_lanchain_db"
add_documents = not os.path.exists(db_location)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # puedes ajustar
    chunk_overlap=100    # pequeño solapamiento
)

documents = []
ids = []

if add_documents:
    for i, row in df_1.iterrows():
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        description = str(row["description"]) if pd.notna(row["description"]) else ""
        userscore = str(row["userscore"]) if pd.notna(row["userscore"]) else ""

        full_text = f"{title}.\n{description}.\n{userscore}"

        chunks = text_splitter.split_text(full_text)

        for j, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": title,
                        "metascore": row["metascore"],
                        "userscore": row["userscore"],
                        "date": row["releaseDate"]
                    },
                    id=f"{i}_{j}"
                )
            )
            ids.append(f"{i}_{j}")

vector_store = Chroma(
    collection_name="tv_shows_ranking",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k":5}
)
"""


# ENFOQUE ANTERIOR 2: Cambio a sentence-transformers (HuggingFace)
# Este es el que se sigue usando actualmente, solo que ahora abstraído en la función build_vectordb()
# Ventaja: sin dependencias externas, modelos precargados, más ligero

# Construcción manual para un solo dataset (antes de generalizar):
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# Carga de datos
df_tv = pd.read_csv("data/metacritic_tv_shows.csv")
df_games = pd.read_csv("data/metacritic_games.csv")
df_imdb = pd.read_csv("data/imdb_top_1000.csv")

persist_dir = "./chroma_langchain_db"

# AHORA en vector.py se usan llamadas a build_vectordb() de utils.py
# Que encapsula toda esta lógica repetitiva en una función reutilizable:
"""
tv_db = build_vectordb(df_tv, "tv_shows", persist_dir, embeddings, text_splitter)
games_db = build_vectordb(df_games, "Games", persist_dir, embeddings, text_splitter)
imdb_db = build_vectordb(df_imdb, "Imdb", persist_dir, embeddings, text_splitter)

tv_retriever = tv_db.as_retriever(search_kwargs={"k": 5})
games_retriever = games_db.as_retriever(search_kwargs={"k": 5})
imdb_retriever = imdb_db.as_retriever(search_kwargs={"k": 5})
"""
