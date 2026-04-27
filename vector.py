from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

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
"""
embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")
"""

model_name = os.getenv("EMBEDDING_MODEL", "all-minilm:l6-v2")
provider = os.getenv("EMBEDDING_PROVIDER", "ollama") # 'ollama' o 'openai'

# 2. Elegimos la clase según el proveedor
if provider.lower() == "openai":
    embeddings = OpenAIEmbeddings(model=model_name)
else:
    embeddings = OllamaEmbeddings(model=model_name)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

def build_vectordb(df, collection_name, persist_dir):
    db_path = f"{persist_dir}/{collection_name}"
    add_documents = not os.path.exists(db_path)

    documents = []
    ids = []

    if add_documents:
        for i, row in df.iterrows():
            full_text = "\n".join([
                f"{col}: {str(row[col])}" 
                for col in df.columns if pd.notna(row[col])
            ])

            chunks = text_splitter.split_text(full_text)

            for j, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata=row.to_dict(),
                        id=f"{collection_name}_{i}_{j}"
                    )
                )
                ids.append(f"{collection_name}_{i}_{j}")

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    if add_documents:
        vector_store.add_documents(documents=documents, ids=ids)

    return vector_store


#Estos son los df xd
df_tv = pd.read_csv("data\metacritic_tv_shows.csv")
df_games = pd.read_csv("data\metacritic_games.csv")
df_imdb = pd.read_csv("data\imdb_top_1000.csv")

persit_dir = "./chroma_langchain_db"


#Estos ya son las bases vectoriales 
tv_db = build_vectordb(df_tv, "tv_shows", persist_dir=persit_dir)
games_db = build_vectordb(df_games, "Games", persist_dir=persit_dir)
imdb_db = build_vectordb(df_imdb, "Imdb", persist_dir=persit_dir)

#Estos son los retrievers
movies_retriever = tv_db.as_retriever(search_kwargs={"k": 5})
tv_retriever = games_db.as_retriever(search_kwargs={"k": 5})
games_retriever = imdb_db.as_retriever(search_kwargs={"k": 5})