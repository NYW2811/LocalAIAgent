from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter



df = pd.read_csv("metacritic_tv_shows.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_lanchain_db"
add_documents = not os.path.exists(db_location)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # puedes ajustar
    chunk_overlap=100    # pequeño solapamiento
)


documents = []
ids = []

if add_documents:
    for i, row in df.iterrows():
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        description = str(row["description"]) if pd.notna(row["description"]) else ""

        full_text = f"{title}.\n{description}"

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
