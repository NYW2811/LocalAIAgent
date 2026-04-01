import chromadb
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model locally (no API needed)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from local SentenceTransformer model."""
    embedding = embedding_model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


# ChromaDB setup
db_location = "./chroma_db"
collection_name = "tv_shows_ranking"

# Initialize persistent Chroma client
chroma_client = chromadb.PersistentClient(path=db_location)

# Create collection with embedding function
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

# Load and process data
df = pd.read_csv("metacritic_tv_shows.csv")

# Text splitter (simple chunking)
def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


# Check if we need to add documents
existing_count = collection.count()
if existing_count == 0:
    documents = []
    metadatas = []
    ids = []

    for i, row in df.iterrows():
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        description = str(row["description"]) if pd.notna(row["description"]) else ""

        full_text = f"{title}.\n{description}"
        chunks = split_text(full_text)

        for j, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "title": title,
                "metascore": str(row["metascore"]) if pd.notna(row["metascore"]) else "",
                "userscore": str(row["userscore"]) if pd.notna(row["userscore"]) else "",
                "date": str(row["releaseDate"]) if pd.notna(row["releaseDate"]) else ""
            })
            ids.append(f"doc_{i}_{j}")

    # Batch add for efficiency (max 5461 per batch in ChromaDB)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # Generate embeddings in batch
        embeddings = [get_embedding(doc) for doc in batch_docs]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
            embeddings=embeddings
        )
    
    print(f"Added {len(documents)} documents to vector store")


def retriever(query: str, k: int = 5) -> list[dict]:
    """Search the vector store and return relevant results."""
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    formatted_results = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            formatted_results.append({
                "content": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None
            })
    
    return formatted_results
