import chromadb
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model locally
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

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
        # Extract all relevant fields
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        release_date = str(row["releaseDate"]) if pd.notna(row["releaseDate"]) else ""
        season_count = str(row["seasonCount"]) if pd.notna(row["seasonCount"]) else ""
        rating = str(row["rating"]) if pd.notna(row["rating"]) else ""
        genres = str(row["genres"]) if pd.notna(row["genres"]) else ""
        description = str(row["description"]) if pd.notna(row["description"]) else ""
        duration = str(row["duration"]) if pd.notna(row["duration"]) else ""
        tagline = str(row["tagline"]) if pd.notna(row["tagline"]) else ""
        metascore = str(row["metascore"]) if pd.notna(row["metascore"]) else ""
        metascore_count = str(row["metascore_count"]) if pd.notna(row["metascore_count"]) else ""
        metascore_sentiment = str(row["metascore_sentiment"]) if pd.notna(row["metascore_sentiment"]) else ""
        userscore = str(row["userscore"]) if pd.notna(row["userscore"]) else ""
        userscore_count = str(row["userscore_count"]) if pd.notna(row["userscore_count"]) else ""
        userscore_sentiment = str(row["userscore_sentiment"]) if pd.notna(row["userscore_sentiment"]) else ""
        created_by = str(row["created_by"]) if pd.notna(row["created_by"]) else ""
        production_companies = str(row["production_companies"]) if pd.notna(row["production_companies"]) else ""
        director = str(row["director"]) if pd.notna(row["director"]) else ""
        writer = str(row["writer"]) if pd.notna(row["writer"]) else ""
        top_cast = str(row["top_cast"]) if pd.notna(row["top_cast"]) else ""

        # Build comprehensive text for embeddings
        full_text = f"""Title: {title}
Tagline: {tagline}
Description: {description}
Genres: {genres}
Rating: {rating}
Duration: {duration} min
Seasons: {season_count}
Release Date: {release_date}
Metascore: {metascore} ({metascore_sentiment}, {metascore_count} reviews)
User Score: {userscore} ({userscore_sentiment}, {userscore_count} reviews)
Created By: {created_by}
Director: {director}
Writer: {writer}
Top Cast: {top_cast}
Production Companies: {production_companies}"""

        chunks = split_text(full_text)

        for j, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "title": title,
                "releaseDate": release_date,
                "seasonCount": season_count,
                "rating": rating,
                "genres": genres,
                "duration": duration,
                "tagline": tagline,
                "metascore": metascore,
                "metascore_count": metascore_count,
                "metascore_sentiment": metascore_sentiment,
                "userscore": userscore,
                "userscore_count": userscore_count,
                "userscore_sentiment": userscore_sentiment,
                "created_by": created_by,
                "production_companies": production_companies,
                "director": director,
                "writer": writer,
                "top_cast": top_cast
            })
            ids.append(f"doc_{i}_{j}")

    # Batch add for efficiency (max 5461 per batch in ChromaDB)
    batch_size = 1000
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
