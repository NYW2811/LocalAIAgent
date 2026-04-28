import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import build_vectordb

# Configuración de embeddings y text splitter

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

#Estos son los df xd
df_tv = pd.read_csv("data/metacritic_tv_shows.csv")
df_games = pd.read_csv("data/metacritic_games.csv")
df_imdb = pd.read_csv("data/imdb_top_1000.csv")

persit_dir = "./chroma_langchain_db"


#Estos ya son las bases vectoriales 
tv_db = build_vectordb(df_tv, "tv_shows", persit_dir, embeddings, text_splitter)
games_db = build_vectordb(df_games, "Games", persit_dir, embeddings, text_splitter)
imdb_db = build_vectordb(df_imdb, "Imdb", persit_dir, embeddings, text_splitter)

#Estos son los retrievers
tv_retriever = tv_db.as_retriever(search_kwargs={"k": 5})
games_retriever = games_db.as_retriever(search_kwargs={"k": 5})
imdb_retriever = imdb_db.as_retriever(search_kwargs={"k": 5})