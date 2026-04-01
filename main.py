from openai import OpenAI
from vector import retriever
from dotenv import load_dotenv
import os

# Load credentials from .env file
load_dotenv('.env')
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Leave model empty - LM Studio uses whatever is loaded
MODEL_NAME = "qwen/qwen3-4b-2507"  # or use "local-model"

template = """
You are "Jhon Tv Show" an expert in answering questions about entretainment focusing on TV and movies

Here are some relevant shows rankings: {shows_rankings}

Here is the question to answer: {question}
"""

while True:
    print("\n\n---------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    shows_rankings = retriever(question)
    print(f"Retrieved {len(shows_rankings)} documents")
    prompt = template.format(shows_rankings=shows_rankings, question=question)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        if response.choices and response.choices[0]:
            message = response.choices[0].message
            # Qwen models use reasoning_content for the actual response
            output = message.content or getattr(message, 'reasoning_content', '')
            print(output)
        else:
            print("Error: Empty response from API")
            print(f"Response: {response}")
    except Exception as e:
        print(f"API Error: {e}")



