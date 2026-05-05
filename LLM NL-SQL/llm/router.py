import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def classify_query(query):
    prompt = f"""
    Classify the query into one of: SQL, RAG, HYBRID.
    Query: "{query}"
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()
