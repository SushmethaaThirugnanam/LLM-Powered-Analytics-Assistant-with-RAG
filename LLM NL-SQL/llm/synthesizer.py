import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def synthesize(sql_result, rag_result):
    prompt = f"""
    Combine the following SQL and RAG results into a single coherent answer:
    SQL Result: {sql_result}
    RAG Result: {rag_result}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":prompt}]
    )
    return response["choices"][0]["message"]["content"]
