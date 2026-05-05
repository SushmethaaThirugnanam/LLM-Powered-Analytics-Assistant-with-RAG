import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_sentiment(reviews):
    context = "\n".join(reviews)
    prompt = f"""
    Analyze sentiment (positive/negative/mixed) and extract top 3 complaint themes.
    Reviews:
    {context}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":prompt}]
    )
    return response["choices"][0]["message"]["content"]
