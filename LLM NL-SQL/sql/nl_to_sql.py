import sqlite3
import pandas as pd
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_sql(query, schema_prompt, db_path="data/olist.db"):
    prompt = f"""
    You are an expert SQL assistant. Schema: {schema_prompt}
    Convert the following natural language query into valid SQLite SQL:
    {query}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":prompt}]
    )
    sql_query = response["choices"][0]["message"]["content"]
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(sql_query, conn)
    conn.close()
    return df, sql_query
