import streamlit as st
import sqlite3
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# -----------------------------
# Paths
# -----------------------------
DB_PATH = r"D:\VSCode\LLM NL-SQL\data\olist.db"
FAISS_INDEX_PATH = "data/faiss_index.bin"
EMBEDDINGS_META_PATH = "data/embeddings_meta.csv"

# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    meta = pd.read_csv(EMBEDDINGS_META_PATH)
    return index, meta

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_translator():
    # Use a robust multilingual model
    return pipeline("text2text-generation", model="facebook/m2m100_418M")

def load_m2m100():
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    return tokenizer, model

# -----------------------------
# Helper functions
# -----------------------------
def run_sql(query: str):
    conn = get_connection()
    return pd.read_sql(query, conn)

def search_reviews(user_query: str, k: int = 5):
    index, meta = load_faiss_index()
    model = load_model()
    emb = model.encode([user_query], convert_to_numpy=True)
    D, I = index.search(emb, k)
    return meta.iloc[I[0]]

def safe_translate(text: str) -> str:
    try:
        tokenizer, model = load_m2m100()
        tokenizer.src_lang = "pt"  # source language Portuguese
        encoded = tokenizer(text, return_tensors="pt")

        # Force output in English
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("en"))
        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated
    except Exception:
        return text  # fallback to original if translation fails

def extract_themes(chunks):
    text = " ".join(chunks).lower()
    themes = []
    if "entrega" in text or "prazo" in text or "correio" in text or "delivery" in text:
        themes.append("Delivery delays / incomplete orders")
    if "produto" in text or "defeito" in text or "costurada" in text or "indisponível" in text or "product" in text:
        themes.append("Product quality issues")
    if "email" in text or "resposta" in text or "atendimento" in text or "cliente" in text or "customer" in text:
        themes.append("Customer service problems")
    return themes[:3]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("LLM-Powered Analytics Assistant")

mode = st.radio("Choose query type:", ["SQL", "RAG", "Hybrid"])

if mode == "SQL":
    sql_query = st.text_area("Enter SQL query:")
    if st.button("Run SQL"):
        try:
            df = run_sql(sql_query)
            st.dataframe(df)
        except Exception as e:
            st.error(f"SQL error: {e}")

elif mode == "RAG":
    user_query = st.text_input("Enter your question about reviews:")
    if st.button("Search Reviews"):
        try:
            results = search_reviews(user_query)
            st.write("Top matching review chunks:")
            sentiment_model = load_sentiment_model()
            chunks = []
            for _, row in results.iterrows():
                chunk_text = row['chunk']
                chunks.append(chunk_text)
                sentiment = sentiment_model(chunk_text[:200])[0]['label']

                translated = safe_translate(chunk_text)

                st.write(f"- Order {row['order_id']} ({sentiment}):")
                st.write(f"   Original: {chunk_text[:200]}...")
                st.write(f"   English: {translated}")

            themes = extract_themes(chunks)
            st.subheader("Top Complaint Themes")
            for theme in themes:
                st.write(f"- {theme}")

        except Exception as e:
            st.error(f"RAG error: {e}")

elif mode == "Hybrid":
    st.write("Hybrid query combines SQL + RAG results.")
    sql_query = st.text_area("Enter SQL query:")
    user_query = st.text_input("Enter your question about reviews:")
    if st.button("Run Hybrid"):
        try:
            # SQL
            df = run_sql(sql_query)
            st.subheader("SQL Results")
            st.dataframe(df)

            # RAG
            results = search_reviews(user_query)
            st.subheader("Review Insights")
            sentiment_model = load_sentiment_model()
            chunks = []
            sentiments = []
            for _, row in results.iterrows():
                chunk_text = row['chunk']
                chunks.append(chunk_text)
                sentiment = sentiment_model(chunk_text[:200])[0]['label']
                sentiments.append(sentiment)

                translated = safe_translate(chunk_text)

                st.write(f"- Order {row['order_id']} ({sentiment}):")
                st.write(f"   Original: {chunk_text[:200]}...")
                st.write(f"   English: {translated}")

            themes = extract_themes(chunks)
            st.subheader("Top Complaint Themes")
            for theme in themes:
                st.write(f"- {theme}")

            # Synthesis
            st.subheader("Hybrid Answer")
            summary = f"""
            Based on your SQL query, here are the structured results above.
            Review analysis shows sentiment leaning {max(set(sentiments), key=sentiments.count)} overall.
            The top complaint themes are: {', '.join(themes)}.
            Together, this suggests that while the data shows {len(df)} rows returned,
            customer feedback highlights issues in these areas.
            """
            st.write(summary)

        except Exception as e:
            st.error(f"Hybrid error: {e}")
