import numpy as np

class Retriever:
    def __init__(self, index, texts):
        self.index = index
        self.texts = texts

    def retrieve(self, query, embedder, top_k=5):
        q_emb = embedder.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        return [self.texts[i] for i in I[0]]
