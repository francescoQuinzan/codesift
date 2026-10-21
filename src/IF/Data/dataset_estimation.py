from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np


class ModelTrainingDatasetEstimate:
    def __init__(
        self, seed=42, model_name="all-MiniLM-L6-v2", random_selection_size=1000
    ):
        self.model = SentenceTransformer(model_name)
        self.random_selection_size = random_selection_size
        self.seed = seed
        self.ds = self.load_dataset()

    def load_dataset(self):
        ds = load_dataset("wikimedia/wikipedia", "20231101.en")
        ds = ds["train"]
        ds = ds.shuffle(seed=self.seed).select(range(100))
        return ds

    def findTop(self, query, k=5):
        # 1) Read files

        # Example: list of file paths

        ids = self.ds["id"]
        texts = self.ds["text"]

        # 2) Embed in batches
        model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # small & fast; swap for larger if needed

        batch_size = 64
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_emb = model.encode(
                batch, show_progress_bar=False, convert_to_numpy=True
            )
            embs.append(batch_emb)
        embs = np.vstack(embs)  # shape (N, D)

        # 3) Normalise (unit vectors): makes cosine similarity a simple dot product
        embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        # 4) Embed query and compute cosine similarities (cosine distance = 1 - similarity)
        def query_top_k(query, k=5):
            q_emb = model.encode([query], convert_to_numpy=True)[0]
            q_emb = q_emb / np.linalg.norm(q_emb)
            sims = embs_norm @ q_emb  # dot product = cosine similarity
            topk_idx = np.argsort(-sims)[:k]
            results = [ids[i] for i in topk_idx]
            return results

        return query_top_k(query, k=k)

    def run(self, query=None, k=5):
        if query is None:
            query = "neural attention mechanisms"
        results = self.findTop(query, k=k)
        return list(self.ds.filter(lambda x: x["id"] in results)["text"])
