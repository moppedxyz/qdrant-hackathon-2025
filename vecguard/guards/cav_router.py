import numpy as np
from sentence_transformers import SentenceTransformer
from typing import NamedTuple

from vecguard import dataloader

SEED = 42
THRESHOLD = 0.15  # similarity threshold below which we mark as OOD


class RouterPrediction(NamedTuple):
    route: str
    similarity_score: float


class CAVRouter:
    """Concept Activation Vector (CAV) based router.

    For each domain we compute a concept direction:
        cav_domain = mean(embeddings_domain) - mean(embeddings_rest)
    New text is encoded and routed to the domain whose CAV has the highest
    cosine similarity with the text embedding, provided it's above THRESHOLD.
    Otherwise we return 'ood'.
    """

    def __init__(self) -> None:
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.train_data, _ = dataloader.load_train_data()
        self._prepare_vectors()
        self._compute_cavs()

    def _prepare_vectors(self) -> None:
        # If vectors already exist (like in PCA router) reuse them, else encode.
        if "vector" in self.train_data.columns:
            self.train_data["_vec"] = self.train_data["vector"].apply(lambda v: np.array(v, dtype=np.float32))
        else:
            # Encode in batches for efficiency
            texts = self.train_data["text"].tolist()
            embeddings = self.encoder.encode(texts, show_progress_bar=False, batch_size=64)
            self.train_data["_vec"] = list(embeddings)

    def _compute_cavs(self) -> None:
        self.cavs: dict[str, np.ndarray] = {}
        domains = self.train_data["domain"].unique().tolist()
        # Pre-cache domain mean embeddings
        domain_means: dict[str, np.ndarray] = {}
        for d in domains:
            domain_vecs = np.stack(self.train_data[self.train_data["domain"] == d]["_vec"].to_list())
            domain_means[d] = domain_vecs.mean(axis=0)
        for d in domains:
            rest = [domain_means[o] for o in domains if o != d]
            if len(rest) == 0:
                continue
            rest_mean = np.mean(rest, axis=0)
            cav = domain_means[d] - rest_mean
            # Normalize CAV
            norm = np.linalg.norm(cav) + 1e-9
            cav = cav / norm
            self.cavs[d] = cav.astype(np.float32)

    def __call__(self, text: str) -> RouterPrediction:
        emb = self.encoder.encode([text], show_progress_bar=False)[0].astype(np.float32)
        emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
        best_domain = None
        best_score = -1.0
        for domain, cav in self.cavs.items():
            score = float(np.dot(emb_norm, cav))  # cosine since both normalized
            if score > best_score:
                best_score = score
                best_domain = domain
        if best_domain is None:
            return RouterPrediction(route="ood", similarity_score=float(best_score))
        if best_score <= THRESHOLD:
            return RouterPrediction(route="ood", similarity_score=float(best_score))
        return RouterPrediction(route=best_domain, similarity_score=float(best_score))


rl = CAVRouter()


def scoring_function(text: str) -> int:
    response_dict = {"finance": 1, "healthcare": 2, "law": 0, "ood": 3}
    prediction = rl(text)
    if not isinstance(prediction, RouterPrediction):
        return 3
    similarity_score = prediction.similarity_score if prediction.similarity_score else 0
    if similarity_score <= THRESHOLD:
        return 3
    return response_dict.get(prediction.route, 3)
