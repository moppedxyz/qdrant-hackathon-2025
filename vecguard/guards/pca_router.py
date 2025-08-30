from typing import NamedTuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from vecguard import dataloader

SEED = 42
THRESHOLD = 0.15

train_data, train_eval_data = dataloader.load_train_data()
test_domain_data, test_ood_data = dataloader.load_test_data()


class Route(NamedTuple):
    name: str
    utterances: list[str]


class RouterPrediction(NamedTuple):
    route: str
    similarity_score: float


class PCARouter:
    def __init__(
        self, routes: list[Route] | None = None, data: pd.DataFrame | None = None
    ) -> None:
        if routes is None and data is None:
            raise ValueError("Either 'routes' or 'data' must be provided.")
        self.routes = routes
        self.data = data
        self.route_embeddings: dict[str, np.ndarray] = {}
        self.pca = None
        self.transformed_embeddings: dict[str, np.ndarray] = {}
        self.encoder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # Initialize encoder for text input
        self.init_router()

    def __call__(self, text: str) -> RouterPrediction:
        """Route the input text to the most appropriate route."""

        # Encode the input text
        text_embedding = self.encoder.encode([text], show_progress_bar=False)[0]
        text_embedding = np.array(text_embedding)

        if self.pca is not None:
            text_embedding_transformed = self.pca.transform(
                text_embedding.reshape(1, -1)
            )[0]
        else:
            text_embedding_transformed = text_embedding

        best_route = None
        best_similarity = -1

        for route_name, route_embeddings in (
            self.transformed_embeddings if self.pca else self.route_embeddings
        ).items():
            similarities = cosine_similarity(
                text_embedding_transformed.reshape(1, -1), route_embeddings
            )[0]
            avg_similarity = np.mean(similarities)

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_route = route_name

        return RouterPrediction(
            route=best_route or "ood", similarity_score=float(best_similarity)
        )

    def init_router(self) -> None:
        """Initialize the router by computing embeddings for all route utterances or using dataframe vectors"""

        if self.data is not None:
            # Use dataframe data with pre-computed vectors
            print("Using pre-computed vectors from dataframe...")

            # Group by domain to create route embeddings
            for domain in self.data["domain"].unique():
                domain_data = self.data[self.data["domain"] == domain]

                # Extract vectors and convert to numpy array
                vectors = []
                for vector in domain_data["vector"]:
                    vectors.append(np.array(vector))

                self.route_embeddings[domain] = np.array(vectors)
                print(f"Loaded {len(vectors)} vectors for route '{domain}'")

        elif self.routes is not None:
            # Use routes with utterances (original behavior)
            print("Computing embeddings for route utterances...")

            for route in self.routes:
                utterance_embeddings = []
                for utterance in route.utterances:
                    embedding = self.encoder.encode(
                        [utterance], show_progress_bar=False
                    )[0]
                    utterance_embeddings.append(np.array(embedding))

                self.route_embeddings[route.name] = np.array(utterance_embeddings)
                print(
                    f"Computed {len(utterance_embeddings)} embeddings for route '{route.name}'"
                )

        # Fit PCA on all embeddings
        all_embeddings = []
        for embeddings in self.route_embeddings.values():
            all_embeddings.extend(embeddings)

        if len(all_embeddings) > 0:
            all_embeddings = np.array(all_embeddings)
            print(f"Fitting PCA on {len(all_embeddings)} total embeddings...")

            # Use PCA to reduce dimensionality (TODO: adjust n_components as needed)
            n_components = min(50, all_embeddings.shape[1], len(all_embeddings))
            self.pca = PCA(n_components=n_components)
            self.pca.fit(all_embeddings)

            # Transform route embeddings using PCA
            for route_name, embeddings in self.route_embeddings.items():
                self.transformed_embeddings[route_name] = self.pca.transform(embeddings)

            print(
                f"PCA fitted with {n_components} components, explained variance ratio: {self.pca.explained_variance_ratio_[:5]}"
            )


rl = PCARouter(data=train_data)


def scoring_function(text: str) -> int:
    response_dict = {"finance": 1, "healthcare": 2, "law": 0, "ood": 3}
    prediction = rl(text)
    if not isinstance(prediction, RouterPrediction):
        return 3
    if prediction is None:
        return 3
    similarity_score = prediction.similarity_score if prediction.similarity_score else 0
    if similarity_score <= THRESHOLD:
        return 3
    return response_dict.get(prediction.route, 3)
