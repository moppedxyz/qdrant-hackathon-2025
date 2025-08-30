import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer

from vecguard import dataloader

SEED = 42
THRESHOLD = 0.20

train_data, train_eval_data = dataloader.load_train_data()
test_domain_data, test_ood_data = dataloader.load_test_data()

class LEACE:
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """
        Initializes the LEACE debiasing model.

        Args:
            embeddings (np.ndarray): The embeddings used to learn the debiasing direction.
            labels (np.ndarray): The labels corresponding to the embeddings.
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        self._mean_embedding = np.mean(embeddings, axis=0)
        centered_embeddings = embeddings - self._mean_embedding

        concept_direction = np.linalg.lstsq(centered_embeddings, labels, rcond=None)[0]
        projection_onto_concept = concept_direction @ np.linalg.pinv(concept_direction)
        self._projection_matrix = np.eye(embeddings.shape[1]) - projection_onto_concept
        self._domain_means: dict[str, np.ndarray] = {}

    def transform(self, embeddings_to_debias: np.ndarray) -> np.ndarray:
        """
        Debiases a set of embeddings by projecting them onto the debiasing subspace.

        Args:
            embeddings_to_debias (np.ndarray): The embeddings to be debiased.

        Returns:
            np.ndarray: The debiased embeddings.
        """
        centered_embeddings = embeddings_to_debias - self._mean_embedding
        debiased_embeddings = centered_embeddings @ self._projection_matrix
        return debiased_embeddings + self._mean_embedding

    def learn_domain_means(self, domains_to_embeddings: dict[str, np.ndarray]) -> None:
        """
        Calculates and stores the mean embeddings for various domains after debiasing.

        Args:
            domains_to_embeddings (dict[str, np.ndarray]): A dictionary mapping domain names
                                                           to their corresponding embeddings.
        """
        for domain, embeddings in domains_to_embeddings.items():
            if not isinstance(embeddings, np.ndarray) or embeddings.ndim < 2:
                embeddings = np.vstack(embeddings)
            debiased_embeddings = self.transform(embeddings)
            self._domain_means[domain] = np.mean(debiased_embeddings, axis=0)

    def predict(self, text: str) -> int:
        """
        Predicts the domain of a given text and returns a corresponding integer.

        Args:
            text (str): The input text to classify.
            model (SentenceTransformer): The model used to encode the text.
            threshold (float): The minimum cosine similarity to be considered in-domain.

        Returns:
            int: The predicted domain's integer code (0, 1, 2) or 3 for out-of-domain.
        """
        if not self._domain_means:
            raise ValueError(
                "Domain means have not been initialized. Call learn_domain_means() first."
            )

        # Define the mapping from domain names to integers
        domain_to_int = {"law": 0, "finance": 1, "healthcare": 2}
        ood_code = 3  # Code for out-of-domain

        embedding = self.model.encode(text)
        debiased_embedding = self.transform(embedding).squeeze()

        similarities = {
            domain: 1 - spatial.distance.cosine(debiased_embedding, mean_vec)
            for domain, mean_vec in self._domain_means.items()
        }

        max_similarity = max(similarities.values())

        if max_similarity < THRESHOLD:
            return ood_code

        predicted_domain = max(similarities, key=similarities.get)

        # Return the integer code for the predicted domain
        return domain_to_int.get(predicted_domain, ood_code)


embeddings_for_debiasing = np.vstack(train_data["vector"].values)
labels_for_debiasing = train_data["label"].values
domains_to_embeddings = {
    "finance": train_data[train_data["domain"] == "finance"]["vector"].values,
    "law": train_data[train_data["domain"] == "law"]["vector"].values,
    "healthcare": train_data[train_data["domain"] == "healthcare"]["vector"].values,
}

leace_transformer = LEACE(embeddings_for_debiasing, labels_for_debiasing)
leace_transformer.learn_domain_means(domains_to_embeddings)


def scoring_function(text: str) -> int:
    return leace_transformer.predict(text)
