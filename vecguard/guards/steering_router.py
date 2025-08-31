import numpy as np
from geom_median.numpy import compute_geometric_median
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from vecguard import dataloader

SEED = 42
THRESHOLD = 0.15

class MultiLabelSteeringClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.steering_vectors: dict[str, np.ndarray] = {}

    def fit(self, data: list[tuple[str, list[str]]]) -> None:

        all_texts = [item[0] for item in data]
        all_labels = set()
        if len(data[0]) != 3:
            raise ValueError("Each data item must be a tuple of (text, vector, labels)")
        for _, _, labels in data:
            for label in labels:
                all_labels.add(label)

        print("Encoding all texts... (this may take a moment)")
        # Pre-compute all embeddings to avoid re-computing
        all_embeddings = [item[1] for item in data]
        text_to_embedding = dict(zip(all_texts, all_embeddings, strict=False))

        print("Creating steering vectors for each label...")
        for label in sorted(all_labels):
            positive_embeddings = []
            negative_embeddings = []

            # Separate embeddings into positive (has the label) and negative (does not have the label)
            for text, _, labels in data:
                if label in labels:
                    positive_embeddings.append(text_to_embedding[text])
                else:
                    negative_embeddings.append(text_to_embedding[text])

            median_pos = compute_geometric_median(np.array(positive_embeddings)).median
            median_neg = compute_geometric_median(np.array(negative_embeddings)).median
            self.steering_vectors[label] = median_pos - median_neg

    def predict(self, text: str, threshold: float = THRESHOLD) -> int:
        text_embedding = self.model.encode(text)
        predicted = {}

        for label, vector in self.steering_vectors.items():
            similarity = cosine_similarity([text_embedding], [vector])
            predicted[label] = similarity

        # find max but then max is below threshold return 3
        if predicted:
            max_label = max(predicted, key=predicted.get)
            return max_label if predicted[max_label] > threshold else 3
        return 3


train_data, train_eval_data = dataloader.load_train_data()
train_dataset = []
for text, vector, label in zip(
    train_data["text"].values,
    train_data["vector"].values,
    train_data["label"].values,
    strict=False,
):
    train_dataset.append((text, vector, [int(label)]))


# 2. Initialize and fit the classifier
classifier = MultiLabelSteeringClassifier()
classifier.fit(train_dataset)


def scoring_function(text: str) -> int:
    return classifier.predict(text)
