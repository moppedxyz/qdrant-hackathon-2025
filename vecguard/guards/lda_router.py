import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from vecguard import dataloader

SEED = 42
THRESHOLD = 0.05

train_data, train_eval_data = dataloader.load_train_data()
test_domain_data, test_ood_data = dataloader.load_test_data()

class LDARouter:
    def __init__(
        self, data: pd.DataFrame, n_components : int = 1
    ) -> None:
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.encoder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
        self.lda.fit(np.vstack(data['vector'].values), data['label'].values)

    def __call__(self, text: str) -> int:
        text_embedding = self.encoder.encode([text], show_progress_bar=False)[0]
        text_embedding = np.array(text_embedding)

        prob = self.lda.predict_proba([text_embedding])
        max_prob = np.max(prob)
        predicted_class = np.argmax(prob)
        if max_prob < THRESHOLD:
            return 3
        else:
            return int(predicted_class)

rl = LDARouter(data=train_data, n_components=2)


def scoring_function(text: str) -> int:
    return rl(text)

