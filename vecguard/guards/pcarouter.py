from vecguard.pca_router.models import Route, RouterPrediction
from vecguard.pca_router.router import PCARouter
from semantic_router.encoders import LocalEncoder

from vecguard import dataloader

SEED = 42
THRESHOLD = 0.15

encoder = LocalEncoder(name="all-MiniLM-L6-v2")
train_data, train_eval_data = dataloader.load_train_data()
test_domain_data, test_ood_data = dataloader.load_test_data()

# finance = Route(
#     name="finance",
#     utterances=None
#     utterances=train_data[train_data["domain"] == "finance"]
#     .sample(40, random_state=SEED)["text"]
#     .values.tolist(),
# )

# healthcare = Route(
#     name="healthcare",
#     utterances=train_data[train_data["domain"] == "healthcare"]
#     .sample(40, random_state=SEED)["text"]
#     .values.tolist(),
# )

# law = Route(
#     name="law",
#     utterances=train_data[train_data["domain"] == "law"]
#     .sample(40, random_state=SEED)["text"]
#     .values.tolist(),
# )

# routes = [finance, healthcare, law]
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
