from semantic_router import Route
from semantic_router.encoders import LocalEncoder
from semantic_router.routers import SemanticRouter
from semantic_router.schema import RouteChoice

from vecguard import dataloader

SEED = 42
THRESHOLD = 0.15

encoder = LocalEncoder(name="all-MiniLM-L6-v2")
train_data, train_eval_data = dataloader.load_test_data()

finance = Route(
    name="finance",
    utterances=train_data[train_data['domain'] == 'finance'].sample(5, random_state=SEED)['text'].values.tolist()
)

healthcare = Route(
    name="healthcare",
    utterances=train_data[train_data['domain'] == 'healthcare'].sample(5, random_state=SEED)['text'].values.tolist()
)

law = Route(
    name="law",
    utterances=train_data[train_data['domain'] == 'law'].sample(5, random_state=SEED)['text'].values.tolist()
)

routes = [finance, healthcare, law]
rl = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

def scoring_function(text : str) -> int:
    response_dict= {
        "finance": 1,
        "healthcare": 2,
        "law": 0,
        "ood" : 3
    }
    prediction = rl(text)
    if not isinstance(prediction, RouteChoice):
        return 3
    if prediction is None:
        return 3
    similarity_score =  prediction.similarity_score if prediction.similarity_score else 0
    if similarity_score <= THRESHOLD:
        return 3
    return response_dict.get(prediction.name, 3)  # type: ignore
