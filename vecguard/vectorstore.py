import uuid

import gqr
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

try:
    domain_test_data_embeddings = np.load("data/domain_test_data_embeddings.npy")
except Exception:
    raise Exception("Please run data/prepare.py to generate the necessary files.")

domain_test_data_df = gqr.load_id_test_dataset()
VECTOR_LEN = len(domain_test_data_embeddings[0])
COLLECTION_NAME = "chat_collection"


def _build_points(embeddings: np.ndarray, data: pd.DataFrame) -> list:
    points = []
    for vector, (_, row) in zip(embeddings, data.iterrows(), strict=False):
        payload = {"text": row["text"], "domain": row["domain"], "label": row["label"]}
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()), payload=payload, vector=vector.tolist()
            )
        )
    return points


def get_qdrant_client() -> QdrantClient:

    client = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_LEN, distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(
            m=0,
        ),
    )

    print(f"Collection '{COLLECTION_NAME}' created successfully!")

    points = _build_points(domain_test_data_embeddings, domain_test_data_df)

    print(f"Upserting {len(points)} points to collection '{COLLECTION_NAME}'...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Upserted {len(points)} points to collection '{COLLECTION_NAME}'")
    client.update_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_LEN, distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16,
        ),
    )
    print(f"Collection '{COLLECTION_NAME}' ready for usage!")
    return client
