import gqr
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

train_data, train_eval_data = gqr.load_train_dataset()
print("Train data loaded...")
domain_test_data = gqr.load_id_test_dataset()
print("Domain test data loaded...")
ood_test_data = gqr.load_ood_test_dataset()
print("OOD test data loaded...")

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
batch_size = 1024
print(f"Encoder is using {encoder.device}")


def batch_encode_data(
    data: np.ndarray, encoder: SentenceTransformer, batch_size: int
) -> np.ndarray:
    embeddings = np.empty(
        (0, encoder.get_sentence_embedding_dimension()), dtype=np.float32
    )
    for i in tqdm(range(0, len(data), batch_size), desc="Encoding batches"):
        batch = data[i : i + batch_size]
        batch_embeddings = encoder.encode(batch, show_progress_bar=False)
        embeddings = np.vstack((embeddings, batch_embeddings))
    return embeddings


train_data_embeddings = batch_encode_data(
    train_data["text"].values, encoder, batch_size
)
np.save("data/train_data_embeddings.npy", train_data_embeddings)
train_data["vector"] = train_data_embeddings.tolist()
train_data.to_parquet("data/train_data.parquet", index=False)

print("Train data embeddings saved.")
train_eval_data_embeddings = batch_encode_data(
    train_eval_data["text"].values, encoder, batch_size
)
np.save("data/train_eval_data_embeddings.npy", train_eval_data_embeddings)
train_eval_data["vector"] = train_eval_data_embeddings.tolist()
train_eval_data.to_parquet("data/train_eval_data.parquet", index=False)

print("Train eval data embeddings saved.")

domain_test_data_embeddings = batch_encode_data(
    domain_test_data["text"].values, encoder, batch_size
)
np.save("data/domain_test_data_embeddings.npy", domain_test_data_embeddings)
domain_test_data["vector"] = domain_test_data_embeddings.tolist()
domain_test_data.to_parquet("data/domain_test_data.parquet", index=False)
print("Domain test data embeddings saved.")

ood_test_data_embeddings = batch_encode_data(
    ood_test_data["text"].values, encoder, batch_size
)
np.save("data/ood_test_data_embeddings.npy", ood_test_data_embeddings)
ood_test_data["vector"] = ood_test_data_embeddings.tolist()
ood_test_data.to_parquet("data/ood_test_data.parquet", index=False)
print("OOD test data embeddings saved.")
