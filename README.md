# vecguard (name idea TODO)

## setup

Clone the repository
```bash
git clone git@github.com:moppedxyz/qdrant-hackathon-2025.git
```

Create a virtual environment
```bash
uv venv --python 3.12
```

Install dependencies (via [uv](https://docs.astral.sh/uv/getting-started/installation/))
```bash
uv sync --all-extras
```

### ensure you have access to all datasets

```bash 
huggingface-cli login
```

```bash
https://huggingface.co/datasets/DDSC/dkhate
```


### data preparation - embeddings

```bash
python data/prepare.py
```

### load data

```
from vecguard import dataloader

train_data, train_eval_data = dataloader.load_train_data()
domain_test_data, ood_test_data = dataloader.load_test_data()
```

## read qdrant ID collection
```bash
from vecguard import vectorstore
client = vectorstore.get_qdrant_client()
```


# Results

| Approach | ID Score | OOD Score | GQR Score | Resources |  Test by |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [semantic-router (threshold=0.2)](https://github.com/aurelio-labs/semantic-router) | 0.61 | 0.94 |  0.74 |[Link](https://github.com/aurelio-labs/semantic-router) | William |
