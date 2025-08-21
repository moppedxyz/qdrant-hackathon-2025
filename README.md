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

# Ideas

## Least-Squares Concept Erasure (LEACE)

Method for removing a specific concept from a set of embeddings. It finds the linear direction or subspace that represents the concept and then projects the embeddings onto a space orthogonal to it, effectively "erasing" the concept's linear representation. For a set of vectors X and corresponding labels Y (representing the concept to erase), LEACE finds a linear transformation P such that PX is uncorrelated with Y, while minimizing the change to the original vectors. It's a closed-form solution that can be calculated directly.

## Steering Vectors

This approach involves directly manipulating the internal states (activations) of a model (encoder) during inference to influence its output. Adding a "steering vector" to the model's activations at specific layers to guide the output towards a desired attribute. To remove a component, you would subtract this vector. A steering vector is created by finding the average difference between embeddings of texts with and without a desired attribute. For example, steering_vector = avg_embedding("positive text") - avg_embedding("neutral text"). Very similar to semantic router.

- Create pairs of texts (e.g., one with a positive sentiment, one neutral).
- Get the internal activations/embeddings from a specific layer of your model for each text.
- Calculate the difference vector for each pair and average them to get the final steering vector.
- During generation, add or subtract this vector from the model's activations to steer or suppress the attribute.

## PCA / SVD Component Removal

- Fit PCA on a set of embeddings.
- Identify dominant components (e.g., domain-specific variance).
- Subtract projections onto those components.
- Example: Used in SIF embeddings to remove common directions.

## Linear Discriminant Analysis (LDA) / Class Projections

- If you have labeled classes (in-domain categories), compute directions that best separate them.
- Project embeddings into class subspaces or compute residuals to get out-of-class parts.

## Concept Activation Vectors (CAVs)

- Train a linear classifier for a concept (e.g., "finance domain").
- Use the weight vector as a direction in embedding space.
- Project any embedding onto it to measure/remove that concept.

## Independent Component Analysis (ICA)

While PCA finds orthogonal directions of maximum variance, ICA goes a step further by finding directions that are statistically independent. In text, these often correspond to more interpretable semantic features than the principal components from PCA. ICA assumes that the observed embeddings are linear mixtures of underlying, independent source signals (the "semantic components"). It aims to find the transformation matrix to un-mix these signals.

- Collect a large set of sentence embeddings from your domain.
- Apply an ICA algorithm (e.g., from scikit-learn) to this set of vectors.
- The resulting components are vectors that represent independent semantic axes (e.g., one component might capture formality, another might capture topic).
- Once you identify a component you wish to remove, you can use projection techniques (like those in LEACE) to nullify that component in new embeddings.

## Variational Autoencoders (VAEs)

You can train a VAE to encode a sentence into a latent space that is explicitly designed to be disentangled. This means different dimensions of the latent vector correspond to different, independent aspects of the data. A VAE is trained to reconstruct its input after compressing it into a probabilistic latent space. By modifying the VAE's loss function (e.g., using the β-VAE objective), you can put more pressure on the model to learn disentangled, interpretable latent dimensions. Architectures like the DSS-VAE use separate latent variables for syntax and semantics.

- Design a VAE architecture (encoder and decoder).
- Train it on a large corpus of sentences. The training objective is a combination of reconstruction loss and a Kullback-Leibler (KL) divergence term that encourages the desired latent structure.
- After training, the encoder will produce multiple sub-embeddings for a given sentence, each corresponding to a disentangled factor.

## Class Centroid Residuals

- Compute centroids for each class/domain.
- Represent an embedding as (projection onto centroid space) + (residual).
- Residual vector often captures OOD or “orthogonal” meaning.

## Token based

### Sub-embedding by Token Clusters

- Instead of just the pooled embedding, cluster token embeddings within a sentence.
- Aggregate (mean/max) each cluster → yields multiple “sub-embeddings.”

### Attention-Guided Extraction

- Use attention weights to select subsets of tokens (e.g., nouns vs verbs).
- Pool embeddings over those subsets → gives multiple sub-embeddings aligned with syntactic/semantic roles.

## Layer-wise Decomposition

- Extract embeddings from different transformer layers.
- Early layers ≈ syntactic, later layers ≈ semantic.

Combine or analyze separately.

## Sparse Autoencoders for Feature Disentanglement

This approach creates an overcomplete autoencoder, where the hidden layer is much larger than the input layer (n ggd). Sparsity is enforced not with a traditional penalty like L1 loss, but by directly keeping only the top-k strongest activations in the hidden layer. This forces the model to learn a dictionary of features where any given input can be represented by a small combination of them. [link](https://gemini.google.com/u/1/app/d8c4757fc7d4ddbd)

## Supervised Dictionary Learning

You can use Supervised Dictionary Learning to break down a word's vector into a sparse combination of interpretable building blocks, called "atoms." ⚛️ Think of it like a recipe: instead of one complex vector, a word is represented by a few core "ingredients" (the atoms) and their amounts. The "supervised" part ensures that these atoms learn to represent specific, meaningful concepts like grammar (e.g., "noun-ness" or "verb-ness") or sentiment. [link](https://gemini.google.com/u/1/app/f9a152fb20d10808)

or

- Learn a dictionary of basis vectors (sklearn.decomposition.DictionaryLearning).
- Each embedding is expressed as a sparse combination of basis vectors.
- Active dimensions ≈ interpretable sub-embeddings

## Partitioning into Semantic Sub-Embeddings (S³BERT)

Decompose a dense sentence embedding into fixed sub-vectors, each representing a predefined semantic feature (e.g., negation, semantic roles), while a residual captures the rest. This allows selective removal by zeroing out sub-vectors. [link](https://gemini.google.com/u/1/app/d96589836ffa68d0)

## Routing to Specialized Subspaces (MixSP)

This method creates a model that learns to split a sentence's meaning into two distinct parts (e.g., a "high similarity" part and a "low similarity" part). It does this by using a small neural network to decide which part of the embedding vector corresponds to which meaning. [link](https://gemini.google.com/u/1/app/99d1357ccf593a96)

- Train multiple linear projections, each tuned to a subset of data (e.g., in-domain vs OOD).
- Route embeddings through different projectors depending on the sentence type.

## Variational Dropout for Disentangled Transformations

The core idea is to learn a rotation of your embedding space where specific dimensions (a "subspace") are strongly correlated with a specific semantic attribute (like a word's meaning). Variational dropout is the mechanism used to automatically discover which dimensions belong to which attribute by trying to "turn off" as many dimensions as possible for each attribute. The dimensions that resist being turned off are the ones that are important. [link](https://gemini.google.com/u/1/app/db8f1b0d9729f954)

# Results

| Approach                                                                           | ID Score | OOD Score | GQR Score | Resources                                                                                       | Test by |
| :--------------------------------------------------------------------------------- | :------- | :-------- | :-------- | :---------------------------------------------------------------------------------------------- | :------ |
| [semantic-router (threshold=0.2)](https://github.com/aurelio-labs/semantic-router) | 0.61     | 0.94      | 0.74      | [Link](https://github.com/aurelio-labs/semantic-router)                                         | William |
| PCA based router (threshold=0.15, unoptimized)                                     | 0.69     | 0.95      | 0.80      | [Sci-kit PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) | Sebo    |

# GQR-Bench results

🚀[github - gqr](https://github.com/williambrach/gqr), 📈[paper link](https://arxiv.org/pdf/2505.14524)


| Model | Jigsaw | OLID | HateXplain | dkhate | TUKE SK | Web Q | ML Q | Unsafe Avg. | ID Acc. | OOD Acc. | GQR score |
|-------|--------|------|------------|--------|---------|-------|------|-------------|----------------|----------|-----------|
| Llama-Guard-3-8B | 27.07 | 24.77 | 93.28 | 5.17 | 7.51 | 0.10 | 0.00 | 31.56 | - | 22.56 | - |
| Llama-Guard-3-1B | 51.40 | 61.40 | 91.47 | 12.77 | 20.13 | 2.31 | 0.00 | 47.43 | - | 34.21 | - |
| Llama3.2:3B | **99.69** | **99.88** | **99.98** | **100.00** | **100.00** | 99.16 | **100.00** | **99.91** | 26.37 | **99.82** | 41.72 |
| Llama3.1:8B | 94.43 | 93.60 | 97.99 | 95.74 | 97.60 | 90.55 | 46.09 | 95.87 | 95.66 | 88.00 | **91.67** |
| NeMo Guardrails + Llama3.2:3B | 61.42 | 59.65 | 43.15 | 61.09 | 67.88 | 1.67 | 0.00 | 58.64 | - | 58.64 | - |
| NeMo Guardrails + Llama3.1:8B | 51.99 | 36.40 | 20.83 | 10.33 | 27.11 | 0.00 | 0.00 | 29.33 | - | 29.33 | - |
| NeMo Guardrails + GPT-4o-mini | 98.26 | 94.19 | 99.78 | 91.49 | 96.14 | 57.19 | 79.69 | 95.97 | - | 95.97 | - |
| GPT-4o-mini | 94.71 | 93.49 | 98.10 | 94.53 | 98.02 | 90.80 | 45.31 | 95.77 | 95.70 | 87.85 | 91.61 |
| all-MiniLM-L6-v2 + Semantic Router (s=5, t=0.5) | 22.96 | 31.74 | 36.71 | 39.51 | 20.33 | 96.70 | 30.25 | 49.22 | 90.00 | 42.45 | 57.69 |
| bge-small-en-v1.5 + Semantic Router (s=5, t=0.5) | 15.15 | 28.95 | 32.67 | 31.91 | 12.41 | 95.42 | 31.25 | 24.22 | 90.70 | 35.39 | 50.91 |
| fastText | 74.46 | 61.51 | 54.46 | 74.77 | 83.11 | 70.37 | 63.28 | 69.66 | 95.80 | 68.85 | 80.12 |
| WideMLP (t=0.99) | 93.83 | 93.49 | 91.00 | 86.93 | 80.60 | 99.16 | 93.75 | 89.17 | 84.49 | 91.25 | 87.74 |
| WideMLP (t=0.90) | 87.87 | 83.26 | 77.56 | 71.73 | 56.93 | 95.57 | 89.84 | 75.47 | 90.91 | 80.39 | 85.33 |
| WideMLP (t=0.75) | 84.04 | 76.74 | 70.48 | 57.45 | 47.34 | 92.91 | 84.38 | 67.21 | 93.67 | 73.33 | 82.26 |
| ModernBERT-base (t=0.99) | 27.10 | 17.91 | 18.06 | 10.33 | 2.50 | 62.30 | 0.00 | 15.18 | **99.94** | 19.74 | 32.97 |
| BERT-base-multilingual-cased (t=0.99) | 20.91 | 28.26 | 25.44 | 25.84 | 30.87 | 7.28 | 0.00 | 26.26 | 99.90 | 19.80 | 33.05 |
| bge-small-en-v1.5 + SVM | 77.47 | 75.00 | 63.81 | 61.40 | 63.82 | 59.69 | 96.88 | 68.30 | 99.42 | 71.15 | 82.94 |
| bge-small-en-v1.5 + XGBoost | 81.95 | 68.26 | 72.15 | 47.72 | 59.02 | 58.81 | 92.97 | 65.82 | 98.78 | 68.70 | 81.04 |
| all-MiniLM-L6-v2 + SVM | 59.61 | 71.74 | 61.63 | 37.99 | 34.62 | 81.89 | 94.53 | 53.12 | 86.06 | 63.14 | 72.84 |
| all-MiniLM-L6-v2 + XGBoost | 47.57 | 77.44 | 53.14 | 57.45 | 60.17 | 95.47 | 89.84 | 59.15 | 92.93 | 68.73 | 79.02 |
| all-MiniLM-L12-v2 + MLP | 74.77 | 80.47 | 85.59 | 56.23 | 18.87 | 68.45 | 32.81 | 63.19 | 95.17 | 59.60 | 73.23 |
| TF-IDF + SVM | 24.58 | 26.16 | 21.72 | 75.38 | 96.98 | 54.87 | 87.50 | 48.96 | 37.76 | 55.31 | 49.26 |
| TF-IDF + XGBoost | 58.31 | 67.44 | 66.40 | **100.00** | 99.90 | **99.36** | **100.00** | 78.41 | 34.76 | 84.49 | 42.39 |
