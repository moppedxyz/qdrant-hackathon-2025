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

# Results

| Approach | ID Score | OOD Score | GQR Score | Resources |  Test by |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [semantic-router (threshold=0.2)](https://github.com/aurelio-labs/semantic-router) | 0.61 | 0.94 |  0.74 |[Link](https://github.com/aurelio-labs/semantic-router) | William |
