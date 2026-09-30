# Airline Sentiment Classification with an Artificial Neural Network

Multi-class sentiment classification of US airline tweets (negative / neutral / positive)
using a feed-forward neural network.

## Dataset

— 14,640 tweets about six US airlines, each labelled with one of three sentiment classes.

The classes are imbalanced:

- negative: 9,178 tweets (62.7%)
- neutral: 3,099 tweets (21.2%)
- positive: 2,363 tweets (16.1%)

Because of this, macro-averaged F1 is the primary evaluation metric, reported alongside accuracy.


## Results

Four configurations were trained: two text representations (tf-idf and word2vec), each with and without class weighting.
The final model was selected on validation macro F1

- **TF-IDF, no class weights (selected):** validation macro F1 0.7484, test accuracy 78.2%,
  test macro F1 0.7195
- TF-IDF with class weights: validation macro F1 0.7270, test accuracy 73.3%, test macro F1 0.6868
- Word2Vec, no class weights: validation macro F1 0.7214, test accuracy 76.3%, test macro F1 0.6740
- Word2Vec with class weights: validation macro F1 0.6958, test accuracy 69.7%, test macro F1 0.6504


## Approach

- Cleaning: every rule was justified by counting how many tweets it affects. HTML entities are decoded, URLs and `@mentions` removed whole, contractions expanded, digits and emojis dropped, and `!` and `?` kept.
- Features: TF-IDF with 5,000 unigram and bigram terms, compared against Word2Vec embeddings
  trained on the same corpus and averaged per tweet.
- Both vectorizers are fitted on the training split.
- Model: two hidden layers of 256 units, each with batch normalisation, ReLU and dropout of 0.5.
- Training: Adam, learning rate 1e-4, batch size 128, early stopping on validation loss with a
  patience of 5 epochs. The weights of the best validation epoch are restored before evaluation.
- Split: 80% train, 10% validation, 10% test, stratified to preserve the class proportions.

## Key findings


- Class weighting did not help. It raises recall on the minority classes but lowers their precision
  by more, so F1 falls for all three classes.
- TF-IDF outperforms Word2Vec by 2.7 to 3.1 macro F1 points.
- `neutral` is the hardest class.

## Project structure

- `data/raw/Tweets.csv` — the original dataset, never modified
- `src/config.py` — paths, random seed and column names
- `src/data_loader.py` — loading and initial quality checks
- `src/eda.py` — noise statistics, vocabulary statistics and plots
- `src/preprocessing.py` — cleaning, contraction expansion and tokenization
- `src/vectorizers.py` — label encoding, splitting, TF-IDF and Word2Vec
- `src/model.py` — the neural network
- `src/train.py` — batching, class weights and the training loop
- `src/evaluate.py` — metrics, loss curves and the confusion matrix
- `src/experiments.py` — the comparison of the four configurations
- `notebooks/01_airline_sentiment_ann.ipynb` — notebook with markdowns for explanation
- `reports/figures/` — plots

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then open `notebooks/01_airline_sentiment_ann.ipynb` and run all cells. A full run trains four
models. Results are reproducible: every random source
is seeded from a single value in `src/config.py`.

## Author

Vasileios Loizidis
