# Airline Sentiment Classification with an Artificial Neural Network

Multi-class sentiment classification of US airline tweets (negative / neutral / positive)
using a feed-forward neural network built in PyTorch.

## Dataset

[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
— 14,640 tweets about six US airlines, each labelled with one of three sentiment classes.

| Class | Count | Share |
|-----------|-------|-------|
| negative  | 9,178 | 62.7% |
| neutral   | 3,099 | 21.2% |
| positive  | 2,363 | 16.1% |

The dataset is imbalanced, so macro-averaged F1 is used as the primary evaluation metric
alongside accuracy.

## Project structure

```
airline-sentiment-ann/
├── data/raw/Tweets.csv     # Original dataset
├── src/                    # Source modules
├── notebooks/              # Analysis notebook (main deliverable)
├── reports/figures/        # Generated plots
├── requirements.txt        # Pinned dependencies
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Status

Work in progress.

- [x] Project setup
- [ ] Data loading and initial checks
- [ ] Exploratory data analysis
- [ ] Text preprocessing
- [ ] Feature extraction (TF-IDF, Word2Vec)
- [ ] ANN model
- [ ] Training with early stopping
- [ ] Evaluation and experiments

## Author

Vasileios Loizidis — Athens University of Economics and Business,
Deep Learning / Natural Language Processing
