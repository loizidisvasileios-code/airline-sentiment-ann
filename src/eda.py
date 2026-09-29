"""Exploratory Data Analysis: noise, vocabulary size and plots."""

from collections import Counter  # Counts how often each token appears

import matplotlib.pyplot as plt  
import nltk  
from nltk.tokenize import word_tokenize  
from wordcloud import WordCloud, STOPWORDS 

from src.config import FIGURES_DIR, TEXT_COL, LABEL_COL 

nltk.download("punkt_tab", quiet=True)  


NOISE_PATTERNS = {
    "@mentions": r"@\w+",                       # Twitter mentions
    "#hashtags": r"#\w+",                       # Hashtags
    "URLs": r"https?://\S+|www\.\S+",           # Url's
    "HTML entities": r"&\w+;|&#\d+;",           # Encoded characters
    "HTML tags": r"<[^>]+>",                    # Markup such as <b> or <br/>
    "digits": r"\d",                            # Numeric characters
    "emojis": "[\U0001F300-\U0001FAFF☀-➿]",   # Emoji and symbol blocks
}


def _save_and_show(fig, filename):
    """Save a figure into reports/figures and display it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)  # Create the folder on first use
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")  # bbox_inches trims white space
    plt.show()  # Render it in the notebook
    plt.close(fig)  # Release the memory the figure holds


def _corpus_tokens(df, column):
    """Flatten the whole corpus into one lowercase token list."""
    return [token for text in df[column] for token in word_tokenize(text.lower())]


def count_noise_patterns(df, column=TEXT_COL):
    """Report how many tweets contain each pattern."""
    total = len(df) 
    for name, pattern in NOISE_PATTERNS.items():
        matches = df[column].str.contains(pattern, regex=True).sum()  # Tweets with at least one match
        print(f"{name:<15} {matches:>6}  ({matches / total * 100:5.1f}%)")


def print_vocabulary_stats(df, column=TEXT_COL):
    """Report corpus size, vocabulary size and average tweet length."""
    tokens = _corpus_tokens(df, column)  # Every token in the corpus, in order
    print(f"Total tokens          : {len(tokens)}")
    print(f"Unique tokens         : {len(set(tokens))}")  # set() keeps each token once
    print(f"Average tokens/tweet  : {len(tokens) / len(df):.1f}")


def plot_class_distribution(df):
    """Bar plot of how many tweets belong to each sentiment class."""
    counts = df[LABEL_COL].value_counts()  
    fig, ax = plt.subplots(figsize=(6, 4)) 
    ax.bar(counts.index, counts.values, color="steelblue")
    for position, value in enumerate(counts.values):  # Annotate each bar with count and share
        ax.text(position, value, f"{value}\n{value / len(df) * 100:.1f}%", ha="center", va="bottom")
    ax.set_title("Sentiment class distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of tweets")
    ax.set_ylim(0, counts.max() * 1.15)  # Headroom so the labels are not clipped
    _save_and_show(fig, "class_distribution.png")


def plot_tweet_lengths(df, column=TEXT_COL):
    """Histogram of tweet length measured in tokens."""
    lengths = df[column].apply(lambda text: len(word_tokenize(text)))  # Token count per tweet
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=30, color="steelblue", edgecolor="black")
    ax.set_title("Distribution of tweet length")
    ax.set_xlabel("Tokens per tweet")
    ax.set_ylabel("Number of tweets")
    _save_and_show(fig, "tweet_lengths.png")
    print(f"Mean {lengths.mean():.1f} | Median {lengths.median():.0f} | Min {lengths.min()} | Max {lengths.max()}")


def plot_top_words(df, column=TEXT_COL, n=10, exclude_common=True):
    """Horizontal bar plot of the most frequent words.
    """
    tokens = _corpus_tokens(df, column)
    if exclude_common:
        tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS]  # isalpha() drops punctuation
    words, counts = zip(*Counter(tokens).most_common(n))  # Unzip the (word, count) pairs
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(words[::-1], counts[::-1], color="steelblue")  # Reversed so the largest sits on top
    ax.set_title(f"Top {n} most frequent words")
    ax.set_xlabel("Frequency")
    _save_and_show(fig, "top_words.png")


def plot_wordcloud(df, column=TEXT_COL):
    """Word cloud where each word is sized by how often it occurs."""
    text = " ".join(df[column].str.lower()) 
    cloud = WordCloud(width=800, height=400, background_color="white", stopwords=STOPWORDS).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(cloud, interpolation="bilinear")  
    ax.axis("off")  
    ax.set_title("Word cloud of the raw tweets")
    _save_and_show(fig, "wordcloud.png")
