# %% [markdown]
# # Textual Emotion Classification Dataset — EDA & Quality Analysis
#
# This notebook analyzes:
# - Emotion/class distribution
# - Language distribution
# - Concept/product/scenario distributions
# - Sentence length and vocabulary statistics
# - Duplicate and near-duplicate samples
# - Emotion × language relationships
# - Emotion × concept relationships
# - TF-IDF similarity distribution
# - Highly similar / potentially duplicated samples
# - Class imbalance
# - Dataset quality issues
# - Train/validation/test split leakage
#
# Expected JSON structure:
#
# [
#     {
#         "sentence": "Nice, the dsa path is finally becoming much clearer.",
#         "language": "en",
#         "emotion": "Neutral",
#         "concept": "dsa",
#         "product_context": "Zoe",
#         "scenario": "case 500 during dsa practice"
#     }
# ]

# %% [markdown]
# ## 1. Install dependencies
#
# Run this cell once if required.

# %%
# Uncomment if packages are not installed

# !pip install pandas numpy matplotlib seaborn scikit-learn scipy wordcloud openpyxl

# Optional:
# !pip install sentence-transformers umap-learn

# %% [markdown]
# ## 2. Imports

# %%
import json
import re
import os
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

# %% [markdown]
# ## 3. Configuration

# %%
# Change this to your JSON file
DATA_PATH = "dataset.json"

# Directory where analysis outputs will be saved
OUTPUT_DIR = Path("emotion_dataset_analysis")

OUTPUT_DIR.mkdir(exist_ok=True)

# Similarity threshold for identifying potentially duplicated samples
SIMILARITY_THRESHOLD = 0.85

# Number of top similar pairs to display
TOP_SIMILAR_PAIRS = 50

# Random seed for reproducibility
RANDOM_STATE = 42

# %% [markdown]
# ## 4. Load JSON dataset

# %%
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Number of records: {len(data):,}")

df = pd.DataFrame(data)

print("\nColumns:")
print(df.columns.tolist())

display(df.head())

# %% [markdown]
# ## 5. Basic dataset validation

# %%
EXPECTED_COLUMNS = [
    "sentence",
    "language",
    "emotion",
    "concept",
    "product_context",
    "scenario"
]

missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    print("WARNING: Missing columns:")
    print(missing_columns)
else:
    print("All expected columns are present.")

# %% [markdown]
# ## 6. Dataset schema and missing values

# %%
print("Dataset shape:")
print(df.shape)

print("\nData types:")
display(df.dtypes.to_frame("dtype"))

print("\nMissing values:")
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percentage": df.isna().mean() * 100
})

display(
    missing.sort_values(
        "missing_percentage",
        ascending=False
    )
)

# %% [markdown]
# ## 7. Basic cardinality analysis

# %%
cardinality = pd.DataFrame({
    "column": df.columns,
    "unique_values": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ],
    "missing": [
        df[col].isna().sum()
        for col in df.columns
    ]
})

display(cardinality)

# %% [markdown]
# ## 8. Clean text fields for analysis

# %%
def normalize_text(text):
    """
    Basic normalization used for duplicate/similarity analysis.
    Does NOT replace the original sentence.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Normalize punctuation spacing
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text


df["sentence_normalized"] = df["sentence"].apply(normalize_text)

# Sentence length features
df["char_count"] = df["sentence"].fillna("").astype(str).str.len()

df["word_count"] = (
    df["sentence"]
    .fillna("")
    .astype(str)
    .str.findall(r"\b\w+\b")
    .str.len()
)

df["unique_word_count"] = (
    df["sentence"]
    .fillna("")
    .astype(str)
    .apply(
        lambda x: len(set(re.findall(r"\b\w+\b", x.lower())))
    )
)

df["lexical_diversity"] = (
    df["unique_word_count"] /
    df["word_count"].replace(0, np.nan)
)

display(
    df[
        [
            "sentence",
            "char_count",
            "word_count",
            "unique_word_count",
            "lexical_diversity"
        ]
    ].head()
)

# %% [markdown]
# ## 9. Emotion distribution
#
# This is one of the most important analyses for emotion classification.
#
# A strongly imbalanced dataset can cause the classifier to perform well on
# majority classes while performing poorly on minority emotions.

# %%
emotion_counts = df["emotion"].value_counts(dropna=False)

emotion_distribution = pd.DataFrame({
    "count": emotion_counts,
    "percentage": emotion_counts / len(df) * 100
})

display(emotion_distribution)

# %%
plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    y="emotion",
    order=df["emotion"].value_counts().index
)

plt.title("Emotion Distribution")
plt.xlabel("Number of Samples")
plt.ylabel("Emotion")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "emotion_distribution.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# %% [markdown]
# ## 10. Emotion imbalance ratio

# %%
emotion_counts = df["emotion"].value_counts()

majority_count = emotion_counts.max()
minority_count = emotion_counts.min()

print(f"Majority class: {emotion_counts.idxmax()} ({majority_count:,})")
print(f"Minority class: {emotion_counts.idxmin()} ({minority_count:,})")

print(
    f"\nMajority / minority ratio: "
    f"{majority_count / minority_count:.2f}:1"
)

# %% [markdown]
# ## 11. Language distribution

# %%
language_counts = df["language"].value_counts(dropna=False)

language_distribution = pd.DataFrame({
    "count": language_counts,
    "percentage": language_counts / len(df) * 100
})

display(language_distribution)

# %%
plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    y="language",
    order=df["language"].value_counts().index
)

plt.title("Language Distribution")
plt.xlabel("Number of Samples")
plt.ylabel("Language")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "language_distribution.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# %% [markdown]
# ## 12. Emotion × language distribution
#
# This is especially important for multilingual emotion classification.
#
# For example, if most English samples are Positive/Neutral but most Hindi
# samples are Negative, the model might learn language-specific shortcuts
# instead of emotion.

# %%
emotion_language = pd.crosstab(
    df["language"],
    df["emotion"]
)

display(emotion_language)

# %%
emotion_language_percentage = pd.crosstab(
    df["language"],
    df["emotion"],
    normalize="index"
) * 100

display(
    emotion_language_percentage.round(2)
)

# %%
plt.figure(figsize=(14, 8))

sns.heatmap(
    emotion_language_percentage,
    annot=True,
    fmt=".1f",
    cmap="Blues"
)

plt.title("Emotion Distribution Within Each Language (%)")
plt.xlabel("Emotion")
plt.ylabel("Language")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "emotion_language_heatmap.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# %% [markdown]
# ## 13. Concept distribution

# %%
concept_counts = df["concept"].value_counts(dropna=False)

print(f"Number of unique concepts: {df['concept'].nunique()}")

display(
    pd.DataFrame({
        "count": concept_counts,
        "percentage": concept_counts / len(df) * 100
    }).head(30)
)

# %%
plt.figure(figsize=(14, 10))

top_concepts = df["concept"].value_counts().head(30)

sns.barplot(
    x=top_concepts.values,
    y=top_concepts.index
)

plt.title("Top 30 Concepts")
plt.xlabel("Number of Samples")
plt.ylabel("Concept")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_concepts.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# %% [markdown]
# ## 14. Product context distribution

# %%
product_counts = df["product_context"].value_counts(dropna=False)

print(
    f"Unique product contexts: "
    f"{df['product_context'].nunique()}"
)

display(
    pd.DataFrame({
        "count": product_counts,
        "percentage": product_counts / len(df) * 100
    }).head(30)
)

# %% [markdown]
# ## 15. Scenario distribution

# %%
scenario_counts = df["scenario"].value_counts(dropna=False)

print(
    f"Unique scenarios: "
    f"{df['scenario'].nunique()}"
)

display(
    pd.DataFrame({
        "count": scenario_counts,
        "percentage": scenario_counts / len(df) * 100
    }).head(30)
)

# %% [markdown]
# ## 16. Emotion distribution by concept

# %%
concept_emotion = pd.crosstab(
    df["concept"],
    df["emotion"]
)

display(
    concept_emotion.head(30)
)

# %%
# Concepts dominated by a single emotion

concept_emotion_pct = pd.crosstab(
    df["concept"],
    df["emotion"],
    normalize="index"
)

concept_dominant_emotion = pd.DataFrame({
    "dominant_emotion": concept_emotion_pct.idxmax(axis=1),
    "dominant_percentage": concept_emotion_pct.max(axis=1) * 100,
    "sample_count": df["concept"].value_counts()
})

display(
    concept_dominant_emotion
    .sort_values(
        "dominant_percentage",
        ascending=False
    )
    .head(30)
)

# %% [markdown]
# ## 17. Check whether concepts leak the emotion label
#
# A useful diagnostic:
#
# If certain concepts almost always correspond to one emotion, a model may
# exploit concept-specific vocabulary rather than understanding the sentence.

# %%
concept_emotion_summary = (
    concept_dominant_emotion
    .sort_values(
        "dominant_percentage",
        ascending=False
    )
)

highly_dominated = concept_emotion_summary[
    (concept_emotion_summary["dominant_percentage"] >= 90) &
    (concept_emotion_summary["sample_count"] >= 10)
]

print(
    f"Concepts with >=90% one emotion and >=10 samples: "
    f"{len(highly_dominated)}"
)

display(highly_dominated.head(50))

# %% [markdown]
# ## 18. Sentence length statistics

# %%
length_stats = df[
    ["char_count", "word_count", "unique_word_count", "lexical_diversity"]
].describe().T

display(length_stats)

# %% [markdown]
# ## 19. Sentence word-count distribution

# %%
plt.figure(figsize=(12, 6))

sns.histplot(
    data=df,
    x="word_count",
    bins=50,
    kde=True
)

plt.title("Sentence Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Samples")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sentence_length_distribution.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# %% [markdown]
# ## 20. Sentence length by emotion

# %%
plt.figure(figsize=(14, 7))

sns.boxplot(
    data=df,
    x="emotion",
    y="word_count"
)

plt.xticks(rotation=45)

plt.title("Sentence Length by Emotion")
plt.xlabel("Emotion")
plt.ylabel("Word Count")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sentence_length_by_emotion.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# %% [markdown]
# ## 21. Very short and very long samples

# %%
print("Shortest sentences:")

display(
    df[
        [
            "sentence",
            "emotion",
            "language",
            "concept",
            "word_count"
        ]
    ]
    .sort_values("word_count")
    .head(20)
)

print("\nLongest sentences:")

display(
    df[
        [
            "sentence",
            "emotion",
            "language",
            "concept",
            "word_count"
        ]
    ]
    .sort_values("word_count", ascending=False)
    .head(20)
)

# %% [markdown]
# ## 22. Exact duplicate analysis
#
# Exact duplicates are particularly important.
#
# If the same sentence appears in both train and validation/test sets,
# evaluation metrics can become artificially high.

# %%
duplicate_mask = df["sentence_normalized"].duplicated(
    keep=False
)

duplicates = df[duplicate_mask].copy()

print(
    f"Rows belonging to duplicate sentence groups: "
    f"{len(duplicates):,}"
)

print(
    f"Unique duplicated sentences: "
    f"{duplicates['sentence_normalized'].nunique():,}"
)

# %%
if len(duplicates) > 0:
    display(
        duplicates[
            [
                "sentence",
                "emotion",
                "language",
                "concept",
                "product_context",
                "scenario"
            ]
        ]
        .sort_values("sentence_normalized")
        .head(100)
    )

# %% [markdown]
# ## 23. Conflicting labels for duplicate sentences
#
# This detects the same normalized sentence having different emotion labels.
#
# These samples need manual investigation.

# %%
duplicate_label_groups = (
    df.groupby("sentence_normalized")
    .agg(
        sample_count=("emotion", "size"),
        unique_emotions=("emotion", "nunique"),
        emotions=("emotion", lambda x: sorted(set(x))),
        unique_languages=("language", "nunique")
    )
    .reset_index()
)

conflicting_duplicates = duplicate_label_groups[
    duplicate_label_groups["unique_emotions"] > 1
]

print(
    f"Duplicate sentence groups with conflicting emotions: "
    f"{len(conflicting_duplicates):,}"
)

display(conflicting_duplicates.head(100))

# %% [markdown]
# ## 24. TF-IDF representation
#
# TF-IDF gives us a simple lexical representation that can be used to
# calculate sentence-to-sentence similarity.
#
# This is useful for identifying:
# - Near duplicates
# - Template-generated samples
# - Repeated sentence patterns
# - Potential train/test leakage

# %%
texts = df["sentence_normalized"].fillna("").tolist()

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.98,
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(texts)

print("TF-IDF matrix shape:")
print(tfidf_matrix.shape)

# %% [markdown]
# ## 25. Similarity distribution
#
# Calculating a complete NxN similarity matrix can become expensive for
# large datasets.
#
# For datasets up to several tens of thousands of samples this may be
# manageable, but for very large datasets use the alternative block-based
# method further below.

# %%
# For smaller datasets
MAX_FULL_SIMILARITY_SAMPLES = 10000

if len(df) <= MAX_FULL_SIMILARITY_SAMPLES:

    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Remove self-similarity
    np.fill_diagonal(similarity_matrix, 0)

    upper_triangle = similarity_matrix[
        np.triu_indices_from(similarity_matrix, k=1)
    ]

    similarity_stats = pd.Series(upper_triangle).describe()

    display(similarity_stats)

else:
    print(
        f"Dataset has {len(df):,} rows."
    )
    print(
        "Skipping full NxN similarity matrix because it may require "
        "too much memory."
    )

# %% [markdown]
# ## 26. Plot similarity distribution

# %%
if len(df) <= MAX_FULL_SIMILARITY_SAMPLES:

    plt.figure(figsize=(12, 6))

    sns.histplot(
        upper_triangle,
        bins=100,
        kde=True
    )

    plt.title("TF-IDF Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of Sentence Pairs")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "similarity_distribution.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

# %% [markdown]
# ## 27. Find highly similar sentence pairs

# %%
if len(df) <= MAX_FULL_SIMILARITY_SAMPLES:

    rows, cols = np.where(
        similarity_matrix >= SIMILARITY_THRESHOLD
    )

    similar_pairs = []

    for i, j in zip(rows, cols):

        if i >= j:
            continue

        similar_pairs.append({
            "index_1": i,
            "index_2": j,
            "similarity": similarity_matrix[i, j],
            "sentence_1": df.iloc[i]["sentence"],
            "sentence_2": df.iloc[j]["sentence"],
            "emotion_1": df.iloc[i]["emotion"],
            "emotion_2": df.iloc[j]["emotion"],
            "language_1": df.iloc[i]["language"],
            "language_2": df.iloc[j]["language"],
            "concept_1": df.iloc[i]["concept"],
            "concept_2": df.iloc[j]["concept"]
        })

    similar_pairs_df = pd.DataFrame(similar_pairs)

    if len(similar_pairs_df) > 0:

        similar_pairs_df = (
            similar_pairs_df
            .sort_values(
                "similarity",
                ascending=False
            )
            .reset_index(drop=True)
        )

        print(
            f"Highly similar pairs: "
            f"{len(similar_pairs_df):,}"
        )

        display(
            similar_pairs_df.head(TOP_SIMILAR_PAIRS)
        )

    else:
        print(
            f"No pairs found above similarity threshold "
            f"{SIMILARITY_THRESHOLD}."
        )

# %% [markdown]
# ## 28. Similarity vs emotion agreement
#
# This is particularly useful for understanding annotation consistency.
#
# For highly similar sentences:
#
# - Same emotion → probably good
# - Different emotion → investigate manually
#
# Some differences are legitimate, but a large number can indicate noisy
# labels or template problems.

# %%
if (
    len(df) <= MAX_FULL_SIMILARITY_SAMPLES
    and len(similar_pairs_df) > 0
):

    similar_pairs_df["same_emotion"] = (
        similar_pairs_df["emotion_1"]
        ==
        similar_pairs_df["emotion_2"]
    )

    print(
        "Emotion agreement among highly similar pairs:"
    )

    display(
        similar_pairs_df["same_emotion"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percentage")
        .to_frame()
    )

# %% [markdown]
# ## 29. Highly similar sentences with DIFFERENT emotions
#
# These are high-priority samples for manual review.

# %%
if (
    len(df) <= MAX_FULL_SIMILARITY_SAMPLES
    and len(similar_pairs_df) > 0
):

    conflicting_similar = similar_pairs_df[
        similar_pairs_df["emotion_1"]
        !=
        similar_pairs_df["emotion_2"]
    ]

    print(
        f"Highly similar pairs with different emotions: "
        f"{len(conflicting_similar):,}"
    )

    display(
        conflicting_similar.head(100)
    )

# %% [markdown]
# ## 30. Most similar pair for every sentence
#
# This provides another way to inspect dataset redundancy.

# %%
if len(df) <= MAX_FULL_SIMILARITY_SAMPLES:

    nearest_neighbor_idx = similarity_matrix.argmax(axis=1)
    nearest_neighbor_score = similarity_matrix.max(axis=1)

    df["nearest_neighbor_index"] = nearest_neighbor_idx
    df["nearest_neighbor_similarity"] = nearest_neighbor_score

    df["nearest_neighbor_sentence"] = [
        df.iloc[idx]["sentence"]
        for idx in nearest_neighbor_idx
    ]

    display(
        df[
            [
                "sentence",
                "emotion",
                "nearest_neighbor_similarity",
                "nearest_neighbor_sentence"
            ]
        ]
        .sort_values(
            "nearest_neighbor_similarity",
            ascending=False
        )
        .head(50)
    )

# %% [markdown]
# ## 31. Emotion-specific similarity
#
# Check whether samples belonging to the same emotion tend to be more
# similar to one another than samples from different emotions.

# %%
if len(df) <= MAX_FULL_SIMILARITY_SAMPLES:

    emotion_pairs = []

    n = len(df)

    # Sample pairs to avoid excessive computation
    max_pairs = min(200000, n * (n - 1) // 2)

    rng = np.random.default_rng(RANDOM_STATE)

    for _ in range(max_pairs):

        i, j = rng.choice(n, size=2, replace=False)

        emotion_pairs.append({
            "similarity": similarity_matrix[i, j],
            "same_emotion": (
                df.iloc[i]["emotion"]
                ==
                df.iloc[j]["emotion"]
            )
        })

    emotion_similarity_df = pd.DataFrame(emotion_pairs)

    display(
        emotion_similarity_df
        .groupby("same_emotion")["similarity"]
        .describe()
    )

# %% [markdown]
# ## 32. Emotion × language × concept coverage
#
# Identify combinations with very few examples.
#
# Small groups can cause unstable validation/test results.

# %%
coverage = (
    df.groupby(
        ["language", "emotion"],
        dropna=False
    )
    .size()
    .reset_index(name="count")
    .sort_values("count")
)

display(coverage.head(50))

# %%
# Groups with fewer than 10 examples

rare_combinations = coverage[
    coverage["count"] < 10
]

print(
    f"Language/emotion combinations with <10 samples: "
    f"{len(rare_combinations)}"
)

display(rare_combinations)

# %% [markdown]
# ## 33. Emotion distribution within each language

# %%
language_emotion_counts = (
    df.groupby(
        ["language", "emotion"]
    )
    .size()
    .reset_index(name="count")
)

language_totals = (
    df.groupby("language")
    .size()
    .reset_index(name="language_total")
)

language_emotion_counts = language_emotion_counts.merge(
    language_totals,
    on="language"
)

language_emotion_counts["percentage"] = (
    language_emotion_counts["count"]
    /
    language_emotion_counts["language_total"]
    * 100
)

display(
    language_emotion_counts.sort_values(
        ["language", "count"],
        ascending=[True, False]
    )
)

# %% [markdown]
# ## 34. Dataset entropy by language
#
# A useful measure of whether emotions are reasonably distributed within
# each language.
#
# Higher entropy generally means more balanced emotion coverage.

# %%
from scipy.stats import entropy

language_entropy = []

for language, group in df.groupby("language"):

    counts = group["emotion"].value_counts()

    probabilities = counts / counts.sum()

    language_entropy.append({
        "language": language,
        "emotion_entropy": entropy(probabilities),
        "num_emotions": len(counts),
        "sample_count": len(group)
    })

language_entropy_df = pd.DataFrame(
    language_entropy
).sort_values(
    "emotion_entropy"
)

display(language_entropy_df)

# %% [markdown]
# ## 35. Emotion entropy by concept
#
# Low entropy means a concept is strongly associated with a small number
# of emotions.

# %%
concept_entropy = []

for concept, group in df.groupby("concept"):

    counts = group["emotion"].value_counts()

    probabilities = counts / counts.sum()

    concept_entropy.append({
        "concept": concept,
        "emotion_entropy": entropy(probabilities),
        "num_emotions": len(counts),
        "sample_count": len(group)
    })

concept_entropy_df = pd.DataFrame(
    concept_entropy
).sort_values(
    "emotion_entropy"
)

display(
    concept_entropy_df.head(50)
)

# %% [markdown]
# ## 36. Find suspiciously templated sentences
#
# We look for sentences with a very common normalized pattern.
#
# This is useful when data was generated from templates.

# %%
sentence_counts = (
    df["sentence_normalized"]
    .value_counts()
)

print("Most repeated sentences:")

display(
    sentence_counts
    .head(30)
    .to_frame("count")
)

# %% [markdown]
# ## 37. Common sentence prefixes

# %%
def first_words(text, n=5):
    words = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(words[:n])


df["sentence_prefix"] = df["sentence"].apply(
    lambda x: first_words(x, 5)
)

prefix_counts = (
    df["sentence_prefix"]
    .value_counts()
)

display(
    prefix_counts.head(50).to_frame("count")
)

# %% [markdown]
# ## 38. Common sentence suffixes

# %%
def last_words(text, n=5):
    words = re.findall(r"\b\w+\b", str(text).lower())
    return " ".join(words[-n:])


df["sentence_suffix"] = df["sentence"].apply(
    lambda x: last_words(x, 5)
)

suffix_counts = (
    df["sentence_suffix"]
    .value_counts()
)

display(
    suffix_counts.head(50).to_frame("count")
)

# %% [markdown]
# ## 39. Vocabulary statistics

# %%
all_words = []

for sentence in df["sentence"].dropna():

    words = re.findall(
        r"\b[\w'-]+\b",
        str(sentence).lower()
    )

    all_words.extend(words)

word_counts = Counter(all_words)

print(f"Total tokens: {len(all_words):,}")
print(f"Unique tokens: {len(word_counts):,}")

vocab_df = pd.DataFrame(
    word_counts.most_common(100),
    columns=["word", "frequency"]
)

display(vocab_df.head(50))

# %% [markdown]
# ## 40. Vocabulary frequency plot

# %%
plt.figure(figsize=(14, 7))

top_words = word_counts.most_common(30)

words = [x[0] for x in top_words]
counts = [x[1] for x in top_words]

sns.barplot(
    x=counts,
    y=words
)

plt.title("Top 30 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Word")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_words.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# %% [markdown]
# ## 41. Label consistency check
#
# Check whether the same sentence has:
# - Multiple emotions
# - Multiple languages
# - Multiple concepts
#
# These aren't necessarily errors, but they deserve inspection.

# %%
consistency = (
    df.groupby("sentence_normalized")
    .agg(
        samples=("sentence_normalized", "size"),
        emotions=("emotion", "nunique"),
        languages=("language", "nunique"),
        concepts=("concept", "nunique"),
        products=("product_context", "nunique"),
        scenarios=("scenario", "nunique")
    )
    .reset_index()
)

display(
    consistency[
        (consistency["samples"] > 1)
    ]
    .sort_values(
        "samples",
        ascending=False
    )
    .head(100)
)

# %% [markdown]
# ## 42. Identify possible label conflicts

# %%
label_conflicts = consistency[
    (
        (consistency["emotions"] > 1)
        |
        (consistency["languages"] > 1)
    )
]

print(
    f"Potential label conflicts: "
    f"{len(label_conflicts):,}"
)

display(label_conflicts.head(100))

# %% [markdown]
# ## 43. Train/validation/test split analysis
#
# IMPORTANT:
#
# A random row-level split can cause leakage if there are many near-duplicate
# or template-generated samples.
#
# First we'll create a basic stratified split.

# %%
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["emotion"],
    random_state=RANDOM_STATE
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["emotion"],
    random_state=RANDOM_STATE
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# %% [markdown]
# ## 44. Compare emotion distributions across splits

# %%
split_distribution = pd.DataFrame({
    "train": train_df["emotion"].value_counts(normalize=True),
    "validation": val_df["emotion"].value_counts(normalize=True),
    "test": test_df["emotion"].value_counts(normalize=True)
}).fillna(0) * 100

display(
    split_distribution.round(2)
)

# %%
split_distribution.plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title("Emotion Distribution Across Dataset Splits")
plt.xlabel("Emotion")
plt.ylabel("Percentage")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "split_emotion_distribution.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# %% [markdown]
# ## 45. Exact sentence leakage between splits

# %%
train_sentences = set(train_df["sentence_normalized"])
val_sentences = set(val_df["sentence_normalized"])
test_sentences = set(test_df["sentence_normalized"])

train_val_overlap = train_sentences.intersection(
    val_sentences
)

train_test_overlap = train_sentences.intersection(
    test_sentences
)

val_test_overlap = val_sentences.intersection(
    test_sentences
)

print(
    f"Train ↔ Validation exact overlap: "
    f"{len(train_val_overlap):,}"
)

print(
    f"Train ↔ Test exact overlap: "
    f"{len(train_test_overlap):,}"
)

print(
    f"Validation ↔ Test exact overlap: "
    f"{len(val_test_overlap):,}"
)

# %% [markdown]
# ## 46. Save duplicate samples for manual review

# %%
if len(duplicates) > 0:

    duplicate_output = (
        OUTPUT_DIR /
        "duplicate_sentences.csv"
    )

    duplicates.to_csv(
        duplicate_output,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Saved: {duplicate_output}"
    )

# %% [markdown]
# ## 47. Save conflicting samples

# %%
if len(conflicting_duplicates) > 0:

    conflict_output = (
        OUTPUT_DIR /
        "conflicting_duplicate_labels.csv"
    )

    conflicting_duplicates.to_csv(
        conflict_output,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Saved: {conflict_output}"
    )

# %% [markdown]
# ## 48. Save highly similar pairs

# %%
if (
    len(df) <= MAX_FULL_SIMILARITY_SAMPLES
    and len(similar_pairs_df) > 0
):

    similarity_output = (
        OUTPUT_DIR /
        "high_similarity_pairs.csv"
    )

    similar_pairs_df.to_csv(
        similarity_output,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Saved: {similarity_output}"
    )

# %% [markdown]
# ## 49. Create a dataset quality report

# %%
quality_report = {
    "total_samples": len(df),
    "unique_sentences": df["sentence_normalized"].nunique(),
    "duplicate_rows": int(duplicate_mask.sum()),
    "duplicate_sentence_groups": int(
        duplicates["sentence_normalized"].nunique()
    ) if len(duplicates) > 0 else 0,
    "conflicting_duplicate_groups": len(conflicting_duplicates),
    "num_emotions": df["emotion"].nunique(),
    "num_languages": df["language"].nunique(),
    "num_concepts": df["concept"].nunique(),
    "num_products": df["product_context"].nunique(),
    "num_scenarios": df["scenario"].nunique(),
    "mean_word_count": df["word_count"].mean(),
    "median_word_count": df["word_count"].median(),
    "min_word_count": df["word_count"].min(),
    "max_word_count": df["word_count"].max(),
    "missing_sentence": int(df["sentence"].isna().sum()),
    "missing_emotion": int(df["emotion"].isna().sum()),
    "missing_language": int(df["language"].isna().sum()),
}

quality_report_df = pd.DataFrame(
    quality_report.items(),
    columns=["metric", "value"]
)

display(quality_report_df)

quality_report_df.to_csv(
    OUTPUT_DIR / "quality_report.csv",
    index=False
)

# %% [markdown]
# ## 50. Export useful analysis tables to Excel

# %%
excel_path = OUTPUT_DIR / "dataset_analysis.xlsx"

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    emotion_distribution.to_excel(
        writer,
        sheet_name="Emotion"
    )

    language_distribution.to_excel(
        writer,
        sheet_name="Language"
    )

    emotion_language.to_excel(
        writer,
        sheet_name="Emotion_Language"
    )

    coverage.to_excel(
        writer,
        sheet_name="Coverage",
        index=False
    )

    concept_dominant_emotion.to_excel(
        writer,
        sheet_name="Concept_Emotion"
    )

    language_entropy_df.to_excel(
        writer,
        sheet_name="Language_Entropy",
        index=False
    )

    concept_entropy_df.to_excel(
        writer,
        sheet_name="Concept_Entropy",
        index=False
    )

    quality_report_df.to_excel(
        writer,
        sheet_name="Quality_Report",
        index=False
    )

print(
    f"Analysis workbook saved to:\n{excel_path}"
)

# %% [markdown]
# ## 51. Optional: semantic similarity using Sentence Transformers
#
# TF-IDF only captures lexical similarity.
#
# For emotion datasets, semantic similarity is often more useful.
#
# Example:
#
# "I'm really happy with this"
#
# and
#
# "This made me feel fantastic"
#
# have low lexical overlap but high semantic similarity.
#
# Uncomment the following cells if you want embedding-based analysis.

# %%
# !pip install sentence-transformers

# %%
# from sentence_transformers import SentenceTransformer
#
# embedding_model = SentenceTransformer(
#     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )
#
# embeddings = embedding_model.encode(
#     df["sentence"].fillna("").tolist(),
#     show_progress_bar=True,
#     normalize_embeddings=True
# )
#
# print(embeddings.shape)

# %% [markdown]
# ## 52. Semantic similarity analysis
#
# This can identify paraphrases that TF-IDF misses.

# %%
# semantic_similarity = cosine_similarity(embeddings)

# Remove self similarity
# np.fill_diagonal(semantic_similarity, 0)

# semantic_rows, semantic_cols = np.where(
#     semantic_similarity >= 0.85
# )

# semantic_pairs = []

# for i, j in zip(semantic_rows, semantic_cols):

#     if i >= j:
#         continue

#     semantic_pairs.append({
#         "similarity": semantic_similarity[i, j],
#         "sentence_1": df.iloc[i]["sentence"],
#         "sentence_2": df.iloc[j]["sentence"],
#         "emotion_1": df.iloc[i]["emotion"],
#         "emotion_2": df.iloc[j]["emotion"],
#         "language_1": df.iloc[i]["language"],
#         "language_2": df.iloc[j]["language"]
#     })

# semantic_pairs_df = pd.DataFrame(
#     semantic_pairs
# ).sort_values(
#     "similarity",
#     ascending=False
# )

# display(
#     semantic_pairs_df.head(100)
# )

# %% [markdown]
# ## 53. Optional: embedding-based visualization
#
# UMAP can provide a 2D visualization of the semantic structure of the
# dataset.
#
# Useful questions:
#
# - Do emotions form distinct clusters?
# - Are languages separated?
# - Are some concepts dominating the embedding space?
# - Are there outliers?

# %%
# !pip install umap-learn

# %%
# import umap
#
# reducer = umap.UMAP(
#     n_components=2,
#     random_state=RANDOM_STATE
# )
#
# embedding_2d = reducer.fit_transform(embeddings)

# %%
# plt.figure(figsize=(12, 8))
#
# sns.scatterplot(
#     x=embedding_2d[:, 0],
#     y=embedding_2d[:, 1],
#     hue=df["emotion"],
#     alpha=0.6,
#     s=30
# )
#
# plt.title("Semantic Embedding Space — Colored by Emotion")
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
#
# plt.legend(
#     bbox_to_anchor=(1.05, 1),
#     loc="upper left"
# )
#
# plt.tight_layout()
#
# plt.savefig(
#     OUTPUT_DIR / "embedding_emotion_umap.png",
#     dpi=200,
#     bbox_inches="tight"
# )
#
# plt.show()

# %% [markdown]
# ## 54. Final dataset summary

# %%
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Total samples       : {len(df):,}")
print(f"Unique sentences    : {df['sentence_normalized'].nunique():,}")
print(f"Unique emotions     : {df['emotion'].nunique():,}")
print(f"Unique languages    : {df['language'].nunique():,}")
print(f"Unique concepts     : {df['concept'].nunique():,}")
print(f"Unique products     : {df['product_context'].nunique():,}")
print(f"Unique scenarios    : {df['scenario'].nunique():,}")

print("\nEmotion distribution:")
print(df["emotion"].value_counts())

print("\nLanguage distribution:")
print(df["language"].value_counts())

print("\nSentence length:")
print(
    f"Mean words   : {df['word_count'].mean():.2f}"
)

print(
    f"Median words : {df['word_count'].median():.2f}"
)

print(
    f"Min words    : {df['word_count'].min()}"
)

print(
    f"Max words    : {df['word_count'].max()}"
)

print("\nDuplicate analysis:")
print(
    f"Duplicate rows: {duplicate_mask.sum():,}"
)

print(
    f"Conflicting duplicate groups: "
    f"{len(conflicting_duplicates):,}"
)

print("=" * 70)

# %% [markdown]
# ## 55. Recommended manual-review queues

# %%
print("Files generated in:", OUTPUT_DIR.resolve())

for path in sorted(OUTPUT_DIR.iterdir()):
    print(" -", path.name)