
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "books_processed.csv"
)


def load_processed_books():
    """Load the cleaned book dataset."""

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_DATA_PATH}"
        )

    books = pd.read_csv(PROCESSED_DATA_PATH)

    required_columns = ["title", "content"]

    for column in required_columns:
        if column not in books.columns:
            raise ValueError(
                f"Required column '{column}' is missing."
            )

    books = books.reset_index(drop=True)

    books["title"] = books["title"].fillna("").astype(str)
    books["content"] = books["content"].fillna("").astype(str)

    print("Processed books loaded successfully!")
    print("Total books:", len(books))

    return books


def build_tfidf_matrix(books):
    """Convert book information into numerical vectors."""

    print("\nBuilding TF-IDF model...")

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(books["content"])

    print("TF-IDF model created successfully!")
    print("Number of features:", len(tfidf.get_feature_names_out()))
    print("Matrix shape:", tfidf_matrix.shape)

    return tfidf, tfidf_matrix


def recommend_books(book_title, books, tfidf_matrix, top_n=10):
    """Recommend books similar to the selected book."""

    # Find the selected book
    matching_books = books[
        books["title"].str.casefold()
        == str(book_title).strip().casefold()
    ]

    if matching_books.empty:
        raise ValueError(
            f"Book '{book_title}' was not found in the dataset."
        )

    # Get the position of the selected book
    book_index = matching_books.index[0]

    # Compare the selected book with all other books
    similarity_scores = cosine_similarity(
        tfidf_matrix[book_index],
        tfidf_matrix
    ).flatten()

    # Sort books by similarity
    similar_indices = similarity_scores.argsort()[::-1]

    # Exclude the selected book itself
    similar_indices = [
        index for index in similar_indices
        if index != book_index
    ]

    # Select the top recommendations
    recommended_indices = similar_indices[:top_n]

    recommendations = books.iloc[
        recommended_indices
    ].copy()

    recommendations["similarity_score"] = (
        similarity_scores[recommended_indices]
    )

    return recommendations


if __name__ == "__main__":

    print("\n========== BOOK RECOMMENDATION ENGINE ==========")

    # Load dataset
    books = load_processed_books()

    # Build machine learning model
    tfidf, tfidf_matrix = build_tfidf_matrix(books)

    # Select a sample book
    sample_book = books.iloc[0]["title"]

    print("\nSelected book:", sample_book)

    # Generate recommendations
    recommendations = recommend_books(
        sample_book,
        books,
        tfidf_matrix,
        top_n=10
    )

    # Display recommendations
    display_columns = [
        column for column in [
            "title",
            "authors",
            "average_rating",
            "similarity_score"
        ]
        if column in recommendations.columns
    ]

    print("\nRecommended Books:\n")

    print(
        recommendations[display_columns].to_string(index=False)
    )

    print("\nRecommendation engine working successfully!")
    print("================================================")
