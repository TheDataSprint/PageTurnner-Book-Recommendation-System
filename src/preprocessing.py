
from pathlib import Path
import pandas as pd

from src.data_loader import load_core_book_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def clean_books(books):
    """
    Clean book information.
    """

    print("\nStarting book cleaning...")

    # Remove duplicate book IDs
    books = books.drop_duplicates(
        subset="book_id",
        keep="first"
    ).copy()

    # Use original title if title is missing
    if "original_title" in books.columns:
        books["title"] = books["title"].fillna(
            books["original_title"]
        )

    # Remove books without a usable title
    books["title"] = books["title"].fillna("").astype(str).str.strip()
    books = books[books["title"] != ""].copy()

    # Handle missing authors
    books["authors"] = (
        books["authors"]
        .fillna("Unknown Author")
        .astype(str)
        .str.strip()
    )

    # Clean numerical columns
    if "average_rating" in books.columns:
        books["average_rating"] = pd.to_numeric(
            books["average_rating"],
            errors="coerce"
        )

    if "original_publication_year" in books.columns:
        books["original_publication_year"] = pd.to_numeric(
            books["original_publication_year"],
            errors="coerce"
        )

    print("Book cleaning completed!")

    return books


def create_book_tags(books, tags, book_tags):
    """
    Connect book information with tag names.
    """

    print("\nConnecting books with tags...")

    # Convert tag counts into numbers
    book_tags = book_tags.copy()

    book_tags["count"] = pd.to_numeric(
        book_tags["count"],
        errors="coerce"
    ).fillna(0)

    # Select the 10 most frequently assigned tags per book
    top_tags = (
        book_tags
        .sort_values("count", ascending=False)
        .groupby("goodreads_book_id")
        .head(10)
    )

    # Connect tag IDs with actual tag names
    tagged_books = top_tags.merge(
        tags[["tag_id", "tag_name"]],
        on="tag_id",
        how="left"
    )

    # Normalize tag names
    tagged_books["tag_name"] = (
        tagged_books["tag_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
    )

    # Remove reading-status tags
    ignored_tags = {
        "to read",
        "currently reading",
        "owned",
        "books i own",
        "favorites",
        "favourites"
    }

    tagged_books = tagged_books[
        ~tagged_books["tag_name"].isin(ignored_tags)
        & (tagged_books["tag_name"] != "")
    ]

    # Combine multiple tags into one text column
    tags_by_book = (
        tagged_books
        .groupby("goodreads_book_id")["tag_name"]
        .agg(" ".join)
        .reset_index(name="tags_text")
    )

    # Attach the tags to the correct books
    books = books.merge(
        tags_by_book,
        on="goodreads_book_id",
        how="left"
    )

    books["tags_text"] = books["tags_text"].fillna("")

    print("Book tags connected successfully!")

    return books


def create_book_content(books):
    """
    Combine book title, author and tags
    into one text field for machine learning.
    """

    books["content"] = (
        books["title"] + " " +
        books["authors"] + " " +
        books["tags_text"]
    )

    return books


def prepare_dataset():
    """
    Main preprocessing function.
    """

    # Load original data
    books, tags, book_tags = load_core_book_data()

    original_count = len(books)

    # Clean books
    books = clean_books(books)

    # Connect tags
    books = create_book_tags(
        books,
        tags,
        book_tags
    )

    # Create combined book information
    books = create_book_content(books)

    # Create processed directory
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed dataset
    output_path = PROCESSED_DATA_DIR / "books_processed.csv"

    books.to_csv(output_path, index=False)

    print("\n========== PREPROCESSING REPORT ==========")
    print("Original books:", original_count)
    print("Cleaned books:", len(books))
    print("Duplicate or unusable records removed:",
          original_count - len(books))
    print("Books with tags:", (books["tags_text"] != "").sum())
    print("Processed file:", output_path)
    print("==========================================")

    return books


if __name__ == "__main__":
    prepare_dataset()
