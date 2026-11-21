from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_csv(filename):
    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    print(f"Loading {filename}...")

    return pd.read_csv(file_path)


def load_core_book_data():
    books = load_csv("books.csv")
    tags = load_csv("tags.csv")
    book_tags = load_csv("book_tags.csv")

    print("All core datasets loaded successfully!")

    return books, tags, book_tags