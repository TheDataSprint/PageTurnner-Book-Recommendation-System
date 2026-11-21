# 📚 PageTurnner — Book Recommendation System

> Discover your next favourite book with recommendations based on the books' content, authors, and reader-generated tags.

<p align="center">
  <a href="https://pageturnner-book-recommendation-system-28mfc9e7gkhy4cs4vtu2a2.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20PageTurnner-2E7D5B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open live demo">
  </a>
  <a href="https://github.com/TheDataSprint/PageTurnner-Book-Recommendation-System">
    <img src="https://img.shields.io/badge/Source%20Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="View source code on GitHub">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-Project-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/ML-TF--IDF%20%7C%20Cosine%20Similarity-6C5CE7" alt="Machine learning">
  <img src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey" alt="CC BY-SA 4.0 license">
</p>

## 🌐 Live Demo

**Try PageTurnner:** [Launch the live application](https://pageturnner-book-recommendation-system-28mfc9e7gkhy4cs4vtu2a2.streamlit.app/)

**Repository:** [TheDataSprint/PageTurnner-Book-Recommendation-System](https://github.com/TheDataSprint/PageTurnner-Book-Recommendation-System)

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [How to Use](#-how-to-use)
- [Storage Notes and Current Limitations](#-storage-notes-and-current-limitations)
- [Future Improvements](#-future-improvements)
- [Project Information](#-project-information)
- [License and Attribution](#-license-and-attribution)

## 📖 About the Project

**PageTurnner** is a book discovery and recommendation web application built with **Python**, **Streamlit**, and machine learning techniques. It helps readers find books related to a title they already enjoy and explore popular books from the Goodbooks-10k collection.

The recommendation engine uses **content-based filtering**. It creates a text profile for each book from information such as its title, author, and reader-generated tags. It then uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to represent the text numerically and **cosine similarity** to compare books.

The project combines a recommendation engine with an interactive interface where users can discover books, filter popular titles, and save books to a reading shelf.

## ✨ Features

- **Book discovery:** Search the collection by book title.
- **Similar-book recommendations:** Find books with content related to a selected title.
- **Surprise Me:** Get a randomly selected book as a starting point for discovery.
- **Genre filtering:** Explore books by available genre categories.
- **Rating filter:** Set a minimum average rating for displayed books.
- **Sorting:** Arrange recommendation results by best match or highest rating.
- **Popular Books:** Browse popular titles using rating-count information, with genre and rating filters.
- **My Reading Shelf:** Save books to a personal reading list within the running application.
- **Book details:** View available cover images, authors, average ratings, publication years, and content-match indicators.
- **About page:** Read an overview of the recommendation process and the technologies used.
- **Web deployment:** Access the application through the public Streamlit demo.

## 🧠 How It Works

PageTurnner follows a content-based recommendation pipeline:

1. **Load the prepared book data**  
   The application reads the cleaned dataset from `data/processed/books_processed.csv`.

2. **Build a text profile for each book**  
   Book information, including titles, authors, and reader-generated tags, is represented in a text `content` field.

3. **Convert text into numerical features with TF-IDF**  
   The `TfidfVectorizer` identifies informative words and phrases and converts each book's text profile into a numerical vector. The current implementation uses English stop-word removal, up to 50,000 features, and one- and two-word phrases.

4. **Compare books using cosine similarity**  
   Cosine similarity measures how close the selected book's vector is to the vectors of other books.

5. **Rank and display recommendations**  
   The application presents books with stronger content similarity first. Users can then apply the available genre, rating, and sorting options.

> **Recommendation approach:** The current recommendation engine is content-based. Although the repository includes ratings-related dataset files, the application does not currently train a collaborative-filtering or hybrid recommendation model from individual user ratings.

## 🛠️ Technology Stack

| Technology | Role in the Project |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application and user interface |
| Pandas | Loading, cleaning, and working with tabular book data |
| NumPy | Numerical computing support |
| Scikit-learn | TF-IDF vectorization and cosine-similarity calculations |
| TF-IDF | Converts book text into weighted numerical features |
| Cosine Similarity | Measures similarity between book content vectors |
| Git and GitHub | Source control and project hosting |
| Streamlit Community Cloud | Hosting the live web application |

The exact Python package versions used by the project are listed in [`requirements.txt`](requirements.txt).

## 📊 Dataset

PageTurnner uses the **[Goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k)**, a collection of book metadata and reader-related information based on Goodreads data.

The dataset files are included in this repository, so you do **not** need to download them separately to run the current application.

| File | Description |
|---|---|
| `books.csv` | Raw book information and metadata |
| `tags.csv` | Tag identifiers and tag names |
| `book_tags.csv` | Relationships between books and tags |
| `ratings.csv` | User rating records from the dataset |
| `to_read.csv` | User-to-book reading-list records from the dataset |
| `books_processed.csv` | Cleaned and prepared book data used by the live recommendation application |

The deployed recommendation engine reads `books_processed.csv`. The other CSV files are retained in the repository as source data for the project and for further preprocessing or future development. The current TF-IDF recommendation calculation is based on the prepared book-content text; it is not a collaborative-filtering model trained on `ratings.csv`.

**Original dataset:** [zygmuntz/goodbooks-10k on GitHub](https://github.com/zygmuntz/goodbooks-10k)

## 🏗️ Project Structure

```text
PageTurnner-Book-Recommendation-System/
│
├── app.py                         # Main Streamlit application
├── README.md                      # Project documentation
├── requirements.txt               # Python package dependencies
├── LICENSE                        # Project license
├── .gitignore                     # Files excluded from Git
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Loads raw dataset files
│   ├── preprocessing.py           # Cleans and prepares book data
│   └── recommender.py             # TF-IDF and similarity recommendation logic
│
├── data/
│   ├── raw/
│   │   ├── books.csv
│   │   ├── tags.csv
│   │   ├── book_tags.csv
│   │   ├── ratings.csv
│   │   └── to_read.csv
│   │
│   └── processed/
│       └── books_processed.csv    # Prepared dataset used by the app
│
└── models/
    └── README.md                  # Notes about the models directory
```

**Runtime-created file:** When books are saved to the reading shelf, the application creates `user_data/reading_list.json`. This folder is generated by the app and may not appear in a fresh Git clone until a book is saved.

## 🚀 Installation and Setup

Follow these steps to run PageTurnner on your own computer.

### Prerequisites

- Python installed on your computer
- Git installed on your computer
- An internet connection for cloning the repository and installing packages

### 1. Clone the repository

```bash
git clone https://github.com/TheDataSprint/PageTurnner-Book-Recommendation-System.git
```

### 2. Open the project folder

```bash
cd PageTurnner-Book-Recommendation-System
```

### 3. Create a virtual environment

A virtual environment keeps this project's Python packages separate from other projects.

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**Ubuntu / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install the required packages

Run this command after activating the virtual environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

### 6. Open PageTurnner in your browser

Streamlit normally starts the application at:

```text
http://localhost:8501
```

Open that address in your browser. The first startup may take a little longer while the book data and recommendation model are loaded.

> **Dataset setup:** The prepared dataset is already included at `data/processed/books_processed.csv`. You do not need to download or manually place the Goodbooks-10k files when following the steps above.

## 📚 How to Use

1. Open PageTurnner and select **Discover** from the sidebar.
2. Search for a book by title and choose the matching title from the results.
3. Select any available genre, minimum-rating, and sorting preferences.
4. Choose **Find my next read** to generate recommendations.
5. Review the recommended books and their content-match indicators.
6. Select **Save to shelf** on books you want to keep.
7. Open **My reading shelf** to view saved books. You can clear the shelf from that page.
8. Visit **Popular books** to browse popular titles using the available filters.

## ⚠️ Storage Notes and Current Limitations

- The reading shelf is currently stored in a local JSON file at `user_data/reading_list.json`; the project does not yet use user accounts or a database for reading lists.
- When running locally, the shelf is stored in the project's local folder. In the hosted demo, the app uses the hosting environment's local file storage, which should **not** be treated as permanent or private per-user storage. Saved items may not be retained after a hosting restart or redeployment.
- Recommendations are based on book content, not on a model trained to learn an individual reader's rating history.
- Results depend on the quality and completeness of the available book metadata and tags.

## 🔮 Future Improvements

The following are possible next steps and are not claims about features already implemented:

- Develop a hybrid recommendation engine that combines content similarity with collaborative filtering.
- Use rating data to support additional recommendation signals and evaluation.
- Add user accounts and database-backed, private reading shelves.
- Add richer search options, such as author and publication-year filters.
- Add recommendation evaluation and automated tests.
- Improve recommendation diversity and explain why each book was suggested.

## 🎓 Project Information

| Item | Details |
|---|---|
| Project | PageTurnner — Book Recommendation System |
| Domain | Machine Learning / Data Science |
| Programming Language | Python |
| Framework | Streamlit |
| Recommendation Technique | Content-Based Filtering |
| Text Representation | TF-IDF |
| Similarity Measure | Cosine Similarity |
| Dataset | Goodbooks-10k |
| Deployment | Streamlit Community Cloud |
| Academic Context | Bachelor of Technology — Computer Science and Engineering |

This project was developed as an academic project to apply data processing, machine learning, and web application development concepts.

## 📄 License and Attribution

The project repository includes a **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license. See the [`LICENSE`](LICENSE) file for details.

The Goodbooks-10k dataset is a third-party resource. Please refer to its [original repository](https://github.com/zygmuntz/goodbooks-10k) for dataset information and applicable terms. The project license does not automatically change the terms that apply to third-party materials.

---

<p align="center">
  Made with ❤️ and Python for book lovers.
  <br>
  <a href="https://github.com/TheDataSprint">TheDataSprint on GitHub</a>
</p>
