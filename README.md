# 🩺 ICD-10 Disease Prediction System

A Python-based **semantic disease and ICD-10 code prediction system** that analyzes symptoms or medical conversation text and retrieves the most relevant ICD-10 codes using **Sentence Transformers, spaCy phrase extraction, and cosine similarity**.

The system combines:

* Medical phrase-level semantic similarity
* Full conversation-level similarity
* ICD-10 disease descriptions
* Sentence Transformer embeddings
* Cosine similarity ranking

It returns the **top 5 unique ICD-10 codes** along with their disease descriptions and similarity scores.

---

## 📌 Project Overview

The system is designed to map free-text symptoms or medical conversations to relevant ICD-10 codes.

Instead of training a traditional classification model, the project uses a **semantic similarity approach**.

The ICD-10 records are converted into embeddings using:

```text
all-mpnet-base-v2
```

When a user enters a symptom description or conversation, the system:

1. Extracts meaningful phrases from the input.
2. Converts those phrases into embeddings.
3. Compares the phrase embeddings with the stored ICD-10 embeddings.
4. Encodes the complete input conversation.
5. Calculates full-conversation similarity.
6. Combines both similarity scores.
7. Sorts the ICD-10 records by their final score.
8. Removes duplicate ICD codes.
9. Returns the top 5 unique results.

---

# 🧠 System Architecture

```text
                 User Symptoms /
                Medical Conversation
                         │
                         ▼
              ┌─────────────────────┐
              │  Phrase Extraction  │
              │       spaCy         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Phrase Embeddings   │
              │ all-mpnet-base-v2   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Phrase Similarity   │
              │ Cosine Similarity   │
              └──────────┬──────────┘
                         │
                         │ 70%
                         │
Input Conversation ──────┤
                         │
                         ▼
              ┌─────────────────────┐
              │ Conversation        │
              │ Embedding           │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Conversation        │
              │ Similarity          │
              └──────────┬──────────┘
                         │
                         │ 30%
                         ▼
              ┌─────────────────────┐
              │ Final Similarity    │
              │ Score               │
              └──────────┬──────────┘
                         │
                         ▼
              Remove Duplicate Codes
                         │
                         ▼
                 Top 5 ICD-10 Codes
```

---

# 📂 Project Structure

```text
ICD-10-Disease-Prediction-System/
│
├── fetch.py
├── extract.py
├── create_embeddings.py
├── predict.py
│
├── Valid_icd10_october2025_0.csv
├── dx_codes.pkl
├── README.md
│
└── __pycache__/
```

The repository currently contains these implementation files and data/model artifacts.

---

# 🗄️ Data Retrieval

Data retrieval is implemented in:

```text
fetch.py
```

The project connects to MongoDB using `PyMongo`.

The configured database is:

```text
Database: ANA
Collection: ICD
```

The system retrieves these fields from MongoDB:

```text
CODE
SHORT DESCRIPTION (VALID ICD-10 FY2026)
LONG DESCRIPTION (VALID ICD-10 FY2026)
NF EXCL
```

These fields are renamed internally to:

```text
dx_code
short_desc
long_desc
nf_excl
```

The `_id` field is excluded from the MongoDB query.

---

# 🧹 Data Preparation

After retrieving the ICD-10 records, the implementation removes records where `dx_code` is missing.

It then creates a combined text field:

```python
df["text"] = (
    df["short_desc"].fillna("")
    + " "
    + df["long_desc"].fillna("")
)
```

Therefore, the semantic embedding input is constructed from:

```text
Short Description + Long Description
```

The `nf_excl` field is retrieved but is not included in the combined embedding text.

---

# 🔤 Phrase Extraction

Phrase extraction is implemented in:

```text
extract.py
```

The project uses:

```python
spacy.load("en_core_web_sm")
```

The function:

```python
extract_phrases(paragraph)
```

extracts three types of text structures.

## 1. Noun Chunks

The system extracts noun chunks containing at least two words.

For example, a phrase such as:

```text
shortness of breath
```

can be identified as a multi-word noun phrase.

The extracted phrase is converted to lowercase.

---

## 2. Adjective + Noun Phrases

The system also identifies an adjective followed by a noun.

The implementation checks:

```text
ADJ → NOUN
```

and combines the two tokens into a phrase.

---

## 3. Sentences

The system extracts complete sentences containing at least two words.

All extracted phrases are converted to lowercase and duplicate phrases are removed.

The final result preserves the original order while removing duplicates using:

```python
list(dict.fromkeys(phrases))
```

---

# 🧠 Sentence Transformer Embeddings

The project uses:

```text
all-mpnet-base-v2
```

from the `sentence-transformers` library.

The model is initialized in both:

```text
create_embeddings.py
predict.py
```

---

# 🏗️ Embedding Creation

Embedding generation is implemented in:

```text
create_embeddings.py
```

The process is:

```text
MongoDB
   ↓
Fetch ICD-10 Records
   ↓
Create Combined Text
   ↓
Sentence Transformer
   ↓
Generate Embeddings
   ↓
Save NumPy Array
   ↓
Save ICD Codes
```

The implementation calls:

```python
model.encode(
    texts,
    batch_size=100,
    show_progress_bar=True,
    convert_to_numpy=True
)
```

The resulting embeddings are saved to:

```text
disease_embeddings.npy
```

The corresponding ICD codes are saved to:

```text
dx_codes.pkl
```

---

# 💾 Saved Artifacts

## `disease_embeddings.npy`

Contains the generated Sentence Transformer embeddings for the ICD-10 descriptions.

## `dx_codes.pkl`

Contains the ICD-10 codes corresponding to the embeddings.

The ordering is important because the prediction system uses the same index to associate an embedding with its ICD code.

---

# 🔎 Prediction Process

Prediction is implemented in:

```text
predict.py
```

The main function is:

```python
predict_top_5(paragraph)
```

It accepts a text paragraph containing symptoms or a medical conversation.

---

# 1️⃣ Extract Medical Phrases

The input paragraph is passed to:

```python
extract_phrases(paragraph)
```

If no phrases are extracted, the complete paragraph itself is used as the query.

---

# 2️⃣ Generate Phrase Embeddings

The extracted phrases are encoded using:

```python
model.encode(
    phrases,
    convert_to_numpy=True,
    normalize_embeddings=True
)
```

The embeddings are normalized before similarity calculation.

---

# 3️⃣ Phrase-Level Similarity

Cosine similarity is calculated between the phrase embeddings and the stored disease embeddings:

```python
phrase_scores = cosine_similarity(
    phrase_embeddings,
    disease_embeddings
)
```

For every ICD-10 record, the system selects the highest similarity score among all extracted phrases:

```python
semantic_scores = np.max(
    phrase_scores,
    axis=0
)
```

This produces one semantic similarity score for each ICD-10 record.

---

# 4️⃣ Full Conversation Similarity

The complete input paragraph is also encoded separately:

```python
query_embedding = model.encode(
    [paragraph],
    convert_to_numpy=True,
    normalize_embeddings=True
)
```

The complete conversation embedding is then compared with every stored ICD-10 embedding.

```python
conversation_scores = cosine_similarity(
    query_embedding,
    disease_embeddings
)[0]
```

---

# 5️⃣ Score Combination

The system combines the two similarity measurements.

The implemented formula is:

```text
Final Score =
    70% × Phrase Similarity
  + 30% × Conversation Similarity
```

In code:

```python
final_scores = (
    0.70 * semantic_scores
    +
    0.30 * conversation_scores
)
```

Therefore, the phrase-level semantic similarity has greater influence on the final ranking.

---

# 6️⃣ Ranking

The final scores are sorted in descending order:

```python
sorted_indices = np.argsort(
    final_scores
)[::-1]
```

The highest-scoring ICD-10 records are considered first.

---

# 7️⃣ Duplicate ICD Code Removal

The prediction system maintains a set:

```python
seen_codes = set()
```

If an ICD-10 code has already been added to the results, it is skipped.

This ensures that the returned results contain **unique ICD codes**.

---

# 8️⃣ Top 5 Results

The system returns a maximum of **5 unique predictions**.

Each result contains:

```python
{
    "dx_code": code,
    "disease": df.iloc[index]["short_desc"],
    "similarity": ...
}
```

Therefore, every prediction contains:

| Field        | Description                            |
| ------------ | -------------------------------------- |
| `dx_code`    | ICD-10 code                            |
| `disease`    | ICD-10 short description               |
| `similarity` | Final similarity score as a percentage |

The similarity score is rounded to two decimal places.

---

# 📊 Example Output Format

The command-line interface displays results in the following format:

```text
1. DX Code: XXXXX | Disease: XXXXX | Similarity: XX.XX%
2. DX Code: XXXXX | Disease: XXXXX | Similarity: XX.XX%
3. DX Code: XXXXX | Disease: XXXXX | Similarity: XX.XX%
4. DX Code: XXXXX | Disease: XXXXX | Similarity: XX.XX%
5. DX Code: XXXXX | Disease: XXXXX | Similarity: XX.XX%
```

The actual codes and descriptions depend on the input and the ICD-10 data stored in MongoDB.

---

# 🗃️ ICD-10 Dataset

The repository also contains:

```text
Valid_icd10_october2025_0.csv
```

This is an ICD-10 data file included in the repository.

However, the current Python prediction pipeline retrieves its active ICD data through the MongoDB `ANA.ICD` collection rather than directly loading this CSV file in `fetch.py`.

Therefore, the CSV should not be described as the direct runtime data source for the current prediction pipeline.

---

# 🛠️ Technologies Used

| Technology            | Purpose                                    |
| --------------------- | ------------------------------------------ |
| Python                | Core implementation                        |
| MongoDB               | ICD-10 data storage/retrieval              |
| PyMongo               | MongoDB connection                         |
| Pandas                | ICD-10 data processing                     |
| NumPy                 | Embedding storage and numerical operations |
| spaCy                 | Medical phrase extraction                  |
| `en_core_web_sm`      | spaCy language model                       |
| Sentence Transformers | Semantic embeddings                        |
| `all-mpnet-base-v2`   | Embedding model                            |
| Scikit-learn          | Cosine similarity                          |
| Pickle                | ICD-code serialization                     |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/HacBoy777/ICD-10-Disease-Prediction-System.git
```

Navigate to the project:

```bash
cd ICD-10-Disease-Prediction-System
```

Install the Python dependencies:

```bash
pip install pandas numpy pymongo sentence-transformers scikit-learn spacy
```

Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

# 🗄️ MongoDB Configuration

The current implementation expects a MongoDB database and collection configured as:

```text
Database:
ANA

Collection:
ICD
```

The MongoDB collection should provide the fields used by `fetch.py`:

```text
CODE
SHORT DESCRIPTION (VALID ICD-10 FY2026)
LONG DESCRIPTION (VALID ICD-10 FY2026)
NF EXCL
```

The MongoDB URI is configured in:

```text
fetch.py
```

For a production or public deployment, credentials should be supplied through environment variables rather than committed directly to source code.

---

# 🏗️ Generate Embeddings

Before running prediction, create the disease embeddings:

```bash
python create_embeddings.py
```

This performs:

```text
Fetch ICD-10 Data
      ↓
Combine Short + Long Descriptions
      ↓
Generate Sentence Embeddings
      ↓
disease_embeddings.npy
      +
dx_codes.pkl
```

---

# ▶️ Run Prediction

Once the embedding files have been generated, run:

```bash
python predict.py
```

The program prompts:

```text
Enter symptoms/conversation:
```

Enter a symptom description or medical conversation.

For example:

```text
Patient reports difficulty breathing and respiratory discomfort.
```

The system then returns the highest-ranked ICD-10 matches based on semantic similarity.

---

# 🔄 Complete Pipeline

```text
                 ICD-10 Data
                     │
                     ▼
                 MongoDB
                     │
                     ▼
                fetch.py
                     │
                     ▼
             Data Preparation
                     │
                     ▼
        Short Description + Long Description
                     │
                     ▼
          Sentence Transformer
            all-mpnet-base-v2
                     │
                     ▼
             Disease Embeddings
                     │
             ┌───────┴───────┐
             ▼               ▼
 disease_embeddings.npy   dx_codes.pkl
             │
             └───────┬───────┘
                     │
                     ▼
               User Input
                     │
                     ▼
                extract.py
                     │
                     ▼
              Medical Phrases
                     │
                     ▼
          Phrase Embeddings
                     │
                     ▼
          Phrase Similarity
                  70%
                     │
                     ├──────────────┐
                     │              │
                     ▼              ▼
              Full Paragraph    Conversation
                Embedding        Similarity
                                  30%
                     │              │
                     └──────┬───────┘
                            ▼
                     Final Score
                            │
                            ▼
                  Sort + Remove Duplicates
                            │
                            ▼
                       Top 5 ICD Codes
```

---

# 📌 Important Implementation Details

### Semantic Retrieval Rather Than Traditional Classification

The project does not train a conventional supervised classifier such as:

```text
Random Forest
Logistic Regression
SVM
KNN
```

Instead, it uses pretrained Sentence Transformer embeddings and similarity ranking.

### Two-Level Similarity

The prediction combines:

```text
Phrase-level similarity → 70%
Full conversation similarity → 30%
```

This allows both extracted phrases and the complete input context to influence the final ranking.

### Unique ICD Codes

Duplicate ICD codes are explicitly removed before the final five results are returned.

---

# ⚠️ Current Scope and Limitations

The current implementation should be understood as a **semantic ICD-10 retrieval/prediction system**, not an official medical coding or diagnostic system.

It currently:

* Maps free text to ICD-10 codes using semantic similarity.
* Uses ICD-10 descriptions as the embedding corpus.
* Returns similarity-based rankings.
* Does not provide a clinically validated diagnosis.
* Does not connect to a medical decision-support workflow.
* Does not use a supervised disease-classification model.
* Does not perform official ICD-10 code validation beyond matching against the retrieved dataset.
* Does not use the `NF EXCL` field when creating embeddings.

The returned similarity percentage represents the system's calculated similarity score; it should **not be interpreted as medical probability, diagnostic confidence, or disease prevalence**.

---

# 🔐 Data & Privacy

Medical symptoms and conversations may contain sensitive information.

For real-world use:

* Do not commit real patient information to GitHub.
* Avoid storing identifiable clinical conversations in logs.
* Protect MongoDB credentials.
* Use environment variables for secrets.
* Restrict access to medical datasets.
* Use anonymized or synthetic data during development.

This repository should therefore be treated as a **technical/educational implementation**, not a production clinical system.

---

# 🔮 Possible Future Improvements

Possible extensions to the current implementation include:

* Add a dedicated configuration file or environment variables for MongoDB.
* Add proper ICD-10 version selection.
* Add confidence/similarity thresholding.
* Evaluate predictions against a labeled test set.
* Add precision, recall, and F1-score evaluation.
* Add batch prediction support.
* Add caching for embeddings.
* Add API or web-interface support.
* Add structured clinical-text preprocessing.
* Improve phrase extraction for medical terminology.
* Add explanation of why a particular ICD-10 code was ranked highly.
* Add automated tests.
* Remove generated `__pycache__` files from the repository.
* Add `requirements.txt`.

---

# 🎯 Learning Outcomes

This project demonstrates practical experience with:

* Natural Language Processing
* Semantic text similarity
* Sentence Transformers
* Text embeddings
* spaCy NLP pipelines
* Noun phrase extraction
* Cosine similarity
* Vector-based information retrieval
* MongoDB integration
* Pandas data processing
* NumPy arrays
* Pickle serialization
* Ranking and deduplication
* ICD-10 data processing

---

# 👨‍💻 Author

**HacBoy777**

GitHub:

https://github.com/HacBoy777

---

# ⭐ Repository

[ICD-10 Disease Prediction System](https://github.com/HacBoy777/ICD-10-Disease-Prediction-System)

If you find this project useful, consider giving the repository a ⭐.

> **From clinical text to semantic ICD-10 matching using NLP and embeddings. 🧠**
