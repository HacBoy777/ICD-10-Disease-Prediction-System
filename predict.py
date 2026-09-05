import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from fetch import fetch_data
from extract import extract_phrases


model = SentenceTransformer("all-mpnet-base-v2")

disease_embeddings = np.load(
    "disease_embeddings.npy"
)

with open("dx_codes.pkl", "rb") as f:
    dx_codes = pickle.load(f)

df = fetch_data()

if df is None:
    raise Exception("Unable to fetch ICD data from MongoDB.")


def predict_top_5(paragraph):

    # Extract medical phrases internally
    phrases = extract_phrases(paragraph)

    if not phrases:
        phrases = [paragraph]

    # Encode phrases
    phrase_embeddings = model.encode(
        phrases,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Phrase-level similarity
    phrase_scores = cosine_similarity(
        phrase_embeddings,
        disease_embeddings
    )

    # Best score for every disease
    semantic_scores = np.max(
        phrase_scores,
        axis=0
    )

    # Full conversation similarity
    query_embedding = model.encode(
        [paragraph],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    conversation_scores = cosine_similarity(
        query_embedding,
        disease_embeddings
    )[0]

    # Combine semantic + conversation scores
    final_scores = (
        0.70 * semantic_scores
        +
        0.30 * conversation_scores
    )

    # Sort diseases
    sorted_indices = np.argsort(
        final_scores
    )[::-1]

    results = []
    seen_codes = set()

    for index in sorted_indices:

        code = dx_codes[index]

        if code in seen_codes:
            continue

        seen_codes.add(code)

        results.append({
            "dx_code": code,
            "disease": df.iloc[index]["short_desc"],
            "similarity": round(
                float(final_scores[index]) * 100,
                2
            )
        })

        if len(results) == 5:
            break

    return results


if __name__ == "__main__":

    paragraph = input(
        "Enter symptoms/conversation: "
    )

    predictions = predict_top_5(
        paragraph
    )

    print("\nTop 20 Predictions:\n")

    for i, prediction in enumerate(
        predictions,
        start=1
    ):

        print(
            f"{i}. "
            f"DX Code: {prediction['dx_code']} | "
            f"Disease: {prediction['disease']} | "
            f"Similarity: {prediction['similarity']}%"
        )