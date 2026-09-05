import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from fetch import fetch_data

# Load the Sentence Transformer model
model = SentenceTransformer("all-mpnet-base-v2")

def create_embeddings():
    print("Fetching data from MongoDB...")

    # Fetch data from MongoDB
    df = fetch_data()

    if df is None or df.empty:
        print("No data found.")
        return
    print(f"Total records: {len(df)}")

    # Get the combined text column
    texts = df["text"].tolist()
    print(f"Total texts to embed: {len(texts)}")

    print("Generating embeddings...")
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=100,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print("Saving embeddings...")

    # Save embeddings
    np.save(
        "disease_embeddings.npy",
        embeddings
    )

    # Save dx codes
    with open("dx_codes.pkl", "wb") as f:
        pickle.dump(
            df["dx_code"].tolist(),
            f
        )

    print("\nEmbeddings created successfully!")
    print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    create_embeddings()