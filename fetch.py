from pymongo import MongoClient
import pandas as pd


# MongoDB connection string
MONGO_URI = "mongodb+srv://jainambsr_db_user:p6aphh0feh2jH5ae@cluster0.rnk5ywb.mongodb.net/"


# Database and collection names
DATABASE_NAME = "ANA"
COLLECTION_NAME = "ICD"


def fetch_data():
    
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)

        # Access database
        db = client[DATABASE_NAME]

        # Access collection
        collection = db[COLLECTION_NAME]

        # Fetch all documents
        data = list(
            collection.find(
                {},
                {
                    "_id": 0,
                    "CODE": 1,
                    "SHORT DESCRIPTION (VALID ICD-10 FY2026)": 1,
                    "LONG DESCRIPTION (VALID ICD-10 FY2026)": 1,
                    "NF EXCL": 1
                }
            )
        )

        if not data:
            print("No records found in MongoDB.")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        df = df.rename(columns={
            "CODE": "dx_code",
            "SHORT DESCRIPTION (VALID ICD-10 FY2026)": "short_desc",
            "LONG DESCRIPTION (VALID ICD-10 FY2026)": "long_desc",
            "NF EXCL": "nf_excl"
        })
        
        df = df.dropna(subset=["dx_code"])

        # Create combined text column
        df["text"] = (
            df["short_desc"].fillna("") +
            " " +
            df["long_desc"].fillna("")
        )

        return df

    except Exception as e:
        print(f"Error: {e}")
        return None

    finally:
        client.close()


if __name__ == "__main__":

    df = fetch_data()

    if df is not None:
        print(df.head())
        print("\nTotal records:", len(df))
