# file: vectorize_db.py

import os
import pandas as pd
import numpy as np
import faiss
from sqlalchemy import create_engine
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
database_url = os.getenv("DATABASE_URL")

# Setup OpenAI client
client = OpenAI(api_key=openai_api_key)

# Connect to the database
engine = create_engine(database_url)

# Choose the table and fetch data
table_name = "ProductionLinesData"
query = f"""
SELECT * FROM [HMIDatabse].[dbo].[{table_name}]
"""
df = pd.read_sql(query, engine)

# Convert rows into string format
def row_to_text(row):
    # Compose string description of production data row
    return (f"Id: {row['Id']}, Timestamp: {row['Timestamp']}, LineCode: {row['LineCode']}, "
            f"ShiftCode: {row['ShiftCode']}, Total: {row['Total']}, Good: {row['Good']}, "
            f"Waste: {row['Waste']}, Downtime: {row['Dowtime']}, JoggingSpeed: {row['JoggingSpeed']}")

texts = [row_to_text(row) for _, row in df.iterrows()]

# Generate embeddings using OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

print("Generating embeddings...")
embeddings = [get_embedding(text) for text in texts]

# Convert to NumPy array
embedding_dim = len(embeddings[0])
embedding_matrix = np.array(embeddings).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(embedding_matrix)

print(f"Indexed {len(texts)} rows.")

# Save index and data
faiss.write_index(index, "db_index.faiss")
pd.DataFrame({"text": texts}).to_csv("db_texts.csv", index=False)
