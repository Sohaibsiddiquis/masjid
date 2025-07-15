# file: ask_question.py

import os
import faiss
import pandas as pd
import numpy as np
import re
import calendar
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Load FAISS index and associated text data
index = faiss.read_index("db_index.faiss")
df = pd.read_csv("db_texts.csv")  # Assumes a 'text' column

# Helper: extract timestamp from a record string
def generate_all_dates_for_month(year: int, month: int):
    days_in_month = calendar.monthrange(year, month)[1]
    dates = [f"{year:04d}-{month:02d}-{day:02d}" for day in range(1, days_in_month+1)]
    return dates

def normalize_question(question):
    # First, ask GPT if the question refers to a specific month and year
    prompt = (
        "Identify if the question is about a whole month and year.\n"
        "If yes, respond with the year and month as 'YYYY-MM'. If not, respond 'NO'.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You detect month and year from questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0
    )
    answer = response.choices[0].message.content.strip()

    if answer != "NO":
        # Parse year and month from GPT's answer (expecting YYYY-MM)
        try:
            year, month = map(int, answer.split("-"))
            dates = generate_all_dates_for_month(year, month)
            # Build question listing all dates
            dates_str = ", ".join(dates)
            rewritten_question = f"Can you provide the production summary for the dates {dates_str}?"
            return rewritten_question
        except Exception:
            pass

    # Fallback: Use original normalize function (your existing GPT rewrite)
    prompt_full = (
        "Rewrite the following production data question to a standard format with explicit dates in YYYY-MM-DD format.\n"
        "Make sure the question is clear and ready to query the database. For example:\n"
        "- Input: 'compare production between 20 jully and 21 jully'\n"
        "- Output: 'Can you compare the production between 2025-07-20 and 2025-07-21?'\n\n"
        f"Question: {question}\n"
        "Rewritten question:"
    )
    response_full = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You rewrite user questions into a standard query format with ISO date strings."},
            {"role": "user", "content": prompt_full}
        ],
        max_tokens=100,
        temperature=0.2
    )
    rewritten = response_full.choices[0].message.content.strip()
    return rewritten
def extract_timestamp(text):
    match = re.search(r"[Tt]imestamp\s*:\s*(\d{4}[-/]\d{2}[-/]\d{2})", text)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d")
        except ValueError:
            try:
                return datetime.strptime(match.group(1), "%Y/%m/%d")
            except:
                return None
    return None

# Function to get embedding from OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Search top-k similar records, filter by recent days
def search_similar(question, k=5, days_filter=None):
    # Look for two dates in the question
    date_matches = re.findall(r"\b(20\d{2}[-/]\d{2}[-/]\d{2})\b", question)
    
    if len(date_matches) >= 2:
        dates = [datetime.strptime(d.replace('/', '-'), "%Y-%m-%d") for d in date_matches[:2]]
        df["parsed_timestamp"] = df["text"].apply(extract_timestamp)
        filtered = df[df["parsed_timestamp"].isin(dates)]

        if not filtered.empty:
            return filtered["text"].tolist()
        else:
            print(f"[Info] No matching rows for both dates {dates[0].date()} and {dates[1].date()}")
    
    # Fallback to semantic search
    q_embed = np.array([get_embedding(question)]).astype('float32')
    distances, indices = index.search(q_embed, k)
    results = df.iloc[indices[0]].copy()

    if days_filter:
        cutoff = datetime.now() - timedelta(days=days_filter)
        results["parsed_timestamp"] = results["text"].apply(extract_timestamp)
        results = results[results["parsed_timestamp"] >= cutoff]

    return results["text"].tolist()

# === Main conversational loop ===
if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Production Manager providing clear, data-driven answers "
                "to the Director about production line performance, waste, downtime, "
                "and other metrics. Use the provided production data context to give "
                "insightful, professional, and concise responses."
            )
        }
    ]

    print("Ask your production data questions. Type 'exit' to quit.\n")

    while True:
        q = input("Director: ")
        if q.strip().lower() == "exit":
            print("Production Manager: Goodbye!")
            break

        # Normalize question with LLM
        normalized_q = normalize_question(q)
        print(f"[Debug] Normalized Question: {normalized_q}")

        # Retrieve relevant records based on normalized question
        context_texts = search_similar(normalized_q, k=5, days_filter=30)
        context_text = "\n".join(context_texts)

        # Compose and send prompt
        user_message = f"Production data context:\n{context_text}\n\nQuestion: {normalized_q}"
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages
        )
        answer = response.choices[0].message.content

        print(f"\nProduction Manager: {answer}\n")
        messages.append({"role": "assistant", "content": answer})
