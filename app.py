import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from typing import Optional

app = FastAPI()

# Model Loading
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Input Schema
class Questions(BaseModel):
    questions_with_ids: dict

@app.get("/")
async def root():
    return {"message": "Hello, this is the question clustering service."}

@app.post("/cluster-questions")
def cluster_questions(input_data: Questions):
    questions_with_ids = input_data.questions_with_ids
    questions = list(questions_with_ids.values())
    ids = list(questions_with_ids.keys())

    # Step 1: Convert questions to embeddings
    embeddings = model.encode(questions)

    # Step 2: Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Step 3: Cluster using similarity threshold
    threshold = 0.9
    threshold_clusters = {}
    visited = set()

    for i, question in enumerate(questions):
        if i in visited:
            continue
        threshold_clusters[i] = [(ids[i], question)]
        visited.add(i)
        for j in range(i + 1, len(questions)):
            if j not in visited and similarity_matrix[i, j] > threshold:
                threshold_clusters[i].append((ids[j], questions[j]))
                visited.add(j)

    # Prepare result
    result = {}
    for cluster_id, grouped_questions in threshold_clusters.items():
        result[f"Cluster {cluster_id}"] = [
            {"id": q_id, "question": q_text} for q_id, q_text in grouped_questions
        ]

    return {"clusters": result}

# Main entry point
if __name__ == "__main__":
    # Use PORT from environment variable, default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
