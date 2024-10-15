import os
import csv
from datetime import datetime
from constants import EMBEDDING_MODEL_NAME
from langchain.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

def log_to_csv(question, answer):
    log_dir, log_file = "local_chat_history", "qa_log.csv"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file)
    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "question", "answer"])
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer])

def get_embeddings(device_type="cpu"):
    model_kwargs = {"device": device_type}

    if "instructor" in EMBEDDING_MODEL_NAME:
        # Directly load the INSTRUCTOR model using SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device_type)
    elif "bge" in EMBEDDING_MODEL_NAME:
        # For BGE models
        return HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs
        )
    else:
        # For general HuggingFace models
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs
        )
