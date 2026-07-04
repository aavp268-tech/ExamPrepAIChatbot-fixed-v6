import os
import logging

from langchain_community.vectorstores import FAISS

VECTOR_DB_PATH = "vector_store"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def vector_db_exists():
    """
    Check if vector database exists.
    """

    exists = os.path.exists(VECTOR_DB_PATH)

    if exists:
        logging.info("Vector database exists.")
    else:
        logging.info("Vector database does not exist.")

    return exists


def create_vector_db(documents, embeddings):
    """
    Create FAISS vector database from documents and embeddings.
    """

    logging.info("Creating vector database...")

    vector_db = FAISS.from_documents(
        documents,
        embeddings
    )

    vector_db.save_local(
        VECTOR_DB_PATH
    )

    logging.info("Vector database saved successfully.")

    return vector_db


def load_vector_db(embeddings):
    """
    Load FAISS vector database from disk.
    """

    logging.info("Loading vector database...")

    vector_db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    logging.info("Vector database loaded successfully.")

    return vector_db
