import os
import uuid

import chromadb
from pinecone import Pinecone

from embeddings import (
  create_embeddings_for_pdf,
  choose_models_from_input,
  choose_target_dimension,
  choose_vector_count,
)


def load_dotenv_if_available() -> None:
  try:
    from dotenv import load_dotenv

    load_dotenv()
  except Exception:
    # Fallback parser for simple KEY=VALUE lines in .env.
    env_path = ".env"
    if not os.path.exists(env_path):
      return

    with open(env_path, "r", encoding="utf-8") as env_file:
      for line in env_file:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
          continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
          os.environ[key] = value


def build_client() -> chromadb.CloudClient:
  api_key = os.getenv("CHROMA_API_KEY") or os.getenv("Chroma_API_KEY")
  tenant = os.getenv("CHROMA_TENANT", "c5a875a5-6129-4f37-b387-3d0b7ae04f0d")
  database = os.getenv("CHROMA_DATABASE", "GenAI_Assignment")

  if not api_key:
    raise ValueError(
      "Missing Chroma API key. Set CHROMA_API_KEY (or Chroma_API_KEY) in your environment/.env."
    )

  return chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)


def build_pinecone_index(index_name: str):
  api_key = os.getenv("PINECONE_API_KEY") or os.getenv("pinecone_API_key")
  if not api_key:
    raise ValueError(
      "Missing Pinecone API key. Set PINECONE_API_KEY (or pinecone_API_key) in your environment/.env."
    )

  pc = Pinecone(api_key=api_key)
  return pc.Index(index_name)


def push_pdf_embeddings_to_chroma() -> None:
  vector_db = (input("Choose vector DB (chroma/pinecone, default: chroma): ").strip().lower() or "chroma")
  if vector_db not in {"chroma", "pinecone"}:
    vector_db = "chroma"

  file_path = input("Enter Your PDF Path: ").strip()
  print("Choose model(s): 1=all-mpnet-base-v2, 2=all-MiniLM-L6-v2, 3=legal-bert")
  model_choice = input("Enter model number(s), comma-separated (default: 1,2,3): ")
  selected_models = choose_models_from_input(model_choice)

  chunking_strategy = (
    input("Choose chunking strategy (generic/section-wise, default: section-wise): ").strip()
    or "section-wise"
  )

  vector_count_input = input(
    "Enter number of vectors/chunks to generate (default: all): "
  )
  max_vectors = choose_vector_count(vector_count_input)

  dimension_input = input(
    "Enter target embedding dimension (default: combined/original): "
  )
  target_dimension = choose_target_dimension(dimension_input)

  if vector_db == "pinecone":
    collection_name = (
      input("Enter Pinecone index name (default: quickstart): ").strip()
      or "quickstart"
    )
  else:
    collection_name = (
      input("Enter Chroma collection name (default: pdf_embeddings): ").strip()
      or "pdf_embeddings"
    )

  result = create_embeddings_for_pdf(
    file_path=file_path,
    model_names=selected_models,
    max_vectors=max_vectors,
    target_dimension=target_dimension,
    chunking_strategy=chunking_strategy,
  )

  if not result["vectors"]:
    print("No vectors generated. Nothing to store.")
    return

  ids = [f"{uuid.uuid4()}" for _ in result["vectors"]]
  metadatas = [
    {
      "chunk_index": idx,
      "combined_dimension": result["combined_dimension"],
      "final_dimension": result["final_dimension"],
      "models": ",".join(result["model_names"]),
      "act": result["chunk_metadatas"][idx]["act"],
      "section": result["chunk_metadatas"][idx]["section"],
      "court": result["chunk_metadatas"][idx]["court"],
    }
    for idx in range(len(result["vectors"]))
  ]

  if vector_db == "pinecone":
    index = build_pinecone_index(collection_name)
    vectors = []
    for idx in range(len(result["vectors"])):
      metadata = {**metadatas[idx], "text": result["texts"][idx]}
      vectors.append(
        {
          "id": ids[idx],
          "values": result["vectors"][idx],
          "metadata": metadata,
        }
      )

    batch_size = 100
    for start in range(0, len(vectors), batch_size):
      index.upsert(vectors=vectors[start : start + batch_size])
  else:
    client = build_client()
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(
      ids=ids,
      embeddings=result["vectors"],
      documents=result["texts"],
      metadatas=metadatas,
    )

  print(f"Stored {len(result['vectors'])} vectors in {vector_db} '{collection_name}'.")
  print(f"Final embedding dimension: {result['final_dimension']}")


if __name__ == "__main__":
  load_dotenv_if_available()
  push_pdf_embeddings_to_chroma()