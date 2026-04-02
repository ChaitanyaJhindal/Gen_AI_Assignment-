import os
import uuid

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from pydantic import BaseModel, Field

from embeddings import AVAILABLE_MODELS, create_embeddings_for_pdf, detect_device


LLM_MODELS = {
	"1": "openai/gpt-oss-120b",
	"2": "llama-3.3-70b-versatile",
	"3": "llama-3.1-8b-instant",
}

EMBEDDING_MODEL_OPTIONS = AVAILABLE_MODELS
embedding_model_cache: dict[str, HuggingFaceEmbeddings] = {}

app = FastAPI(title="GenAI Assignment RAG API", version="1.0.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class IngestRequest(BaseModel):
	file_path: str
	collection_name: str = "pdf_embeddings"
	embedding_model_keys: list[str] = Field(default_factory=lambda: ["1", "2"])
	max_vectors: int | None = None
	target_dimension: int | None = None


class QueryRequest(BaseModel):
	query: str
	collection_name: str = "pdf_embeddings"
	top_k: int = 4
	llm_model_keys: list[str] = Field(default_factory=lambda: ["1", "2", "3"])


def load_dotenv_if_available() -> None:
	try:
		from dotenv import load_dotenv

		load_dotenv()
	except Exception:
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


def get_llm_client() -> OpenAI:
	api_key = os.environ.get("GROQ_API_KEY")
	if not api_key:
		raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")
	return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def get_chroma_client() -> chromadb.CloudClient:
	api_key = os.getenv("CHROMA_API_KEY") or os.getenv("Chroma_API_KEY")
	tenant = os.getenv("CHROMA_TENANT", "c5a875a5-6129-4f37-b387-3d0b7ae04f0d")
	database = os.getenv("CHROMA_DATABASE", "GenAI_Assignment")
	if not api_key:
		raise HTTPException(status_code=500, detail="Missing CHROMA_API_KEY")
	return chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)


def resolve_embedding_models(keys: list[str]) -> list[str]:
	models = [EMBEDDING_MODEL_OPTIONS[k] for k in keys if k in EMBEDDING_MODEL_OPTIONS]
	return models if models else list(EMBEDDING_MODEL_OPTIONS.values())


def resolve_llm_models(keys: list[str]) -> list[str]:
	models = [LLM_MODELS[k] for k in keys if k in LLM_MODELS]
	return models if models else list(LLM_MODELS.values())


def resize_vector(vector: list[float], target_dimension: int) -> list[float]:
	if target_dimension <= len(vector):
		return vector[:target_dimension]
	return vector + [0.0] * (target_dimension - len(vector))


def get_collection_embedding_settings(collection) -> tuple[list[str], int]:
	peek = collection.peek(limit=1)
	metadatas = peek.get("metadatas", []) if peek else []
	if metadatas and metadatas[0]:
		metadata = metadatas[0]
		models_raw = metadata.get("models", "")
		model_names = [m.strip() for m in models_raw.split(",") if m.strip()]
		final_dimension = int(metadata.get("final_dimension", 0))
		if model_names and final_dimension > 0:
			return model_names, final_dimension

	default_models = list(EMBEDDING_MODEL_OPTIONS.values())
	return default_models, 1152


def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
	if model_name not in embedding_model_cache:
		embedding_model_cache[model_name] = HuggingFaceEmbeddings(
			model_name=model_name,
			model_kwargs={"device": detect_device()},
			encode_kwargs={"normalize_embeddings": False},
		)
	return embedding_model_cache[model_name]


def embed_query(query: str, model_names: list[str], target_dimension: int) -> list[float]:
	merged: list[float] = []
	for model_name in model_names:
		hf = get_embedding_model(model_name)
		merged.extend(hf.embed_query(query))
	return resize_vector(merged, target_dimension)


def retrieve_context(collection, query: str, top_k: int) -> tuple[str, list[str]]:
	model_names, target_dimension = get_collection_embedding_settings(collection)
	query_vector = embed_query(query, model_names, target_dimension)
	result = collection.query(
		query_embeddings=[query_vector],
		n_results=top_k,
		include=["documents", "metadatas", "distances"],
	)
	documents = result.get("documents", [[]])[0]
	context = "\n\n".join(f"Chunk {idx + 1}: {doc}" for idx, doc in enumerate(documents))
	return context, documents


def ask_with_models(
	client: OpenAI,
	query: str,
	context: str,
	selected_models: list[str],
) -> dict[str, str]:
	results: dict[str, str] = {}
	prompt = (
		"You are a document question-answering assistant. Use only the provided context. "
		"If the answer is not in context, clearly say so.\n\n"
		f"Context:\n{context}\n\n"
		f"Question: {query}"
	)

	for model in selected_models:
		try:
			response = client.responses.create(input=prompt, model=model)
			results[model] = response.output_text
		except Exception as exc:
			results[model] = f"Error: {exc}"
	return results


@app.on_event("startup")
def startup_event() -> None:
	load_dotenv_if_available()


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.get("/models")
def model_options() -> dict:
	return {
		"llm_models": LLM_MODELS,
		"embedding_models": EMBEDDING_MODEL_OPTIONS,
	}


@app.post("/ingest")
def ingest_embeddings(payload: IngestRequest) -> dict:
	selected_embedding_models = resolve_embedding_models(payload.embedding_model_keys)

	result = create_embeddings_for_pdf(
		file_path=payload.file_path,
		model_names=selected_embedding_models,
		max_vectors=payload.max_vectors,
		target_dimension=payload.target_dimension,
	)

	if not result["vectors"]:
		raise HTTPException(status_code=400, detail="No vectors were generated from input document")

	client = get_chroma_client()
	collection = client.get_or_create_collection(name=payload.collection_name)

	ids = [f"{uuid.uuid4()}" for _ in result["vectors"]]
	metadatas = [
		{
			"chunk_index": idx,
			"combined_dimension": result["combined_dimension"],
			"final_dimension": result["final_dimension"],
			"models": ",".join(result["model_names"]),
		}
		for idx in range(len(result["vectors"]))
	]

	collection.add(
		ids=ids,
		embeddings=result["vectors"],
		documents=result["texts"],
		metadatas=metadatas,
	)

	return {
		"message": "Embeddings stored successfully",
		"collection_name": payload.collection_name,
		"vectors_stored": len(result["vectors"]),
		"final_dimension": result["final_dimension"],
		"models_used": result["model_names"],
	}


@app.post("/query")
def query_documents(payload: QueryRequest) -> dict:
	selected_llm_models = resolve_llm_models(payload.llm_model_keys)

	chroma_client = get_chroma_client()
	collection = chroma_client.get_or_create_collection(name=payload.collection_name)

	context, docs = retrieve_context(collection, payload.query, payload.top_k)
	if not docs:
		return {
			"query": payload.query,
			"collection_name": payload.collection_name,
			"retrieved_chunks": [],
			"answers": {},
			"message": "No matching documents found",
		}

	llm_client = get_llm_client()
	answers = ask_with_models(
		llm_client,
		payload.query,
		context,
		selected_llm_models,
	)

	return {
		"query": payload.query,
		"collection_name": payload.collection_name,
		"retrieved_chunks": docs,
		"answers": answers,
	}
