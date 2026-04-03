import os
import re
import uuid

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field

from embeddings import AVAILABLE_MODELS, create_embeddings_for_legal_dataset, detect_device


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
	file_path: str | None = None
	collection_name: str = "legal_knowledge_base"
	embedding_model_keys: list[str] = Field(default_factory=lambda: ["1", "2", "3"])
	max_vectors: int | None = None
	target_dimension: int | None = None
	chunking_strategy: str = "section-wise"
	vector_db: str = "both"
	dataset_name: str = "lex_glue"
	dataset_config: str = "eurlex"
	max_dataset_records: int | None = None


class QueryRequest(BaseModel):
	query: str
	collection_name: str = "legal_knowledge_base"
	top_k: int = 4
	llm_model_keys: list[str] = Field(default_factory=lambda: ["1", "2", "3"])
	search_mode: str = "hybrid"
	vector_db: str = "both"


def load_dotenv_if_available() -> None:
	try:
		from dotenv import load_dotenv

		project_root = os.path.dirname(os.path.abspath(__file__))
		load_dotenv(os.path.join(project_root, ".env"))
		load_dotenv()
	except Exception:
		project_root = os.path.dirname(os.path.abspath(__file__))
		candidate_paths = [
			".env",
			os.path.join(project_root, ".env"),
		]
		for env_path in candidate_paths:
			if not os.path.exists(env_path):
				continue
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


def get_pinecone_index(
	index_name: str | None = None,
	create_if_missing: bool = False,
	dimension: int = 768,
):
	api_key = os.getenv("PINECONE_API_KEY") or os.getenv("pinecone_API_key")
	if not api_key:
		raise HTTPException(status_code=500, detail="Missing PINECONE_API_KEY")

	resolved_index_name = index_name or os.getenv("PINECONE_INDEX", "quickstart")
	resolved_index_name = normalize_pinecone_index_name(resolved_index_name)
	pc = Pinecone(api_key=api_key)

	if create_if_missing:
		try:
			existing_names = [idx.name for idx in pc.list_indexes()]
		except Exception as exc:
			raise HTTPException(status_code=500, detail=f"Unable to list Pinecone indexes: {exc}")

		if resolved_index_name not in existing_names:
			cloud = os.getenv("PINECONE_CLOUD", "aws")
			region = os.getenv("PINECONE_REGION", "us-east-1")
			try:
				pc.create_index(
					name=resolved_index_name,
					dimension=dimension,
					metric="cosine",
					spec=ServerlessSpec(cloud=cloud, region=region),
				)
			except Exception as exc:
				raise HTTPException(status_code=500, detail=f"Unable to create Pinecone index '{resolved_index_name}': {exc}")

	return pc.Index(resolved_index_name)


def normalize_pinecone_index_name(name: str) -> str:
	normalized = re.sub(r"[^a-z0-9-]+", "-", name.strip().lower())
	normalized = re.sub(r"-+", "-", normalized).strip("-")
	if not normalized:
		return "quickstart"
	return normalized


def normalize_vector_db(value: str | None) -> str:
	if not value:
		return "both"
	value = value.strip().lower()
	return value if value in {"chroma", "pinecone", "both"} else "both"


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


def tokenize_keywords(text: str) -> list[str]:
	tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
	return [token for token in tokens if len(token) > 2]


def keyword_score(query: str, document: str, metadata: dict | None = None) -> float:
	query_tokens = tokenize_keywords(query)
	if not query_tokens:
		return 0.0

	doc_text = (document or "").lower()
	metadata_text = ""
	if metadata:
		metadata_text = " ".join(
			str(metadata.get(field, "")).lower() for field in ["act", "section", "court"]
		)

	score = 0.0
	for token in query_tokens:
		score += doc_text.count(token)
		score += 2.0 * metadata_text.count(token)
	return score


def retrieve_context_from_chroma(collection, query: str, top_k: int, search_mode: str) -> tuple[str, list[dict]]:
	model_names, target_dimension = get_collection_embedding_settings(collection)
	query_vector = embed_query(query, model_names, target_dimension)
	vector_result = collection.query(
		query_embeddings=[query_vector],
		n_results=max(top_k * 4, top_k),
		include=["documents", "metadatas", "distances"],
	)

	vector_docs = vector_result.get("documents", [[]])[0]
	vector_metadatas = vector_result.get("metadatas", [[]])[0]
	vector_distances = vector_result.get("distances", [[]])[0]

	def normalize_vector_score(distance: float | None) -> float:
		if distance is None:
			return 0.0
		return 1.0 / (1.0 + max(distance, 0.0))

	candidates: dict[str, dict] = {}
	for idx, doc in enumerate(vector_docs):
		if not doc:
			continue
		metadata = vector_metadatas[idx] if idx < len(vector_metadatas) and vector_metadatas[idx] else {}
		distance = vector_distances[idx] if idx < len(vector_distances) else None
		vector_score = normalize_vector_score(distance)
		kw_score = keyword_score(query, doc, metadata)
		hybrid_score = 0.7 * vector_score + 0.3 * kw_score
		candidates[doc] = {
			"text": doc,
			"metadata": metadata,
			"vector_score": vector_score,
			"keyword_score": kw_score,
			"hybrid_score": hybrid_score,
		}

	if search_mode == "hybrid":
		limit = min(max(collection.count(), top_k * 6), 1000)
		keyword_pool = collection.get(
			limit=limit,
			include=["documents", "metadatas"],
		)
		pool_docs = keyword_pool.get("documents", [])
		pool_metas = keyword_pool.get("metadatas", [])
		for idx, doc in enumerate(pool_docs):
			if not doc:
				continue
			metadata = pool_metas[idx] if idx < len(pool_metas) and pool_metas[idx] else {}
			kw_score = keyword_score(query, doc, metadata)
			if kw_score <= 0:
				continue
			existing = candidates.get(doc)
			vector_score = existing["vector_score"] if existing else 0.0
			hybrid_score = 0.7 * vector_score + 0.3 * kw_score
			candidates[doc] = {
				"text": doc,
				"metadata": metadata,
				"vector_score": vector_score,
				"keyword_score": kw_score,
				"hybrid_score": hybrid_score,
			}

	score_key = "hybrid_score" if search_mode == "hybrid" else "vector_score"
	ranked_chunks = sorted(
		candidates.values(),
		key=lambda item: item[score_key],
		reverse=True,
	)[:top_k]

	context = "\n\n".join(
		(
			f"Chunk {idx + 1} | Act: {chunk['metadata'].get('act', 'unknown')} | "
			f"Section: {chunk['metadata'].get('section', 'unknown')} | "
			f"Court: {chunk['metadata'].get('court', 'unknown')}\n"
			f"{chunk['text']}"
		)
		for idx, chunk in enumerate(ranked_chunks)
	)
	return context, ranked_chunks


def retrieve_context_from_pinecone(index, query: str, top_k: int, search_mode: str) -> tuple[str, list[dict]]:
	model_names = list(EMBEDDING_MODEL_OPTIONS.values())
	target_dimension = 1152
	try:
		stats = index.describe_index_stats()
		if isinstance(stats, dict):
			target_dimension = int(stats.get("dimension", target_dimension))
		else:
			target_dimension = int(getattr(stats, "dimension", target_dimension))
	except Exception:
		target_dimension = target_dimension

	query_vector = embed_query(query, model_names, target_dimension)

	query_result = index.query(
		vector=query_vector,
		top_k=max(top_k * 4, top_k),
		include_metadata=True,
	)

	matches = getattr(query_result, "matches", None)
	if matches is None and isinstance(query_result, dict):
		matches = query_result.get("matches", [])
	matches = matches or []

	candidates: dict[str, dict] = {}
	for match in matches:
		metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
		score = match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)
		doc = metadata.get("text", "") if metadata else ""
		if not doc:
			continue

		vector_score = max(float(score or 0.0), 0.0)
		kw_score = keyword_score(query, doc, metadata)
		hybrid_score = 0.7 * vector_score + 0.3 * kw_score
		candidates[doc] = {
			"text": doc,
			"metadata": metadata,
			"vector_score": vector_score,
			"keyword_score": kw_score,
			"hybrid_score": hybrid_score,
		}

	score_key = "hybrid_score" if search_mode == "hybrid" else "vector_score"
	ranked_chunks = sorted(
		candidates.values(),
		key=lambda item: item[score_key],
		reverse=True,
	)[:top_k]

	context = "\n\n".join(
		(
			f"Chunk {idx + 1} | Act: {chunk['metadata'].get('act', 'unknown')} | "
			f"Section: {chunk['metadata'].get('section', 'unknown')} | "
			f"Court: {chunk['metadata'].get('court', 'unknown')}\n"
			f"{chunk['text']}"
		)
		for idx, chunk in enumerate(ranked_chunks)
	)
	return context, ranked_chunks


def combine_ranked_chunks(*chunk_groups: list[dict], top_k: int) -> list[dict]:
	merged: dict[str, dict] = {}
	for group in chunk_groups:
		for chunk in group:
			text = chunk.get("text", "")
			if not text:
				continue
			existing = merged.get(text)
			if not existing or chunk.get("hybrid_score", 0.0) > existing.get("hybrid_score", 0.0):
				merged[text] = chunk

	ranked = sorted(
		merged.values(),
		key=lambda item: item.get("hybrid_score", item.get("vector_score", 0.0)),
		reverse=True,
	)
	return ranked[:top_k]


def build_context_from_chunks(ranked_chunks: list[dict]) -> str:
	return "\n\n".join(
		(
			f"Chunk {idx + 1} | Act: {chunk['metadata'].get('act', 'unknown')} | "
			f"Section: {chunk['metadata'].get('section', 'unknown')} | "
			f"Court: {chunk['metadata'].get('court', 'unknown')}\n"
			f"{chunk['text']}"
		)
		for idx, chunk in enumerate(ranked_chunks)
	)


def is_in_knowledge_base_scope(query: str, ranked_chunks: list[dict]) -> bool:
	if not ranked_chunks:
		return False

	max_vector_score = max(float(chunk.get("vector_score", 0.0) or 0.0) for chunk in ranked_chunks)
	max_keyword_score = max(float(chunk.get("keyword_score", 0.0) or 0.0) for chunk in ranked_chunks)

	if max_keyword_score > 0:
		return True

	return max_vector_score >= 0.35


def ask_with_models(
	client: OpenAI,
	query: str,
	context: str,
	selected_models: list[str],
) -> dict[str, str]:
	results: dict[str, str] = {}
	prompt = (
		"You are a legal knowledge-base assistant for the LexGLUE EURLEX dataset only. "
		"Use only the provided context. If the question is outside this legal dataset context, "
		"refuse and reply: 'Out of scope: I can only answer questions grounded in the LexGLUE EURLEX legal dataset.'\n\n"
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
		"vector_dbs": ["chroma", "pinecone", "both"],
		"dataset": {"name": "lex_glue", "config": "eurlex"},
	}


@app.post("/ingest")
def ingest_embeddings(payload: IngestRequest) -> dict:
	vector_db = normalize_vector_db(payload.vector_db)
	selected_embedding_models = resolve_embedding_models(payload.embedding_model_keys)

	result = create_embeddings_for_legal_dataset(
		model_names=selected_embedding_models,
		max_vectors=payload.max_vectors,
		target_dimension=payload.target_dimension,
		chunking_strategy=payload.chunking_strategy,
		dataset_name=payload.dataset_name,
		dataset_config=payload.dataset_config,
		max_records=payload.max_dataset_records,
	)

	if not result["vectors"]:
		raise HTTPException(status_code=400, detail="No vectors were generated from input document")

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
			"dataset": result["source_metadatas"][idx].get("dataset", f"{payload.dataset_name}/{payload.dataset_config}"),
			"dataset_split": result["source_metadatas"][idx].get("dataset_split", "unknown"),
			"dataset_record": result["source_metadatas"][idx].get("dataset_record", -1),
		}
		for idx in range(len(result["vectors"]))
	]

	backend_status = {
		"pinecone": "skipped",
		"chroma": "skipped",
	}
	errors: list[str] = []

	if vector_db in {"pinecone", "both"}:
		try:
			index = get_pinecone_index(
				payload.collection_name,
				create_if_missing=True,
				dimension=result["final_dimension"],
			)
			vectors = []
			for idx in range(len(result["vectors"])):
				text_preview = result["texts"][idx][:3000]
				metadata = {**metadatas[idx], "text": text_preview}
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
			backend_status["pinecone"] = "ok"
		except Exception as exc:
			backend_status["pinecone"] = "error"
			errors.append(f"pinecone: {exc}")
			if vector_db == "pinecone":
				raise HTTPException(status_code=500, detail=f"Pinecone ingest failed: {exc}")

	if vector_db in {"chroma", "both"}:
		try:
			client = get_chroma_client()
			collection = client.get_or_create_collection(name=payload.collection_name)
			# Chroma Cloud has a strict per-document byte quota on some plans.
			chroma_docs = [text[:12000] for text in result["texts"]]
			collection.add(
				ids=ids,
				embeddings=result["vectors"],
				documents=chroma_docs,
				metadatas=metadatas,
			)
			backend_status["chroma"] = "ok"
		except Exception as exc:
			backend_status["chroma"] = "error"
			errors.append(f"chroma: {exc}")
			if vector_db == "chroma":
				raise HTTPException(status_code=500, detail=f"Chroma ingest failed: {exc}")

	if backend_status["pinecone"] == "skipped" and backend_status["chroma"] == "skipped":
		raise HTTPException(status_code=400, detail="No vector backend selected for ingest")

	if backend_status["pinecone"] != "ok" and backend_status["chroma"] != "ok":
		raise HTTPException(status_code=500, detail="; ".join(errors) if errors else "Ingest failed")

	message = "Embeddings stored successfully"
	if errors:
		message = "Embeddings stored with partial success"

	return {
		"message": message,
		"collection_name": payload.collection_name,
		"vectors_stored": len(result["vectors"]),
		"final_dimension": result["final_dimension"],
		"models_used": result["model_names"],
		"chunking_strategy": payload.chunking_strategy,
		"vector_db": vector_db,
		"backend_status": backend_status,
		"errors": errors,
		"dataset": f"{payload.dataset_name}/{payload.dataset_config}",
	}


@app.post("/query")
def query_documents(payload: QueryRequest) -> dict:
	vector_db = normalize_vector_db(payload.vector_db)
	selected_llm_models = resolve_llm_models(payload.llm_model_keys)

	if vector_db == "pinecone":
		index = get_pinecone_index(payload.collection_name)
		context, ranked_chunks = retrieve_context_from_pinecone(
			index,
			payload.query,
			payload.top_k,
			payload.search_mode,
		)
	elif vector_db == "chroma":
		chroma_client = get_chroma_client()
		collection = chroma_client.get_or_create_collection(name=payload.collection_name)
		context, ranked_chunks = retrieve_context_from_chroma(
			collection,
			payload.query,
			payload.top_k,
			payload.search_mode,
		)
	else:
		pinecone_chunks: list[dict] = []
		chroma_chunks: list[dict] = []
		errors: list[str] = []

		try:
			index = get_pinecone_index(payload.collection_name)
			_, pinecone_chunks = retrieve_context_from_pinecone(
				index,
				payload.query,
				payload.top_k,
				payload.search_mode,
			)
		except Exception as exc:
			errors.append(f"pinecone: {exc}")

		try:
			chroma_client = get_chroma_client()
			collection = chroma_client.get_or_create_collection(name=payload.collection_name)
			_, chroma_chunks = retrieve_context_from_chroma(
				collection,
				payload.query,
				payload.top_k,
				payload.search_mode,
			)
		except Exception as exc:
			errors.append(f"chroma: {exc}")

		if not pinecone_chunks and not chroma_chunks and errors:
			raise HTTPException(status_code=500, detail="; ".join(errors))

		ranked_chunks = combine_ranked_chunks(pinecone_chunks, chroma_chunks, top_k=payload.top_k)
		context = build_context_from_chunks(ranked_chunks)

	if not ranked_chunks:
		return {
			"query": payload.query,
			"collection_name": payload.collection_name,
			"vector_db": vector_db,
			"retrieved_chunks": [],
			"answers": {},
			"message": "No matching documents found",
		}

	if not is_in_knowledge_base_scope(payload.query, ranked_chunks):
		guardrail_text = "Out of scope: I can only answer questions grounded in the LexGLUE EURLEX legal dataset."
		return {
			"query": payload.query,
			"collection_name": payload.collection_name,
			"vector_db": vector_db,
			"search_mode": payload.search_mode,
			"retrieved_chunks": ranked_chunks,
			"answers": {model: guardrail_text for model in selected_llm_models},
			"message": guardrail_text,
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
		"vector_db": vector_db,
		"search_mode": payload.search_mode,
		"retrieved_chunks": ranked_chunks,
		"answers": answers,
	}
