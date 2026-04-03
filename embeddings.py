import json
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from document_ingestion import extract_legal_metadata, load_and_split_pdf


AVAILABLE_MODELS = {
    "1": "sentence-transformers/all-mpnet-base-v2",
    "2": "sentence-transformers/all-MiniLM-L6-v2",
    "3": "nlpaueb/legal-bert-base-uncased",
}


def detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _create_embeddings_from_texts(
    texts: list[str],
    model_names: list[str] | None = None,
    max_vectors: int | None = None,
    target_dimension: int | None = None,
    chunking_strategy: str = "generic",
    source_metadatas: list[dict] | None = None,
) -> dict:
    chunk_metadatas = [extract_legal_metadata(text) for text in texts]
    source_metadatas = source_metadatas or [{} for _ in texts]

    if max_vectors is not None and max_vectors > 0:
        texts = texts[:max_vectors]
        chunk_metadatas = chunk_metadatas[:max_vectors]
        source_metadatas = source_metadatas[:max_vectors]

    if not texts:
        return {
            "texts": [],
            "chunk_metadatas": [],
            "source_metadatas": [],
            "model_names": [],
            "dimensions": {},
            "combined_dimension": 0,
            "vectors": [],
        }

    if not model_names:
        model_names = list(AVAILABLE_MODELS.values())

    model_kwargs = {"device": detect_device()}
    encode_kwargs = {"normalize_embeddings": False}

    per_model_vectors = []
    dimensions = {}
    for model_name in model_names:
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        vectors = hf.embed_documents(texts)
        per_model_vectors.append(vectors)
        dimensions[model_name] = len(vectors[0]) if vectors else 0

    combined_vectors = []
    for idx in range(len(texts)):
        merged = []
        for model_vectors in per_model_vectors:
            merged.extend(model_vectors[idx])
        combined_vectors.append(merged)

    combined_dimension = len(combined_vectors[0]) if combined_vectors else 0

    if target_dimension is not None and target_dimension > 0:
        resized_vectors = []
        for vector in combined_vectors:
            if target_dimension <= len(vector):
                resized_vectors.append(vector[:target_dimension])
            else:
                resized_vectors.append(vector + [0.0] * (target_dimension - len(vector)))
        final_vectors = resized_vectors
    else:
        final_vectors = combined_vectors

    final_dimension = len(final_vectors[0]) if final_vectors else 0

    return {
        "texts": texts,
        "chunk_metadatas": chunk_metadatas,
        "source_metadatas": source_metadatas,
        "chunking_strategy": chunking_strategy,
        "model_names": model_names,
        "dimensions": dimensions,
        "combined_dimension": combined_dimension,
        "final_dimension": final_dimension,
        "vectors": final_vectors,
    }


def create_embeddings_for_pdf(
    file_path: str,
    model_names: list[str] | None = None,
    max_vectors: int | None = None,
    target_dimension: int | None = None,
    chunking_strategy: str = "generic",
) -> dict:
    texts = load_and_split_pdf(file_path, chunking_strategy=chunking_strategy)
    return _create_embeddings_from_texts(
        texts=texts,
        model_names=model_names,
        max_vectors=max_vectors,
        target_dimension=target_dimension,
        chunking_strategy=chunking_strategy,
    )


def create_embeddings_for_legal_dataset(
    model_names: list[str] | None = None,
    max_vectors: int | None = None,
    target_dimension: int | None = None,
    chunking_strategy: str = "section-wise",
    dataset_name: str = "lex_glue",
    dataset_config: str = "eurlex",
    max_records: int | None = None,
) -> dict:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=100,
        chunk_overlap=0,
        separators=[
            "\nSection ",
            "\nSECTION ",
            "\nSec. ",
            "\nArticle ",
            "\nARTICLE ",
            "\nArt. ",
            "\nChapter ",
            "\nCHAPTER ",
            "\nClause ",
            "\nCLAUSE ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    texts: list[str] = []
    source_metadatas: list[dict] = []

    processed_records = 0
    for split_name in dataset.keys():
        split_rows = dataset[split_name]
        for row_idx, row in enumerate(split_rows):
            text_body = str(row.get("text", "") or "")
            if not text_body.strip():
                continue

            # Encourage cleaner legal boundaries before token chunking.
            text_body = re.sub(
                r"(?im)^\s*(section|sec\.?|article|art\.?|chapter|clause)\s+",
                r"\n\1 ",
                text_body,
            )

            title = str(row.get("title", "") or "")
            labels = row.get("labels", [])
            labels_text = ",".join(str(label) for label in labels) if isinstance(labels, list) else str(labels)
            record_text = (
                f"Title: {title}\n\n{text_body}\n\nLegal Labels: {labels_text}"
                if title
                else f"{text_body}\n\nLegal Labels: {labels_text}"
            )

            chunks = splitter.split_text(record_text)
            if not chunks:
                chunks = [record_text]

            for chunk in chunks:
                texts.append(chunk)
                source_metadatas.append(
                    {
                        "dataset": f"{dataset_name}/{dataset_config}",
                        "dataset_split": split_name,
                        "dataset_record": row_idx,
                    }
                )

            processed_records += 1
            if max_records is not None and max_records > 0 and processed_records >= max_records:
                break

        if max_records is not None and max_records > 0 and processed_records >= max_records:
            break

    return _create_embeddings_from_texts(
        texts=texts,
        model_names=model_names,
        max_vectors=max_vectors,
        target_dimension=target_dimension,
        chunking_strategy=chunking_strategy,
        source_metadatas=source_metadatas,
    )


def choose_models_from_input(raw_choice: str) -> list[str]:
    keys = [token.strip() for token in raw_choice.split(",") if token.strip()]
    selected = [AVAILABLE_MODELS[k] for k in keys if k in AVAILABLE_MODELS]
    return selected if selected else list(AVAILABLE_MODELS.values())


def choose_vector_count(raw_value: str) -> int | None:
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
        return value if value > 0 else None
    except ValueError:
        return None


def choose_target_dimension(raw_value: str) -> int | None:
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
        return value if value > 0 else None
    except ValueError:
        return None


if __name__ == "__main__":
    file_path = input("Enter Your PDF Path: ")
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

    result = create_embeddings_for_pdf(
        file_path=file_path,
        model_names=selected_models,
        max_vectors=max_vectors,
        target_dimension=target_dimension,
        chunking_strategy=chunking_strategy,
    )
    print(f"Total vectors created: {len(result['vectors'])}")
    print(f"Combined embedding dimension: {result['combined_dimension']}")
    print(f"Final embedding dimension: {result['final_dimension']}")

    if result["vectors"]:
        with open("embeddings_output.json", "w", encoding="utf-8") as f:
            json.dump(result, f)
        print("Saved vectors to embeddings_output.json")












































# from langchain_ollama import OllamaEmbeddings

# embeddings = OllamaEmbeddings(
#     model="nomic-embed-text-v2-moe:latest",
# )
# # Create a vector store with a sample text
# from langchain_core.vectorstores import InMemoryVectorStore

# text = "LangChain is the framework for building context-aware reasoning applications"

# vectorstore = InMemoryVectorStore.from_texts(
#     [text],
#     embedding=embeddings,
# )

# # Use the vectorstore as a retriever
# retriever = vectorstore.as_retriever()

# # Retrieve the most similar text
# retrieved_documents = retriever.invoke("What is LangChain?")

# # Show the retrieved document's content
# print(retrieved_documents[0].page_content)