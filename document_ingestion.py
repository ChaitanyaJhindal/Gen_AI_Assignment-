import re

from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import CharacterTextSplitter


def split_legal_sections(document_content: str) -> list[str]:
    section_header_pattern = re.compile(
        r"(?im)(?=^\s*(?:section|sec\.?|article|chapter|clause)\s+[a-z0-9().\-/]+)",
    )
    parts = [part.strip() for part in section_header_pattern.split(document_content) if part.strip()]
    if len(parts) > 1:
        return parts

    numbered_heading_pattern = re.compile(r"(?m)(?=^\s*\d+(?:\.\d+)*\s+[A-Z][^\n]{2,})")
    parts = [part.strip() for part in numbered_heading_pattern.split(document_content) if part.strip()]
    return parts if parts else [document_content.strip()]


def extract_legal_metadata(chunk_text: str) -> dict[str, str]:
    act_match = re.search(
        r"(?i)\b([A-Z][A-Za-z,&\-\s]+ Act(?:,?\s*\d{4})?)\b",
        chunk_text,
    )
    section_match = re.search(
        r"(?i)\b(?:section|sec\.?|article|clause)\s+([a-z0-9().\-/]+)",
        chunk_text,
    )
    court_match = re.search(
        r"(?im)^\s*(?:in the\s+)?([A-Z][A-Z\s]*(?:COURT|TRIBUNAL)[^\n]*)",
        chunk_text,
    )

    return {
        "act": act_match.group(1).strip() if act_match else "unknown",
        "section": section_match.group(1).strip() if section_match else "unknown",
        "court": court_match.group(1).strip() if court_match else "unknown",
    }


def load_and_split_pdf(
    file_path: str,
    chunk_size: int = 100,
    chunk_overlap: int = 0,
    encoding_name: str = "cl100k_base",
    chunking_strategy: str = "generic",
) -> list[str]:
    loader = PDFMinerLoader(file_path)
    docs = loader.load()

    if not docs:
        return []

    document_content = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if chunking_strategy == "section-wise":
        section_chunks = split_legal_sections(document_content)
        normalized_chunks: list[str] = []
        for chunk in section_chunks:
            split_parts = text_splitter.split_text(chunk)
            normalized_chunks.extend(split_parts if split_parts else [chunk])
        return normalized_chunks

    return text_splitter.split_text(document_content)


if __name__ == "__main__":
    file_path = input("Enter Your Path:  ")
    print("Extracting and splitting the document")
    texts = load_and_split_pdf(file_path)
    print(f"Splitting is completed. Total chunks: {len(texts)}")
