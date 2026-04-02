from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import CharacterTextSplitter


def load_and_split_pdf(
    file_path: str,
    chunk_size: int = 100,
    chunk_overlap: int = 0,
    encoding_name: str = "cl100k_base",
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
    return text_splitter.split_text(document_content)


if __name__ == "__main__":
    file_path = input("Enter Your Path:  ")
    print("Extracting and splitting the document")
    texts = load_and_split_pdf(file_path)
    print(f"Splitting is completed. Total chunks: {len(texts)}")
