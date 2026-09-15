import os
import json
from langchain_core.documents import Document
from pathlib import Path

from APP.rag.structure_chunking import structure_aware_split

# structure = split on the document's own headings (default).
# recursive = fixed-size character splitting, the fallback.
#
# Semantic chunking (LangChain's SemanticChunker) was removed: it embedded
# every sentence to locate topic breakpoints, which cost 7m28s on a 44-page
# PDF against the recursive splitter's 1m31s — and exceeded Azure Container
# Apps' ~240s ingress timeout — for a retrieval benefit that was never
# demonstrated. Headings already mark topic boundaries and cost no model call.
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "structure").strip().lower()

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None


def _recursive_splitter_factory(chunk_size: int, chunk_overlap: int):
    def build():
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError("RecursiveCharacterTextSplitter not available; cannot chunk documents.")
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )
    return build


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    factory = _recursive_splitter_factory(chunk_size, chunk_overlap)

    if CHUNKING_STRATEGY == "structure":
        chunks = structure_aware_split(documents, chunk_size, chunk_overlap, factory)
        if chunks:
            sectioned = sum(1 for c in chunks if c.metadata.get("section"))
            print(f"✅ Structure-aware chunking: {len(chunks)} chunks, {sectioned} carry a section heading")
        else:
            # No headings found at all (e.g. a flat text dump) — the recursive
            # splitter still produces usable chunks, so degrade rather than fail.
            print("⚠️ Structure-aware chunking produced nothing; falling back to RecursiveCharacterTextSplitter")
            chunks = factory().split_documents(documents)
    else:
        chunks = factory().split_documents(documents)
        print(f"✅ Recursive character chunking: {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]   = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    sizes = [len(c.page_content) for c in chunks]
    print(f"   Total chunks : {len(chunks)}")
    print(f"   Strategy     : {CHUNKING_STRATEGY}")
    print(f"   Chunk size   : {chunk_size} chars (overlap: {chunk_overlap})")
    if sizes:
        print(f"   Avg size     : {sum(sizes) // len(sizes)} chars")
        print(f"   Min size     : {min(sizes)} chars")
        print(f"   Max size     : {max(sizes)} chars")
    return chunks


def save_chunks_jsonl(chunks: list[Document], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    from APP.rag.pdf_loading import load_pdf

    def choose_pdf() -> Path:
        pdf_dir = Path("data")
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError("No PDFs found in the data folder.")

        print("Available PDFs:")
        for idx, pdf in enumerate(pdfs, start=1):
            print(f"  {idx}. {pdf.name}")

        selection = input("Select a PDF number (or press Enter to cancel): ").strip()
        if not selection:
            raise SystemExit("Cancelled.")

        try:
            return pdfs[int(selection) - 1]
        except (ValueError, IndexError):
            raise SystemExit("Invalid selection.")

    parser = argparse.ArgumentParser(description="Load a PDF, chunk it, and save the result.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file")
    parser.add_argument("--output", type=str, default="data/chunks/chunks.jsonl", help="Path to save chunks as JSONL")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path) if args.pdf_path else choose_pdf()

    documents = load_pdf(str(pdf_path))
    chunks = chunk_documents(documents)

    save_chunks_jsonl(chunks, args.output)
    print(f"Saved chunks to {args.output}")
