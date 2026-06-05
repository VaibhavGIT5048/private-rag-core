import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]   = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    sizes = [len(c.page_content) for c in chunks]
    print(f"✅ [Phase 2] Chunking Complete")
    print(f"   Total chunks : {len(chunks)}")
    print(f"   Chunk size   : {chunk_size} chars (overlap: {chunk_overlap})")
    print(f"   Avg size     : {sum(sizes) // len(sizes)} chars")
    print(f"   Min size     : {min(sizes)} chars")
    print(f"   Max size     : {max(sizes)} chars")

    return chunks


def show_overlap(chunks: list[Document], chunk_index: int = 0):
    if chunk_index + 1 >= len(chunks):
        print("Not enough chunks to show overlap.")
        return

    a = chunks[chunk_index].page_content
    b = chunks[chunk_index + 1].page_content

    overlap_text = ""
    for length in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:length]):
            overlap_text = b[:length]
            break

    print(f"\n{'═'*60}")
    print(f"  OVERLAP CHECK: chunk {chunk_index} → chunk {chunk_index + 1}")
    print(f"{'═'*60}")
    print(f"\n[Chunk {chunk_index}] (last 200 chars):")
    print(f"  ...{a[-200:]}")
    print(f"\n[Chunk {chunk_index + 1}] (first 200 chars):")
    print(f"  {b[:200]}...")
    if overlap_text:
        print(f"\n✅ Overlapping text ({len(overlap_text)} chars):")
        print(f"  '{overlap_text[:150]}'")
    else:
        print(f"\n⚠️  No overlap detected — check chunk_overlap setting")
    print(f"\n{'═'*60}\n")


def preview_chunks(chunks: list[Document], n: int = 3):
    print(f"\n{'═'*60}")
    print(f"  CHUNK PREVIEW (first {n})")
    print(f"{'═'*60}")
    for chunk in chunks[:n]:
        m = chunk.metadata
        print(f"\n── chunk_id={m['chunk_id']} | page={m['page']} | size={m['chunk_size']} chars ──")
        print(chunk.page_content[:400])
        if m['chunk_size'] > 400:
            print(f"  ... [{m['chunk_size'] - 400} more chars]")
    print(f"\n{'═'*60}\n")


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
    from pathlib import Path
    from APP.pdf_loading import load_pdf

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

    parser = argparse.ArgumentParser(description="Load a PDF, chunk it, and preview results.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file")
    parser.add_argument("--preview", type=int, default=3, help="Number of chunks to preview")
    parser.add_argument("--overlap-check", action="store_true", help="Show overlap for chunk 0->1")
    parser.add_argument("--output", type=str, default="chunks/chunks.jsonl", help="Path to save chunks as JSONL")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path) if args.pdf_path else choose_pdf()

    documents = load_pdf(str(pdf_path))
    chunks = chunk_documents(documents)

    preview_chunks(chunks, n=args.preview)
    if args.overlap_check:
        show_overlap(chunks, chunk_index=0)

    save_chunks_jsonl(chunks, args.output)
    print(f"Saved chunks to {args.output}")