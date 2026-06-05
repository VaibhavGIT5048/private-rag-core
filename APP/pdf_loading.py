import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List
from uuid import uuid4

try:
    import pdfplumber
except ModuleNotFoundError:
    pdfplumber = None
    from pypdf import PdfReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    page_content: str
    metadata: dict


def load_pdf(pdf_path: str | Path) -> List[PageContent]:
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path.absolute()}")

    documents: List[PageContent] = []
    blank_pages = 0

    try:
        if pdfplumber is not None:
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if not text:
                        blank_pages += 1
                        continue
                    documents.append(PageContent(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "page": page_num,
                            "total_pages": total_pages,
                            "char_count": len(text),
                            "id": str(uuid4()),
                        }
                    ))
        else:
            logger.warning("pdfplumber not found; using pypdf fallback loader")
            pdf = PdfReader(str(path))
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    blank_pages += 1
                    continue
                documents.append(PageContent(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": page_num,
                        "total_pages": total_pages,
                        "char_count": len(text),
                        "id": str(uuid4()),
                    }
                ))
    except Exception as e:
        logger.error(f"Failed to process PDF {path.name}: {e}")
        raise

    total_chars = sum(len(d.page_content) for d in documents)
    logger.info(
        f"Processed '{path.name}': {len(documents)} pages loaded, "
        f"{blank_pages} blank skipped, {total_chars:,} total chars."
    )
    return documents


def load_pdfs(pdf_paths: list[str | Path]) -> List[PageContent]:
    all_docs: List[PageContent] = []
    for pdf_path in pdf_paths:
        docs = load_pdf(pdf_path)
        all_docs.extend(docs)

    logger.info(
        f"Processed {len(pdf_paths)} PDFs -> {len(all_docs)} total extracted pages."
    )
    return all_docs


def preview_pages(documents: List[PageContent], n: int = 2, preview_chars: int = 400) -> None:
    if not documents:
        logger.warning("No documents to preview.")
        return

    print(f"\n{'═'*60}\n   PAGE PREVIEW (first {n} pages)\n{'═'*60}")
    for doc in documents[:n]:
        meta = doc.metadata
        print(f"\n── Page {meta['page']} of {meta['total_pages']} ({meta['char_count']} chars) ──")
        content = doc.page_content
        print(content[:preview_chars] + ("..." if len(content) > preview_chars else ""))
    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
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

    parser = argparse.ArgumentParser(description="Load and extract text from a PDF for RAG pipelines.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file")
    parser.add_argument("--preview", type=int, default=2, help="Number of pages to preview")
    args = parser.parse_args()

    try:
        pdf_path = Path(args.pdf_path) if args.pdf_path else choose_pdf()
        docs = load_pdf(pdf_path)
        preview_pages(docs, n=args.preview)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        exit(1)