from pathlib import Path
from pypdf import PdfReader


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    pdf_files = list(Path("data").glob("*.pdf"))

    for pdf_path in pdf_files:
        print("\n" + "=" * 80)
        print(f"Processing: {pdf_path.name}")
        print("=" * 80)

        text = read_pdf(pdf_path)

        for size in [300, 500, 1000]:
            chunks = chunk_text(text, chunk_size=size, overlap=100)

            print(f"\nChunk size: {size}")
            print(f"Number of chunks: {len(chunks)}")
            print("First chunk preview:")
            print(chunks[0][:500])