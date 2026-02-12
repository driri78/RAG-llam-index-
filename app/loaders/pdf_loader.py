from pathlib import Path
from llama_index.core import SimpleDirectoryReader


def pdf_loader():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    print(BASE_DIR)
    documents = SimpleDirectoryReader(
        input_dir=BASE_DIR / "data" / "documents" / "2025",
        required_exts=[".pdf"],
        recursive=True,
    ).load_data()

    return documents
