from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter

#Create a converter object for docling
converter = DocumentConverter()

def load_pdfs(path):
    documents = []

    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(path, filename)
            result = converter.convert(filepath)
            markdown_text = result.document.export_to_markdown()

            doc = Document (
                page_content = markdown_text,
                metadata = {"source": filename}
            )
            documents.append(doc)
    return documents

def export_markdown_files(documents, output_dir="data/processed_markdown"):
    os.makedirs(output_dir, exist_ok = True)
    for doc in documents:
        filename = doc.metadata["source"].replace(".pdf", ".md")
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc.page_content)

def split_documents_by_structure(documents):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    split_docs = []
    for doc in documents:
        splits = splitter.split_text(doc.page_content)
        split_docs.extend(splits)
    
    return split_docs

#this is not needed but helps see the character count in each pdf
def count_characters(documents):
    for doc in documents:
        filename = doc.metadata["source"]
        count = len(doc.page_content)
        print(f"{filename}: {count} characters")


#call these functions in rag.py
