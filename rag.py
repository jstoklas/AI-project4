import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from docling.document_converter import DocumentConverter
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from operator import itemgetter 
from data_prep import load_pdfs, export_markdown_files, split_documents_by_structure, count_characters

load_dotenv()

'''USE THIS IF YOU DECIDE ON USING CLAUDE'''
#CLAUDE_KEY = os.getenv("CLAUDE_KEY")
#MODEL = "claude-3-7-sonnet-20250219"

'''USE THIS IF YOU DECIDE ON USING OLLAMA'''
MODEL = "llama3.1"

if MODEL.startswith("claude"):
    llm = ChatAnthropic(model=MODEL, api_key=CLAUDE_KEY)
else:
    llm = OllamaLLM(model=MODEL)
    embeddings = OllamaEmbeddings(model=MODEL)
    
    
#llm.invoke("tell me a joke")
parser = StrOutputParser()
chain = llm | parser
#chain.invoke("tell me a joke")

#call the functions completed in data_prep.py here 
documents = load_pdfs("data")
count_characters(documents)
export_markdown_files(documents)
split_docs = split_documents_by_structure(documents)
print(f"Total split chunks: {len(split_docs)}")
#question2 starts here

