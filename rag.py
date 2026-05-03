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
template = '''

You are an AI course-notes assistant designed to provide information based on AI class notes.
Your primary function is to answer user questions using only the information
contained in the retrieved context based on the AI class notes. Do not introduce information from your own
knowledge or training that is not present in the provided documents.

 ### Retrieved Context:
 <context>
 {context}
 </context>

 ### User Query:   
 <query>
 {query}
 </query>

 To answer the user's query effectively, follow these steps :

 1. Carefully analyze the retrieved context documents.
 2. Identify information directly relevant to the user's query.
 3. Formulate a response using only information found in the retrieved context.
 4. If the retrieved documents contain conflicting information, acknowledge this in
your response.
 5. If the context does not contain relevant information, say:
   "I do not have enough information in the retrieved documents to answer this question."
 6. Do not invent facts, examples, definitions, or citations. 

 ### Response Format :

 <relevant_sources>
 List the specific pieces of information from the retrieved context that are
relevant to answering the query. Include source identifiers when available.
 </relevant_sources>

 <response>
 Provide a clear, concise answer based solely on the information you identified in
the relevant_sources section. Structure your response to directly address the
user's query.
 </response>

 ### Important Guidelines :

 - Only use information explicitly stated in the retrieved context.

 - Do not add explanations , examples , or details that aren't present in the
provided documents , even if they would be helpful or you believe them to be
accurate.
 - If the query cannot be fully answered using only the retrieved context , clearly
state this in your response . Provide whatever partial information is available
from the context , and identify what specific information is missing.
 - If the retrieved context contains no information relevant to the query,respond
with:

 <response>
 I don't have enough information in the retrieved documents to answer your question
about [ brief restatement of query ]. The current context doesn't contain
relevant information about this specific topic.
 </response>

 - Maintain objectivity and accuracy based strictly on the provided context.
 - When direct quotes are appropriate, use them and cite the source .
'''
prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)
chain = prompt | llm | parser
