from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "rag-test"
BASE_URL = os.getenv("BASE_URL")
PERSIST_DIRECTORY = "./chroma_db_review"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/gemma-3n-e4b"


document = TextLoader("documentos\GTB_gold_Nov23.txt", encoding="utf-8").load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunks = splitter.split_documents(document)

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=PERSIST_DIRECTORY
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

query = "Como devo proceder caso tenha um item comprado roubado?"
similar_chunks = retriever.invoke(query)
similar_texts = [chunk.page_content for chunk in similar_chunks]

prompt = ChatPromptTemplate.from_messages([
  ("system", """
    Responda usando exclusivamente o conteúdo fornecido. \nContexto:\n{contexto}
   """),
  ("user", "Pergunta: {query}")
])

model = ChatOpenAI(
    model_name=LLM_MODEL,
    base_url=BASE_URL,
    temperature=0.5
)

chain = prompt | model | StrOutputParser()

trechos = retriever.invoke(query)

contexto = "\n\n".join([trecho.page_content for trecho in trechos])

print(chain.invoke({"contexto": contexto, "query": query}))