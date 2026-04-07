import os
import json
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_community.document_loaders import DirectoryLoader
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate 
from langchain.retrievers import MultiQueryRetriever
from langchain.evaluation import QAEvalChain

# Configurações iniciais

os.environ["LANGCHAIN_PROJECT"] = "rag-test"
BASE_URL = os.getenv("BASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

# Carrega os arquivos PDF da pasta "documentos"
pdfs = DirectoryLoader("documentos", glob="*.pdf").load()

# Inicializa o tokenizer do modelo "BAAI/bg3-m3"

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

# Cria um splitter de texto usando o tokenizer do modelo, com um tamanho de chunk de 1250 tokens e uma sobreposição de 150 tokens

splitter = CharacterTextSplitter.from_huggingface_tokenizer(
  tokenizer=tokenizer,
  chunk_size=1250,
  chunk_overlap=150,
)

# Divide os documentos em chunks usando o splitter criado

chunks = splitter.split_documents(pdfs)

# Cria as embeddings usando o modelo "bge-m3" da Ollama e armazena os chunks e as embeddings em um vetor store usando FAISS

embeddings = OllamaEmbeddings(model="bge-m3")

vector_store = FAISS.from_documents(
  documents=chunks,
  embedding=embeddings
)

# Cria um retriever a partir do vetor store, configurando para retornar os 3 chunks mais similares para uma consulta

retriever = vector_store.as_retriever()

# Define a consulta e recupera os chunks mais similares usando o retriever

prompt = ChatPromptTemplate.from_messages([
  ("system", """
    Responda usando exclusivamente o conteúdo fornecido. \nContexto:\n{contexto}
   """),
  ("user", "Pergunta: {query}")
])

# Inicializa o modelo de linguagem da Ollama, configurando para usar o modelo "gemma3:4b"

model = OllamaLLM(model="gemma3:4b")

# Cria a cadeia de RAG, conectando o prompt, o modelo e um parser de saída para formatar a resposta como string

chain = prompt | model | StrOutputParser()

# Define a consulta e executa a cadeia de RAG, passando a consulta e o contexto recuperado pelo retriever

query = "Como fazer um seguro viagem?"

strechts = retriever.invoke(query)

context = "\n\n".join([chunk.page_content for chunk in strechts])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain =  (
  {
    "contexto": lambda x: format_docs(retriever.invoke(x["query"])),  
    "query": lambda x: x["query"]
  }
  | prompt | model | StrOutputParser()
)

# print(rag_chain.invoke(
#   {"query": query}
# ))

query_model = OllamaLLM(model="gemma3:1b")

rewriter_prompt_template = """
Gere consulta de pesquisa para o banco de dados de vetores (VectorDB) a partir de uma pergunta do usuário,
permitindo uma resposta mais precisa por meio da busca semântica.
Basta retornar a consulta revisada do Vector DB, entre aspas.

Pergunta do usuário: {user_question}
Consulta revisada do Vector DB:
"""

rewriter_prompt = PromptTemplate.from_template(rewriter_prompt_template)

rewriter_chain = rewriter_prompt | query_model | StrOutputParser()

rewriter_rag_chain = (
  {
    "contexto": RunnablePassthrough()  | rewriter_chain | retriever,
    "query": RunnablePassthrough()
  }
  | prompt | model | StrOutputParser()
)

# print(rewriter_rag_chain.invoke(query))

multi_query_prompt_template = """Você é um assistente de modelo de linguagem de IA. Sua tarefa é gerar cinco
versões diferentes da pergunta do usuário para recuperar documentos relevantes de um banco de dados vetorial.
Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar
o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
Forneça estas perguntas alternativas separadas por quebras de linha.
Responda apenas com os textos das perguntas, sem introdução ou comentários finais. Não coloque bullets ou numeros nas linhas.
Pergunta original: {question}
Perguntas:"""

multi_query_prompt = PromptTemplate.from_template(multi_query_prompt_template)

multi_query_chain = multi_query_prompt | query_model | CommaSeparatedListOutputParser()

print(multi_query_chain.invoke(query))

multi_retriever = MultiQueryRetriever(retriever=retriever, llm_chain=multi_query_chain)

multi_rag_chain = (
  {
    "contexto": RunnablePassthrough()  | multi_retriever,
    "query": RunnablePassthrough()
  }
  | prompt | model | StrOutputParser()
)

# print(multi_rag_chain.invoke(query))

hyde_prompt_template = """
Escreve um parágrafo que possa responder a pergunta apresentada. Não adicione informações.

Pergunta: {user_question}
Parágrafo:
"""

hyde_prompt = PromptTemplate.from_template(hyde_prompt_template)

hyde_chain = hyde_prompt | query_model | StrOutputParser()

print(hyde_chain.invoke(query))

eval_chain = QAEvalChain.from_llm(query_model)

def avaliar(perguntas_resposta, geracoes):
  avaliacoes = eval_chain.evaluate(perguntas_resposta, geracoes)
  corretas = 0
  for i in enumerate(perguntas_resposta):
    corretas = corretas + (1 if avaliacoes[i]["results"].split("\n")[-1].split(":")[-1].strip() == "CORRECT" else 0)
  return corretas/len(perguntas_resposta)

qa_generate_prompt = PromptTemplate.from_template("""
Com base no texto abaixo, gere pares de pergunta e resposta.

Texto:
{contexto}

Gere 3 pares no formato JSON:
[
  {{
    "query": "...",
    "answer": "..."
  }}
]
""")

qa_generate_chain = qa_generate_prompt | query_model | StrOutputParser()

perguntas_respostas = []

for doc in strechts:
    result = qa_generate_chain.invoke({
        "contexto": doc.page_content
    })
    
    try:
        parsed = json.loads(result)
        perguntas_respostas.extend(parsed)
    except:
        pass

with open("qa_pairs.json", "w") as f:
  pairs = json.load(f)

perguntas_respostas = [p["qa_pairs"] for p in pairs]
  
geracoes_sem_rag = []
for pr in perguntas_respostas[:10]:
  geracoes_sem_rag.append({
        "result": query_model.invoke(pr["query"])
    })
  
geracoes_sem_rag_2 = [{"result": g["results"].content} for g in geracoes_sem_rag]

geracoes_multi_rag = []
for pr in perguntas_respostas[:10]:
  geracoes_multi_rag.append({"result": multi_rag_chain.invoke(pr["query"])})
  
print(avaliar(perguntas_respostas[:10], geracoes_multi_rag))