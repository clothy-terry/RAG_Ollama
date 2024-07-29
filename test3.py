import os

from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings

docs = [
    Document(page_content="The Nitrogen Cycle: Understanding the process through which nitrogen moves through the atmosphere, soil, and living organisms.", metadata={"title": "Nitrogen Cycle", "document_id": "1"}),
    Document(page_content="Hydrogen as a Fuel: Exploring the potential of hydrogen as a clean energy source for the future.", metadata={"title": "Hydrogen Fuel", "document_id": "2"}),
    Document(page_content="The History of Ancient Egypt: From the pyramids to the pharaohs, a journey through the ancient civilization.", metadata={"title": "Ancient Egypt", "document_id": "3"}),
    Document(page_content="Quantum Computing: How quantum mechanics is revolutionizing the field of computing.", metadata={"title": "Quantum Computing", "document_id": "4"}),
    Document(page_content="The Art of French Cuisine: A deep dive into the culinary traditions and famous dishes of France.", metadata={"title": "French Cuisine", "document_id": "5"}),
    Document(page_content="Space Exploration: The milestones and future prospects of humanity's journey into space.", metadata={"title": "Space Exploration", "document_id": "6"}),
    Document(page_content="Climate Change and Its Impact: Understanding the effects of global warming on our planet.", metadata={"title": "Climate Change", "document_id": "7"}),
    Document(page_content="Artificial Intelligence: The rise of AI and its applications in modern society.", metadata={"title": "Artificial Intelligence", "document_id": "8"}),
    Document(page_content="Modern Architecture: Key principles and influential architects shaping the buildings of today.", metadata={"title": "Modern Architecture", "document_id": "9"}),
    Document(page_content="Marine Biology: Discovering the diverse ecosystems and species living in our oceans.", metadata={"title": "Marine Biology", "document_id": "10"}),
    Document(page_content="Solar Power: Harnessing the sun's energy as a sustainable and renewable power source.", metadata={"title": "Solar Power", "document_id": "11"}),
    Document(page_content="Wind Energy: The advantages and challenges of using wind turbines to generate electricity.", metadata={"title": "Wind Energy", "document_id": "12"}),
    Document(page_content="Advancements in Battery Technology: How improving battery storage can support the widespread adoption of renewable energy.", metadata={"title": "Battery Technology", "document_id": "13"})
]

# Test if OllamaEmbeddings works correctly in isolation
embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
print("Embedding_model generated successfully")

# Use the Chroma vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    collection_name="rag-chroma",
    embedding=embedding_model,
)

# when we have limited documents in db, pass k value to retriever to indicate how many docs we retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print('retriever:', retriever)

### Retrieval Grader

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

local_llm = 'mistral:v0.1'
# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
# attempt to retrieve Hydrogen document
question = "clean energy source for the future"
docs = retriever.invoke(question)
# Why docs[1] the second retrieved doc?
doc_txt = docs[0].page_content
doc_id = docs[0].metadata["document_id"]
print('q:', question)
print('doc:', doc_txt)
print('doc_id:', doc_id)
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))


### Generate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

llm = ChatOllama(model=local_llm, temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
question = "What's the solution of clean energy source for the future?"
docs = retriever.invoke(question)
for doc in docs:
    print('doc_id: ', doc.metadata['document_id'])
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)
