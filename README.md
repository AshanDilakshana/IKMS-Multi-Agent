# Building a Knowledge-Based Q&A Application with

- [NewFeatures](#newfeatures)

# LangChain and Pinecone

In this session, we will develop a **document question-answering application** step by step. The application
will load a knowledge document (a PDF), index its content in a vector database, and use a GPT-based
language model to answer questions by retrieving information from the document. We’ll use **LangChain
1.0** (with the new LangGraph framework) for building our pipeline, **Pinecone** as the vector database, and an
OpenAI GPT-3.5 model (a "mini" GPT) for answering questions. Each part below introduces a component of
the system with background and code snippets.

## 1. Selecting and Ingesting a Knowledge Document

**Choosing a Document:** First, choose a PDF document that contains the knowledge your app will use (for
example, a short research paper, a company FAQ, or a technical article). We will use a single PDF for
simplicity. The content of this PDF will be indexed so the AI can later retrieve information from it. Ensure you
have the file path or URL to the PDF.

**PDF Loader:** To extract text from the PDF, we use the LangChain **PyMuPDF4LLM** document loader (a
community integration). This loader uses the PyMuPDF library to convert PDF pages into text (Markdown
format) optimized for LLM processing. It handles complex layouts (multi-columns, tables) and outputs
clean text. We can load the PDF either as one combined document or as separate pages. For large PDFs, it’s
often useful to treat each page or section as a separate chunk for indexing.

Below is a code snippet to load a PDF file using PyMuPDF4LLMLoader. This will read the PDF and return a
list of Document objects (one per page in this example):

```
# Install the integration package first:
# pip install langchain-pymupdf4llm langchain-core
```

```
fromlangchain_pymupdf4llm importPyMuPDF4LLMLoader
```

```
# Initialize the PDF loader for a given file path (or URL)
loader = PyMuPDF4LLMLoader(
file_path="path/to/your/document.pdf",
mode="page" # "page" mode gives one Document per page; use "single" for
whole PDF as one Document
)
```

```
# Load the document(s) from the PDF
docs= loader.load()
```

```
1
```

```
print(f"Loaded {len(docs)} documents from the PDF.")
print(docs[0].page_content[:200]) # preview the first 200 characters of the
first page
```

**Explanation:** In the code above, we create a PyMuPDF4LLMLoader with mode="page" to split the PDF
by pages. The loader’s load() method returns a list of Document objects. Each Document contains the
page text in page_content and metadata (like page number). If your PDF is small or if you prefer a single
combined document, you could use mode="single" to get one Document with the entire PDF content.
Keep in mind that very large documents should be split into smaller chunks (e.g., by page or using a text
splitter) so that they can be embedded and retrieved efficiently.

## 2. Setting Up the Vector Database (Indexing Pipeline)

Once we have the text from the PDF, the next step is to **create vector embeddings** for that text and store
them in a vector database. We’ll use **Pinecone** for this purpose. Pinecone is a fully managed **vector
database** that excels at storing and querying high-dimensional embeddings for semantic search. In
other words, Pinecone allows us to store the document text in vector form and quickly find relevant parts
later using similarity search.

**Embedding Model:** To convert text into vectors (embeddings), we use a pre-trained model. A common
choice is OpenAI’s **text-embedding-ada-002** model, which turns text into a 1536-dimensional vector
representation. (You could also use other embedding models, e.g., from HuggingFace or Cohere, but
we'll use OpenAI for demonstration.) We will use the LangChain OpenAIEmbeddings class to interface
with this model. Make sure to set your OpenAI API key (e.g., via environment variable) before running the
code.

**Pinecone Setup:** You need a Pinecone account to get an API key and an environment name. In production,
you would create a Pinecone **index** (with a certain dimension matching the embedding size). For our
example, we'll create an index (if not already created) and then use LangChain’s integration to add our
document vectors. Ensure the pinecone Python package is installed (pip install pinecone-
client).

Below is a code snippet to generate embeddings for the loaded documents and index them in Pinecone:

```
# Install required packages:
# pip install pinecone-client langchain-pinecone langchain-openai
```

```
import os
import pinecone
fromlangchain_openai importOpenAIEmbeddings
```

```
# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")or "YOUR-PINECONE-API-KEY"
pinecone_env= os.environ.get("PINECONE_ENV") or"YOUR-PINECONE-ENV" # e.g.,
"us-central1-gcp"
```

```
2
```

```
3
```

```
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
```

```
# Create a Pinecone index if it doesn't exist
index_name= "knowledge-index"
ifindex_namenot inpinecone.list_indexes():
pinecone.create_index(index_name, dimension=1536) # 1536 for text-
embedding-ada-
index= pinecone.Index(index_name)
```

```
# Initialize the OpenAI embedding model
embedding_model= OpenAIEmbeddings(model="text-embedding-ada-002")
```

```
# Convert document pages to embeddings and upsert into Pinecone
fromlangchain_pineconeimport PineconeVectorStore
```

```
# Use PineconeVectorStore to add documents
vector_store= PineconeVectorStore(index=index, embedding=embedding_model,
text_key="page_content")
vector_store.add_documents(docs)
```

```
print("Indexed all documents in Pinecone.")
```

**Explanation:** This code connects to Pinecone using an API key and environment, creates a new index called
"knowledge-index" if one doesn’t exist, and initializes the OpenAI embedding model. We use
PineconeVectorStore (from the LangChain Pinecone integration) to store the documents. The
add_documents(docs) call will take each Document in our list, compute its embedding using
embedding_model, and upsert the vector into the Pinecone index with the text stored as metadata
(text_key="page_content"). After running this, our document’s content is now indexed in Pinecone as
vectors.

```
Note: In a real application, you might want to chunk the text further (for example, splitting
long pages into smaller paragraphs) before embedding, to improve retrieval granularity.
LangChain offers text splitters for this. Since our example uses at most a page per chunk, we
proceed with that for simplicity. Also, remember to keep API keys secure (e.g., use
environment variables as shown, rather than hard-coding them).
```

## 3. Integrating a GPT Model for Question Answering

With our knowledge document indexed in Pinecone, we can now build the **question-answering (QA)
component**. This involves using a language model (LLM) to generate answers to user queries, with the help
of the stored knowledge. The typical approach is **Retrieval-Augmented Generation (RAG)** : when a
question is asked, we retrieve the most relevant document chunks from Pinecone and feed those, along
with the question, to the GPT model to help it formulate an informed answer.

**Retrieval:** We will use the LangChain retriever interface to fetch relevant chunks. The
PineconeVectorStore we created can be turned into a retriever. For example,

vector_store.as_retriever(k=3) will allow us to retrieve the top 3 most similar chunks for any
query.

**LLM Choice:** We’ll use **OpenAI GPT-3.5 Turbo** via LangChain’s ChatOpenAI class as our LLM. This model
(sometimes referred to as a “mini” GPT-4) is cost-effective and sufficient for demonstration. You could swap
in a larger model (like GPT-4 or an open-source alternative) if needed, but GPT-3.5 is fast and works well for
Q&A on a single document.

**QA Chain:** LangChain provides a convenient chain type called RetrievalQA that ties a retriever and an
LLM together. It will handle taking a question, retrieving relevant text, and then asking the LLM to answer
using that text. We’ll set this up with our retriever and OpenAI model.

Here’s the code to create a QA chain and perform a sample query:

```
fromlangchain.chat_models importChatOpenAI
fromlangchain.chains importRetrievalQA
```

```
# Initialize the chat model (ensure OPENAI_API_KEY is set in the environment)
chat_model= ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

```
# Create a RetrievalQA chain using the chat model and our Pinecone retriever
qa_chain= RetrievalQA.from_chain_type(
llm=chat_model,
chain_type="stuff", # "stuff" means it will stuff all retrieved docs into
the prompt (simplest method)
retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
return_source_documents=True # to return the source docs along with the
answer (optional)
)
```

```
# Example query to test the QA system
query= "YOUR_QUESTION_HERE" # e.g., "What is the main idea discussed in the
document?"
result = qa_chain({"query": query})
```

```
answer = result["result"]
sources= result.get("source_documents", [])
print("Q:", query)
print("A:", answer)
ifsources:
print(f"Retrieved {len(sources)} source document(s) for reference.")
```

**Explanation:** We create ChatOpenAI with the desired model and parameters (temperature 0 for
deterministic answers). Then we build a RetrievalQA chain with chain_type="stuff", which is a
straightforward method to send all retrieved text to the LLM. We configure the retriever to return the top 3
chunks from our vector_store. When we call qa_chain({"query": ...}), the chain will: (a) use the

retriever to get relevant text from Pinecone for the query, (b) feed the question and that text to the GPT
model, and (c) return the model’s answer. We also request source_documents so we can see which parts
of the PDF were used to derive the answer (this helps with transparency and debugging). The example ends
by printing the question and answer, and optionally info about sources.

At this stage, you can experiment by asking questions about the content of your PDF and verifying that the
answers make sense. The GPT model should pull in details from the document because the retriever
supplies those details as context.

## 4. Creating a Backend API for the Q&A System

To make our application accessible, we can wrap the QA chain into a simple **backend API**. This way, a user
(or another service) can send a question via an HTTP request and receive the AI’s answer. We'll use **FastAPI**
(a popular Python web framework) to create a quick API endpoint. (Alternatively, Flask could be used;
FastAPI just makes it easy to define a JSON response and test interactively.)

Below is a snippet showing how to set up a FastAPI server with an endpoint to answer questions. This
assumes that the qa_chain from the previous step is already created and available:

```
# Install FastAPI and Uvicorn if not already:
# pip install fastapi uvicorn
```

```
fromfastapiimport FastAPI
frompydanticimport BaseModel
```

```
app= FastAPI()
```

```
# Define a request schema for the question
classQuestionRequest(BaseModel):
query: str
```

```
@app.post("/ask")
defask_question(request: QuestionRequest):
"""Endpoint to get an answer for a given question."""
user_query = request.query
result = qa_chain({"query": user_query})
answer = result["result"]
return {"question": user_query, "answer": answer}
```

```
# To run the app, use: uvicorn main:app --reload
```

**Explanation:** We create a FastAPI app and define a POST endpoint /ask. Clients will send a JSON payload
like {"query": "Your question"}. The QuestionRequest Pydantic model enforces that structure.
In the ask_question function, we take the user_query, feed it to our qa_chain, and return the
answer in a JSON response. We include the original question and the answer in the response for clarity. (If

needed, you could also include source information in the response.) To run this API, you would use Uvicorn
as shown in the comment. Once running, any HTTP client (or a simple curl command) can hit [http://](http://)
localhost:8000/ask with a question to get answers from your knowledge base.

This backend setup is useful for demonstration purposes – for example, you could build a simple frontend
or chatbot interface that calls this API. It also mimics how a production service would expose an LLM-
powered QA system as an endpoint.

## 5. Production Considerations and Indexing Pipeline Management

We have a working prototype of a knowledge-powered Q&A system. In a production-like scenario, there are
additional considerations to ensure the system is robust and maintainable:

```
Indexing Pipeline: In a real system, you might have a pipeline that regularly processes and indexes
documents (especially if the knowledge base updates over time). This could be a scheduled job or a
separate service. The steps would include converting documents to text (as we did with the PDF
loader), splitting text into chunks, embedding those chunks, and upserting to Pinecone. For large-
scale deployments, consider using batch upsert operations and monitoring the indexing process for
errors.
```

```
Document Updates: If the content changes or new documents are added, you’ll need to update the
Pinecone index. Pinecone supports updating or deleting vectors by ID. Keeping track of document
IDs and metadata (like timestamps or versions in the metadata) is helpful. In our simple example, we
didn’t explicitly set IDs or metadata aside from the text, but in production you might store titles,
timestamps, or source URLs in the metadata for each vector.
```

```
Environment & Configuration: Ensure that sensitive keys (OpenAI, Pinecone API keys) are kept out
of code (we used os.environ.get which is good practice). Also, configuration like index name,
model names, etc., could be managed via config files or environment variables for flexibility.
```

```
Latency and Cost: Using an embedding API and an LLM API means each question involves network
calls. In production, you might implement caching strategies for repeated questions or popular
documents. Also, if using a smaller model is sufficient (as we chose GPT-3.5 over GPT-4 for cost),
that's a trade-off between cost and performance. You could further optimize by using a local
embedding model (to avoid the overhead per embedding call) if needed.
```

```
LangChain & LangGraph: With LangChain v1.0 and LangGraph, our simple chain is already quite
straightforward. For more complex applications, LangGraph provides a way to define agent
workflows and stateful interactions in a graph structure. In our case, we used a standard retrieval
QA chain (no custom agent logic). The new LangChain 1.0 APIs are more modular and scalable,
which positions us well if we later extend this app (for example, adding tools or multi-step
reasoning). The core retrieval-augmented QA pattern remains the same in LangChain v1.0 – we
create a retriever and an LLM chain to answer queries.
```

By following these steps and considerations, we have a **production-like retrieval augmented QA system**
on a single knowledge document, implemented in a clear and incremental way. The audience (newcomers)

### •

### •

### •

### •

### •

should focus on understanding each component: document loading, vector indexing, querying, and serving
the results. With this foundation, you can scale up to multiple documents or more advanced capabilities as
needed.

## Comprehensive Prompt for AI Code Generation

Finally, if using an AI coding assistant (such as Cursor AI) to develop this system, you can provide it with a
high-level prompt that encapsulates the plan. Below is a comprehensive prompt that instructs the AI to
generate the full application based on our design:

```
You are an expert Python developer and AI assistant.
```

```
**Task**: Build a knowledge-based Q&A application using LangChain v1.0 (with
LangGraph), Pinecone vector DB, and OpenAI GPT-3.5. The application should load
a PDF document, index its content into Pinecone, and answer user questions via
an API.
```

```
**Requirements & Steps**:
```

1. **Document Ingestion**: Use `langchain_pymupdf4llm` to load a PDF file. Split
by page into Document objects.
2. **Vector Indexing**: Initialize Pinecone (use API key and environment from
environment variables). Create a Pinecone index (if not exists) with dimension
1536 (for ada-002 embeddings). Use `OpenAIEmbeddings` (text-embedding-ada-002)
to embed each document page. Store the embeddings in Pinecone, including the
page text as metadata.
3. **QA Chain**: Set up a LangChain `RetrievalQA` chain. Use `ChatOpenAI` with
model `"gpt-3.5-turbo"` for the LLM. Use the Pinecone vector store as a
retriever (top 3 results). Ensure the chain returns the answer (and source
documents for verification).
4. **API Server**: Create a FastAPI application with an endpoint `/ask` that
accepts a JSON question and returns the answer. On each request, query the
`RetrievalQA` chain and return the answer in JSON.
5. **Testing**: Include a brief example of querying the API or chain in code to
demonstrate functionality (e.g., ask a sample question and print the answer).
6. **Good Practices**: Use environment variables for keys (OpenAI, Pinecone).
Add comments in code for clarity. Structure the code in logical sections
(loading, indexing, querying, API setup).

```
Now, please generate the Python code fulfilling the above requirements. Make
sure the code is well-organized, uses the specified libraries and classes, and
is suitable for a tutorial/demo setting.
```

Copy and paste the above prompt into the Cursor AI (or your coding assistant of choice) to guide it in
building the application. The AI should then produce the code for the entire system, following the plan
we've outlined. This approach demonstrates to newcomers how to translate a design into an
implementation with the help of AI coding tools.

langchain-pymupdf4llm · PyPI
<https://pypi.org/project/langchain-pymupdf4llm/>

Building a Vector Store from PDFs documents using Pinecone and LangChain | by Alex Rodrigues |
Medium
<https://medium.com/@alexrodriguesj/building-a-vector-store-from-pdfs-documents-using-pinecone-and-langchain->
a5c991b2a

```
1
```

```
2 3
```

## NewFeatures

- Memorizing past history to refer to it and answer the next question
- Storing past chat in the DB using SQLite
- We can continue past chat
