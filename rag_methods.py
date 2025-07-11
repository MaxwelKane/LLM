# import os
# import dotenv
# from time import time
# import streamlit as st

# from langchain_community.document_loaders.text import TextLoader
# from langchain_community.document_loaders import (
#     WebBaseLoader, 
#     PyPDFLoader, 
#     Docx2txtLoader,
# )

# # pip install docx2txt, pypdf
# from langchain_community.vectorstores import Pinecone as LangchainPinecone
# from langchain_community.vectorstores import Chroma
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# dotenv.load_dotenv()

# os.environ["USER_AGENT"] = "myagent"
# DB_DOCS_LIMIT = 10

# # Function to stream the response of the LLM 
# def stream_llm_response(llm_stream, messages):
#     response_message = ""

#     for chunk in llm_stream.stream(messages):
#         response_message += chunk.content
#         yield chunk

#     st.session_state.messages.append({"role": "assistant", "content": response_message})


# # --- Indexing Phase ---

# def load_doc_to_db():
#     # Use loader according to doc type
#     if "rag_docs" in st.session_state and st.session_state.rag_docs:
#         docs = [] 
#         for doc_file in st.session_state.rag_docs:
#             if doc_file.name not in st.session_state.rag_sources:
#                 if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
#                     os.makedirs("source_files", exist_ok=True)
#                     file_path = f"./source_files/{doc_file.name}"
#                     with open(file_path, "wb") as file:
#                         file.write(doc_file.read())

#                     try:
#                         if doc_file.type == "application/pdf":
#                             loader = PyPDFLoader(file_path)
#                         elif doc_file.name.endswith(".docx"):
#                             loader = Docx2txtLoader(file_path)
#                         elif doc_file.type in ["text/plain", "text/markdown"]:
#                             loader = TextLoader(file_path)
#                         else:
#                             st.warning(f"Document type {doc_file.type} not supported.")
#                             continue

#                         docs.extend(loader.load())
#                         st.session_state.rag_sources.append(doc_file.name)

#                     except Exception as e:
#                         st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
#                         print(f"Error loading document {doc_file.name}: {e}")
                    
#                     finally:
#                         os.remove(file_path)

#                 else:
#                     st.error(F"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

#         if docs:
#             _split_and_load_docs(docs)
#             st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")


# def load_url_to_db():
#     if "rag_url" in st.session_state and st.session_state.rag_url:
#         url = st.session_state.rag_url
#         docs = []
#         if url not in st.session_state.rag_sources:
#             if len(st.session_state.rag_sources) < 10:
#                 try:
#                     loader = WebBaseLoader(url)
#                     docs.extend(loader.load())
#                     st.session_state.rag_sources.append(url)

#                 except Exception as e:
#                     st.error(f"Error loading document from {url}: {e}")

#                 if docs:
#                     _split_and_load_docs(docs)
#                     st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")

#             else:
#                 st.error("Maximum number of documents reached (10).")


# def initialize_vector_db(docs):
#     if "AZ_OPENAI_API_KEY" not in os.environ:
#         embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
#     else:
#         embedding = AzureOpenAIEmbeddings(
#             api_key=os.getenv("AZ_OPENAI_API_KEY"), 
#             azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
#             model="text-embedding-3-large",
#             openai_api_version="2024-02-15-preview",
#         )

#     vector_db = Chroma.from_documents(
#         documents=docs,
#         embedding=embedding,
#         collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
#     )

#     # We need to manage the number of collections that we have in memory, we will keep the last 20
#     chroma_client = vector_db._client
#     collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
#     print("Number of collections:", len(collection_names))
#     while len(collection_names) > 20:
#         chroma_client.delete_collection(collection_names[0])
#         collection_names.pop(0)

#     return vector_db


# def _split_and_load_docs(docs):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=5000,
#         chunk_overlap=1000,
#     )

#     document_chunks = text_splitter.split_documents(docs)

#     # Add to ChromaDB as before
#     if "vector_db" not in st.session_state:
#         st.session_state.vector_db = initialize_vector_db(docs)
#     else:
#         st.session_state.vector_db.add_documents(document_chunks)

#     # # Dual-write: also add to Pinecone
#     # pinecone_api_key = os.getenv("PINECONE_API_KEY")
#     # pinecone_env = os.getenv("PINECONE_ENV")
#     # pinecone_index = os.getenv("PINECONE_INDEX")
#     # if pinecone_api_key and pinecone_env and pinecone_index:
#     #     pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
#     #     if "AZ_OPENAI_API_KEY" not in os.environ:
#     #         embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
#     #     else:
#     #         embedding = AzureOpenAIEmbeddings(
#     #             api_key=os.getenv("AZ_OPENAI_API_KEY"), 
#     #             azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
#     #             model="text-embedding-3-large",
#     #             openai_api_version="2024-02-15-preview",
#     #         )
#     #     index_name = pinecone_index
#     #     if index_name not in pinecone.list_indexes():
#     #         pinecone.create_index(index_name, dimension=1024)
#     #     pinecone_db = LangchainPinecone.from_existing_index(index_name, embedding)
#     #     pinecone_db.add_documents(document_chunks)


# # --- Retrieval Augmented Generation (RAG) Phase ---

# def _get_context_retriever_chain(vector_db, llm):
#     retriever = vector_db.as_retriever()
#     prompt = ChatPromptTemplate.from_messages([
#         MessagesPlaceholder(variable_name="messages"),
#         ("user", "{input}"),
#         ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
#     ])
#     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

#     return retriever_chain


# def get_conversational_rag_chain(llm):
#     retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#         """You are a helpful assistant. You will have to answer to user's queries.
#         You will have some context to help with your answers, but now always would be completely related or helpful.
#         You can also use your knowledge to assist answering the user's queries.\n
#         {context}"""),
#         MessagesPlaceholder(variable_name="messages"),
#         ("user", "{input}"),
#     ])
#     stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

#     return create_retrieval_chain(retriever_chain, stuff_documents_chain)


# def stream_llm_rag_response(llm_stream, messages):
#     conversation_rag_chain = get_conversational_rag_chain(llm_stream)
#     response_message = "*(RAG Response)*\n"
#     for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
#         response_message += chunk
#         yield chunk

#     st.session_state.messages.append({"role": "assistant", "content": response_message})

import os
import dotenv
from time import time
import streamlit as st

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)

# pip install docx2txt, pypdf
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore

# Import the query_alt.py function
from query_alt import query_pinecone_simple

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 10

# Function to stream the response of the LLM 
def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# --- Indexing Phase ---

def load_doc_to_db():
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(F"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)

                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")

            else:
                st.error("Maximum number of documents reached (10).")


def initialize_vector_db(docs):
    if "AZ_OPENAI_API_KEY" not in os.environ:
        embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    else:
        embedding = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZ_OPENAI_API_KEY"), 
            azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
            model="text-embedding-3-large",
            openai_api_version="2024-02-15-preview",
        )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
    )

    # We need to manage the number of collections that we have in memory, we will keep the last 20
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    print("Number of collections:", len(collection_names))
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )
    document_chunks = text_splitter.split_documents(docs) #if len(text_splitter)*text_splitter.chunk_size < 300_000 else text_splitter.split_documents(docs[:300_000 // text_splitter.chunk_size]) +text_splitter.split_documents(docs[300_000 // text_splitter.chunk_size:600_000])

    # Add to ChromaDB as before
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)

    # Dual-write: also add to Pinecone if enabled
    if st.session_state.get("use_pinecone", False):
        _add_to_pinecone(document_chunks)


def _add_to_pinecone(document_chunks):
    """Add documents to Pinecone vector store"""
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX", "ragtest")
        
        if not pinecone_api_key:
            st.warning("Pinecone API key not found. Please set PINECONE_API_KEY in environment.")
            return
        
        # Initialize embedding model for Pinecone
        embeddings = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index)
        
        # Prepare vectors for upsert
        vectors = []
        for i, doc in enumerate(document_chunks):
            doc_id = f"doc_{int(time())}_{i}"
            text = doc.page_content
            vector = embeddings.encode([text])[0].tolist()
            
            metadata = {
                "text": text,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0)
            }
            
            vectors.append({
                "id": doc_id,
                "values": vector,
                "metadata": metadata
            })
        
        # Upsert vectors to Pinecone
        index.upsert(vectors=vectors)
        st.toast("Documents also added to Pinecone!", icon="📌")
        
    except Exception as e:
        st.error(f"Error adding documents to Pinecone: {e}")
        print(f"Error adding documents to Pinecone: {e}")


# --- Retrieval Augmented Generation (RAG) Phase ---

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful assistant. You will have to answer to user's queries.
        You will have some context to help with your answers, but not always would be completely related or helpful.
        You can also use your knowledge to assist answering the user's queries.\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_pinecone_context(query, top_k=8):
    """Get context from Pinecone using the query_alt.py function"""
    try:
        results = query_pinecone_simple(query, top_k=top_k)
        
        # Extract text from results
        context_texts = []
        for match in results['matches']:
            if match['score'] > 0.5:  # Filter by similarity threshold
                metadata = match.get('metadata', {})
                text = metadata.get('text', '')
                source = metadata.get('source', 'unknown')
                context_texts.append(f"[Source: {source}]\n{text}")
        
        return "\n\n".join(context_texts)
    
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return ""


def stream_llm_rag_response(llm_stream, messages):
    # Check if we should use Pinecone
    if st.session_state.get("use_pinecone", False):
        # Use Pinecone for context retrieval
        query = messages[-1].content
        context = get_pinecone_context(query)
        
        # Create prompt with Pinecone context
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
            """You are a helpful assistant. You will have to answer to user's queries.
            You will have some context to help with your answers, but not always would be completely related or helpful.
            You can also use your knowledge to assist answering the user's queries.\n
            Context from Pinecone:
            {context}"""),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
        ])
        
        # Format the prompt
        formatted_messages = prompt_template.format_messages(
            context=context,
            messages=messages[:-1],
            input=messages[-1].content
        )
        
        response_message = "*(RAG Response - Pinecone)*\n"
        for chunk in llm_stream.stream(formatted_messages):
            response_message += chunk.content
            yield chunk
    else:
        # Use ChromaDB (original method)
        conversation_rag_chain = get_conversational_rag_chain(llm_stream)
        response_message = "*(RAG Response - ChromaDB)*\n"
        for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
            response_message += chunk
            yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})