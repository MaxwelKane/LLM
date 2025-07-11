
# # from pinecone import Pinecone
# # from langchain_pinecone import PineconeVectorStore
# # from sentence_transformers import SentenceTransformer
# # import os
# # def query_pinecone_direct(query, top_k=8):
# #     """
# #     Query Pinecone using mxbai-embed-large for embeddings.
# #     Returns Pinecone search results.
# #     """
# #     # Load Pinecone credentials
# #     PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')#os.getenv("PINECONE_API_KEY")
# #     if not (PINECONE_API_KEY):
# #         raise ValueError("Pinecone credentials not set in environment/.env")
    
    
# #     # query_vector = model.encode([query])[0]

# #     # index = Pinecone(api_key=PINECONE_API_KEY).Index("ragtest")
# #     # vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
# #     # retriever = vectorstore.as_retriever(search_type="similarity")
# #     # Query Pinecone

# #     # embeddings = OllamaEmbeddings(name="mixedbread-ai/mxbai-embed-large-v1",config={"device": "cpu"} )
# #     embeddings = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")

# #     index = Pinecone(api_key=PINECONE_API_KEY).Index("ragtest")
# #     vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
# #     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
# #     # return retriever.get_relevant_documents(query)[:top_k]
# #     return retriever.invoke(query)
# # # print(query_pinecone_alt("What is the capital of France?"))
# # if __name__ == "__main__":
# #     try:
# #         results = query_pinecone_direct("What is the capital of France?")
# #         print("Query results:")
# #         for i, result in enumerate(results):
# #             print(f"{i+1}. Score: {result['score']:.4f}")
# #             print(f"   Content: {result['page_content'][:200]}...")
# #             print(f"   Metadata: {result['metadata']}")
# #             print()
# #     except Exception as e:
# #         print(f"Error: {e}")








# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# from sentence_transformers import SentenceTransformer
# import os

# def query_pinecone_alt(query, top_k=8):
#     """
#     Query Pinecone using mxbai-embed-large for embeddings.
#     Returns Pinecone search results.
#     """
#     # Load Pinecone credentials
#     PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') #= os.getenv("PINECONE_API_KEY")
#     if not PINECONE_API_KEY:
#         raise ValueError("Pinecone credentials not set in environment/.env")
    
#     # Initialize embedding model - CORRECTED
#     embeddings = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
    
#     # Initialize Pinecone
#     index = Pinecone(api_key=PINECONE_API_KEY).Index("ragtest")
#     vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    
#     return retriever.invoke(query)

# # Test the function
# if __name__ == "__main__":
#     print(query_pinecone_alt("What is the capital of France?"))

#query_alt.py
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

def query_pinecone_simple(query, top_k=8):
    """
    Simple query to Pinecone using mxbai-embed-large for embeddings.
    Returns Pinecone search results without LangChain dependency.
    """
    # Load Pinecone credentials
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')# os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("Pinecone API key not set in environment/.env")
    
    # Initialize embedding model
    embeddings = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
    
    # Generate query embedding
    query_vector = embeddings.encode([query])[0].tolist()
    
    # Initialize Pinecone and query
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("ragtest")
    
    # Query the index
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    
    return results

# Test the function
if __name__ == "__main__":
    try:
        results = query_pinecone_simple("What is the capital of France?")
        print("Query results:")
        for i, match in enumerate(results['matches']):
            print(f"{i+1}. Score: {match['score']:.4f}")
            print(f"   ID: {match['id']}")
            print(f"   Metadata: {match.get('metadata', {})}")
            print()
    except Exception as e:
        print(f"Error: {e}")