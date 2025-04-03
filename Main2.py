from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import CouchbaseException
from couchbase.options import (ClusterOptions, ClusterTimeoutOptions,QueryOptions)
from datetime import timedelta
import couchbase.search as search
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from sentence_transformers import SentenceTransformer
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

model = SentenceTransformer('all-MiniLM-L6-v2')

COUCHBASE_CONNECTION_STRING = "couchbases://cb.xuvf8kmkcscdzr1l.cloud.couchbase.com"

DB_USERNAME = "EngineerProject"
DB_PASSWORD = "Testing0406!"

auth = PasswordAuthenticator(DB_USERNAME, DB_PASSWORD)
options = ClusterOptions(auth)
cluster = Cluster(COUCHBASE_CONNECTION_STRING, options)

# Wait until the cluster is ready for use.
cluster.wait_until_ready(timedelta(seconds=5))


BUCKET_NAME = "travel-sample"
SCOPE_NAME = "inventory"
COLLECTION_NAME = "airline"
SEARCH_INDEX_NAME = "Airport_Finder"




vector_store = CouchbaseVectorStore(
    cluster=cluster,
    bucket_name=BUCKET_NAME,
    scope_name=SCOPE_NAME,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    index_name=SEARCH_INDEX_NAME,
)
bucket = cluster.bucket(BUCKET_NAME)
collection = bucket.scope(SCOPE_NAME).collection(COLLECTION_NAME)

def generate_embedding(text):
    return model.encode(text)

def insert_data_into_couchbase():
    # Query the travel-sample dataset for airports
    airport_data = cluster.query("SELECT * FROM `travel-sample`.inventory.airport LIMIT 10")
    
    # Loop through each airport and insert it with its embedding
    for doc in airport_data:
        doc_id = doc["airport"]["id"]
        airport_name = doc["airport"]["airportname"]
        airport_city = doc["airport"]["city"]
        
        # Create a text representation of the airport
        text = f"{airport_name}, {airport_city}"
        print(text)
        embeddings = model.encode(text)
        doc_id = f"airport_{doc_id}"
        # Generate the embedding for the text
        for i, (text, emb) in enumerate(zip(text, embeddings)):
            document = {
            "name": text,
            "embedding": emb.tolist()  # Convert numpy array to list for Couchbase
            }
        
        try:
            collection.upsert(doc_id, document)
            print(f"Document {doc_id} stored successfully")
        except CouchbaseException as e:
            print(f"Error storing document {doc_id}: {e}")

insert_data_into_couchbase()


# Cache for results (in-memory) to avoid repeated searches
cached_results = {}

# Function to perform the search
def semantic_search(query, use_cache=True):
    # Check if the result is already cached
    if use_cache and query in cached_results:
        print("Using cached results...")
        return cached_results[query]
    
    # Step 1: Perform keyword search using Couchbase FTS (exact matches)
    keyword_query = {
        "query": {
            "match": {
                "field": "name",
                "query": query  # The user’s query
            }
        },
        "size": 5  # Top 5 results
    }
    
    # Execute the query
    search_results = search.SearchRequest.create(keyword_query)
    
    # Step 2: Perform semantic search using embeddings
    query_embedding = generate_embedding(query)
    
    # Fetch all stored embeddings and compare with the query embedding
    similarity_scores = []
    for row in search_results.rows():
        document = row['name']  # Get the text of the matched document
        embedding = np.array(row['embedding'])  # Get the stored embedding
        
        # Compute cosine similarity between query embedding and stored embedding
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarity_scores.append((document, similarity))
    
    # Sort by similarity score in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top results
    top_results = similarity_scores[:5]
    
    # Cache the results
    cached_results[query] = top_results
    return top_results

# Command-line interface (CLI) to interact with the user
def cli_interface():
    while True:
        query = input("Enter your query: ")
        
        if query.lower() == "exit":
            print("Exiting...")
            break
        
        results = semantic_search(query)
        
        print("Top search results:")
        for result in results:
            print(f"- {result[0]} (Similarity: {result[1]:.4f})")

# Run the CLI
cli_interface()
