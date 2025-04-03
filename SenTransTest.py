
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import CouchbaseException
from couchbase.options import (ClusterOptions, ClusterTimeoutOptions,QueryOptions)
from datetime import timedelta
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from sentence_transformers import SentenceTransformer

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

model = SentenceTransformer(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to generate embedding for a text input
def generate_embedding(text):
    return model.encode(text)

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


# retriever = vector_store.as_retriever()
# SELECT * FROM `travel-sample` WHERE type = 'airport' LIMIT 10

#Function to insert data into Couchbase with embeddings
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
        # Generate the embedding for the text
        embedding = embeddings.embed_query(text)
        
        # Create the document to store in Couchbase
        document = {
            "text": text,
            "embedding": embedding
        }
        doc_result = embeddings.embed_documents(text)
        # Store the document in Couchbase
        vector_store.add_texts(document)
        #vector_store.similarity_search("ola", k=1)
        print(f"Inserted: {document}")

insert_data_into_couchbase()