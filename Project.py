from datetime import timedelta
import traceback
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import CouchbaseException
from couchbase.options import (ClusterOptions, ClusterTimeoutOptions,QueryOptions)


# Couchbase connection details
cluster_uri = "couchbases://cb.xuvf8kmkcscdzr1l.cloud.couchbase.com"  # Couchbase URI
username = "EngineerProject"
password = "Testing0406!"
bucket_name = "travel-sample"  # Travel Sample dataset
scope_name = "inventory"
collection_name = "airline"

# Connect to Couchbase cluster
auth = PasswordAuthenticator(username, password)
options = ClusterOptions(auth)



options.apply_profile("wan_development")
try:
	cluster = Cluster(cluster_uri, options)
	# Wait until the cluster is ready for use.
	cluster.wait_until_ready(timedelta(seconds=5))
	bucket = cluster.bucket(bucket_name)
	collection = bucket.scope(scope_name).collection(collection_name)
	airport_data = cluster.query("SELECT * FROM `travel-sample`.inventory.airport LIMIT 10")
	for row in airport_data:
    # each row is an instance of the query call
		try:
			name = row["airport"]["airportname"]
			callsign = row["airport"]["id"]
			print(f"Airline name: {name}, id: {callsign}")
			collection.upsert(callsign,f"Airline name: {name}")
		except KeyError:
			print("Row does not contain 'name' key")
except CouchbaseException as e:
	print(f"Error connecting to Couchbase: {str(e)}")
	
# # Test connection
# try:
#     # Query the bucket to see if it's reachable
#     result = collection.get("airport_1")  # Try fetching a document
#     print(f"Successfully connected: {result.content_as[dict]}")
# except CouchbaseException as e:
#     print(f"Error connecting to Couchbase: {str(e)}")
