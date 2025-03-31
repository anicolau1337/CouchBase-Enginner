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
scope_name = "<<replace with your scope name>>"
collection_name = "<<replace with your collection name>>"

# Connect to Couchbase cluster
auth = PasswordAuthenticator(username, password)
options = ClusterOptions(auth)



options.apply_profile("wan_development")
try:
	cluster = Cluster(cluster_uri, options)
	# Wait until the cluster is ready for use.
	cluster.wait_until_ready(timedelta(seconds=5))
	print(f"Successfully connected")
except Exception as e:
	traceback.print_exc()
	print(f"Successfully missed")
	
# # Test connection
# try:
#     # Query the bucket to see if it's reachable
#     result = collection.get("airport_1")  # Try fetching a document
#     print(f"Successfully connected: {result.content_as[dict]}")
# except CouchbaseException as e:
#     print(f"Error connecting to Couchbase: {str(e)}")
