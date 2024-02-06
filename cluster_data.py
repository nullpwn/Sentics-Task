from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
import random

# Load the CSV 
file_path = 'test_Data_1.csv'
data = pd.read_csv(file_path)

# Function to convert timestamp string to a sortable format
def parse_timestamp(ts):
    from datetime import datetime
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")

# Preprocess data: Sort by timestamp and parse timestamps
data['parsed_timestamp'] = data['timestamp_id'].apply(parse_timestamp)
data.sort_values('parsed_timestamp', inplace=True)

# Initialize lists to store processed cluster data
fused_data = []

# Iterate through data grouped by timestamp
for _, group in data.groupby('parsed_timestamp'):

    # Use DBSCAN for clustering based on x_position and y_position with a max distance of 2 meters
    # Consider each detection as an individual point, disregarding unique identifiers for initial clustering.
    X = group[['x_position', 'y_position']].values
    clustering = DBSCAN(eps=2, min_samples=1).fit(X)

    # Group data by cluster label to process unique_ids
    group['cluster_label'] = clustering.labels_
    for label, cluster_group in group.groupby('cluster_label'):
        unique_ids = cluster_group['unique_id'].unique()
        # Filter out 0s to find if any known unique_id exists in the cluster
        known_unique_ids = unique_ids[unique_ids != 0]

        # Determine the final unique_id for this cluster
        if known_unique_ids.size > 0:
            final_unique_id = known_unique_ids[0]  # Use the first known unique_id
        else:
            final_unique_id = 0  # No known unique_id, remain as 0

        # Prepare cluster_data
        cluster_data = cluster_group[['x_position', 'y_position', 'sensor_id']].values.tolist()

        # Use the timestamp of the first item in the cluster as the f_timestamp for simplicity
        f_timestamp = cluster_group['parsed_timestamp'].iloc[0]

        # Assign a random f_id for the cluster
        f_id = random.randint(1000, 9999)  # Simple random ID assignment

        # Append fused data entry
        fused_data.append([f_timestamp, f_id, cluster_data, final_unique_id])

# Convert fused data to a DataFrame
fused_df = pd.DataFrame(fused_data, columns=['f_timestamp', 'f_id', 'cluster_data', 'f_u_id'])

# Save to a new CSV file
output_path = 'fused_data.csv'
fused_df.to_csv(output_path, index=False)

output_path
