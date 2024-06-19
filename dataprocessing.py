from sklearn.preprocessing import LabelEncoder
from datacollection import traffic_data
# Convert weather conditions to numerical values
le = LabelEncoder()
traffic_data['weather'] = le.fit_transform(traffic_data['weather'])

# Create time-based features
traffic_data['hour'] = traffic_data['timestamp'].dt.hour
traffic_data['day_of_week'] = traffic_data['timestamp'].dt.dayofweek

# Drop the original timestamp column
traffic_data = traffic_data.drop(columns=['timestamp'])

print(traffic_data.head())
