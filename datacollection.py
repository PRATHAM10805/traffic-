import pandas as pd
import numpy as np

# Simulate traffic data
np.random.seed(42)
date_rng = pd.date_range(start='2023-01-01', end='2023-01-31', freq='H')
traffic_data = pd.DataFrame(date_rng, columns=['timestamp'])
traffic_data['traffic_volume'] = np.random.randint(50, 1000, size=(len(date_rng)))

# Simulate weather data
weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy']
traffic_data['weather'] = np.random.choice(weather_conditions, size=(len(date_rng)))

# Simulate event data
traffic_data['event'] = np.random.choice([0, 1], size=(len(date_rng)))  # 0 = no event, 1 = event

# Convert weather conditions to numerical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
traffic_data['weather'] = le.fit_transform(traffic_data['weather'])

# Create time-based features
traffic_data['hour'] = traffic_data['timestamp'].dt.hour
traffic_data['day_of_week'] = traffic_data['timestamp'].dt.dayofweek

# Drop the original timestamp column
traffic_data = traffic_data.drop(columns=['timestamp'])
