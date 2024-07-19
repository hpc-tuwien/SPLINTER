import numpy as np
import pandas as pd

# Network latency bounds from the table
network_params = {
    'VGG16': {'min': 167.4, 'max': 5072.5},
    'ResNet50': {'min': 66.8, 'max': 1669.3},
    'MobileNetV2': {'min': 10.8, 'max': 391.0},
    'Vision Transformer': {'min': 193.2, 'max': 10608.0}
}

# Set the shape parameter for Weibull distribution
shape_parameter = 1

# Prepare a DataFrame to hold all the latency values for different networks
all_latency_values = []

# Random number generator
rng = np.random.default_rng(seed=123456789)

for network, params in network_params.items():
    min_val = params['min']
    max_val = params['max']

    # Sample from the Weibull distribution
    samples = rng.weibull(shape_parameter, 1000)

    min_sample = np.min(samples)
    max_sample = np.max(samples)

    # Setup linear transformation
    a = (max_val - min_val) / (max_sample - min_sample)
    b = max_val - a * max_sample

    # Scale the samples
    latency_values = [(a * s) + b for s in samples]

    # Append to the list with network name included
    for value in latency_values:
        all_latency_values.append((network, value))

# Convert to DataFrame for easy saving
latency_df = pd.DataFrame(all_latency_values, columns=['Network', 'Latency'])

# Save to CSV file
latency_df.to_csv('network_latency_samples.csv', index=False)
