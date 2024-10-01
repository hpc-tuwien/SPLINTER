import numpy as np
import pandas as pd

# Network latency bounds from the table
network_params = {
    'vgg16': {'min': 90.6, 'max': 5026.8},
    'resnet50': {'min': 66.8, 'max': 1669.3},
    'mobilenetv2': {'min': 10.8, 'max': 391.0},
    'vit': {'min': 118.8, 'max': 10287.6}
}

# Set the shape parameter for Weibull distribution
shape_parameter = 1


def generate_sla_samples(num_samples):
    # Prepare a list to hold all the latency values for different networks
    all_latency_values = []

    # Random number generator
    rng = np.random.default_rng(seed=123456789)

    # Loop through each network and generate the latency samples
    for network, params in network_params.items():
        min_val = params['min']
        max_val = params['max']

        # Sample from the Weibull distribution
        samples = rng.weibull(shape_parameter, num_samples)

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

    # Convert to DataFrame for easy manipulation
    latency_df = pd.DataFrame(all_latency_values, columns=['Network', 'Latency'])

    return latency_df


# Generate 50 QoS samples
df_50_samples = generate_sla_samples(50)

# Manually save the DataFrame to a CSV file
df_50_samples.to_csv('network_latency_50_samples.csv', index=False)

# Generate 10000 QoS samples
df_10000_samples = generate_sla_samples(10000)

# Manually save the DataFrame to a CSV file
df_10000_samples.to_csv('network_latency_10000_samples.csv', index=False)
