import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

# Load the samples from CSV file
latency_df = pd.read_csv('network_latency_samples.csv')

# Reconstruct the parameters used for transformation
network_params = {
    'VGG16': {'min': 167.4, 'max': 5072.5},
    'ResNet50': {'min': 66.8, 'max': 1669.3},
    'MobileNetV2': {'min': 10.8, 'max': 391.0},
    'Vision Transformer': {'min': 193.2, 'max': 10608.0}
}

shape_parameter = 1
samples = np.random.default_rng(seed=123456789).weibull(shape_parameter, 1000)
min_sample = np.min(samples)
max_sample = np.max(samples)

# Determine global min and max values
global_min = min(params['min'] for params in network_params.values())
global_max = max(params['max'] for params in network_params.values())


# Linear transformation parameters for different networks
def get_transformation_params(min_val, max_val):
    a = (max_val - min_val) / (max_sample - min_sample)
    b = max_val - a * max_sample
    return a, b


# Define font size variables
scale_fonts = -1
label_font_size = 14 + scale_fonts
tick_font_size = 12 + scale_fonts
legend_font_size = 12 + scale_fonts
offset_font_size = 12 + scale_fonts

# Plot the PDFs
plt.figure(figsize=(5, 3))
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
for i, (network, params) in enumerate(network_params.items()):
    min_val = params['min']
    max_val = params['max']
    a, b = get_transformation_params(min_val, max_val)

    # Generate x values for the PDF
    x = np.linspace(min_val, max_val, 1000)

    # Inverse transformation to get original scale
    original_x = (x - b) / a

    # Calculate the PDF of the original Weibull distribution
    pdf = weibull_min.pdf(original_x, shape_parameter)

    # Adjust the PDF for the scaling
    scaled_pdf = pdf / a

    # Plot the reconstructed PDF and fill the area under the curve
    plt.plot(x, scaled_pdf, '-', lw=2, label=network, color=colors[i])
    plt.fill_between(x, scaled_pdf, color=colors[i], alpha=0.3)

# Scale the x-axis logarithmically and set limits
plt.xscale('log')
plt.xlim(global_min, global_max)

# Add labels and legend with updated font sizes
plt.xlabel('Latency (ms)', fontsize=label_font_size)
plt.ylabel('Density', fontsize=label_font_size)
plt.legend(fontsize=legend_font_size)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)

# Apply tight layout
plt.tight_layout()

# Save the plot
plt.savefig('request_samples_log.pdf', format='pdf')

# Show the plot
plt.show()
