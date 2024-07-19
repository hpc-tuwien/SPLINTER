import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

# Load the samples from CSV file
latency_df = pd.read_csv('network_latency_samples.csv')

# Filter for VGG16 samples
vgg16_samples = latency_df[latency_df['Network'] == 'VGG16']['Latency']

# Reconstruct the parameters used for transformation
vgg16_min = 167.4
vgg16_max = 5072.5
shape_parameter = 1

# Find the original scale parameter used
samples = np.random.default_rng(seed=123456789).weibull(shape_parameter, 1000)
min_sample = np.min(samples)
max_sample = np.max(samples)

# Linear transformation parameters
a = (vgg16_max - vgg16_min) / (max_sample - min_sample)
b = vgg16_max - a * max_sample

# Generate x values for the PDF
x = np.linspace(vgg16_min, vgg16_max, 1000)

# Inverse transformation to get original scale
original_x = (x - b) / a

# Calculate the PDF of the original Weibull distribution
pdf = weibull_min.pdf(original_x, shape_parameter)

# Adjust the PDF for the scaling
scaled_pdf = pdf / a

# Plot the histogram of VGG16 samples
sns.histplot(vgg16_samples, bins=30, kde=False, color='tab:blue', label='Request Samples', stat='density')

# Plot the reconstructed PDF
plt.plot(x, scaled_pdf, '-', lw=2, label='Weibull PDF (shape=1)', color='tab:red')

# Add labels and legend
plt.xlabel('Latency (ms)')
plt.ylabel('Density')
plt.title('Inference Request Time Distribution')
plt.legend()
plt.savefig('request_samples.pdf', format='pdf')
# Show the plot
plt.show()
