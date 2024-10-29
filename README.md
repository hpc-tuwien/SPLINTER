# SPLINTER Framework

The deployment of machine learning (ML) models on edge devices presents a unique challenge due to limited computational resources and energy constraints. SPLINTER (Split Layer Inference and Hardware-Software Co-Design) is a two-phase framework designed to optimize the deployment of neural networks (NNs) across both edge and cloud devices. SPLINTER dynamically configures both software (e.g., split layers) and hardware parameters (e.g., accelerator usage, CPU frequency) to achieve optimal performance and energy efficiency.

![SPLINTER Overview](./figures/paper/overview.png)

## Abstract

The deployment of ML models on edge devices is challenged by limited computational resources and energy availability. While split computing enables the decomposition of large neural networks (NNs) and allows partial computation on both edge and cloud devices, identifying the most suitable split layer and hardware configurations is a non-trivial task. This process is hindered by the large configuration space, non-linear dependencies between software and hardware parameters, heterogeneous hardware and energy characteristics, and dynamic workload conditions.

To overcome this challenge, we propose SPLINTER, a two-phase framework that dynamically configures parameters across both software (i.e., split layer) and hardware (e.g., accelerator usage, CPU frequency). During the **Offline Phase**, we solve a multi-objective optimization problem using a meta-heuristic approach to discover optimal settings. In the **Online Phase**, a scheduling algorithm identifies the most suitable settings for incoming inference requests and configures the system accordingly.

We evaluate SPLINTER using popular pre-trained NNs on a real-world testbed. Experimental results demonstrate a reduction in energy consumption by up to 72% compared to cloud-only computation, while meeting approximately 90% of user requests' latency thresholds compared to baselines.

---

## Repository Structure

### `communication/`
- **hardware.py**: Script to apply configurations to the edge hardware.
- **service.proto**: Protobuf definition for gRPC communication between the edge and cloud devices.
- **service_pb2.py & service_pb2_grpc.py**: Auto-generated Python gRPC stubs for communication.
- **split_computing_client.py**: The gRPC client, running on the edge device, to handle communication with the cloud.
- **split_computing_server.py**: The gRPC server, running on the cloud, to handle communication from the edge.

### `controller/`
- **evaluate.py**: Evaluates baselines and dynamically runs configurations based on QoS requirements.
- **moop_solver.py**: Multi-Objective Optimization Problem (MOOP) solver, used during the **Offline Phase** of SPLINTER.
- **simulate.py**: Simulates the framework behavior using a pre-generated simulation pool.

### `data/`
- **simulation pool/**: Contains the generated simulation pool data.
- **workload/**: Contains generated workload data for experimentation.

### `figures/`
- **exploratory/**: Contains exploratory figures generated during the project.
- **paper/**: Contains additional figures related to the paper.
- **results/**: Contains the final results figures used in the paper.

### `generate data/`
- **generate_simulation_pool.ipynb**: Jupyter notebook to generate the simulation pool.
- **generate_sla_traces.py**: Script to generate synthetic workload data.

### `models/`
- **mobilenetv2/**, **resnet50/**, **VGG16/**, **ViT/**: Contains code for splitting the respective models. The pre-split models used in the experiments can be downloaded [here](https://mega.nz/file/LqYE0KpK#TF-G6WrdRuHjnp6KrrxkwhO51DnfE4J_bg93f2ZHA7M).

### `plotting/`
- Contains code for generating the various plots used in the evaluation:
  - **exhaustive_plots.ipynb**: Jupyter notebook for generating plots for the exhaustive comparison.
  - **exploratory_plots.ipynb**: Jupyter notebook for plotting the exploratory analysis.
  - **overhead_plots.ipynb**: Jupyter notebook for plotting overhead data.
  - **plot_all_sla.ipynb** and **plot_all_sla.py**: Scripts to plot workload distributions.
  - **result_plots.ipynb**: Plots the final experiment results.
  - **simulation_plots.ipynb**: Plots related to simulation data.

### `results/`
- **additional_trials/**: Results from additional evaluated trials.
- **optimization/**: Results from the optimization phase of SPLINTER.
- **overhead/**: Measurements of system overhead during the experiments.
- **simulation/**: Results from simulation experiments.
- **testbed/**: Results from real-world testbed experiments.

### `scripts/`
- **install-and-run.sh**: Script to automatically start the Docker container on the Grid5000 testbed for cloud deployment.

### Root Directory Files
- **requirements-controller.txt**: Python package dependencies for the controller node.
- **requirements-edge.txt**: Python package dependencies for the edge device.
- **Dockerfile**: Dockerfile used to create the cloud container environment.
- **README.md**: This README file explaining the repository structure and project overview.

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd splinter
   ```

2. **Install dependencies**:
   - For the edge device:
     ```bash
     pip install -r requirements-edge.txt
     ```
   - For the controller:
     ```bash
     pip install -r requirements-controller.txt
     ```

3. **Docker setup for cloud container**:
   - Build the Docker container using the provided Dockerfile:
     ```bash
     docker build -t splinter-cloud .
     ```

4. **Running SPLINTER**:
   - Set up the gRPC communication between the edge and cloud using `split_computing_client.py` (edge) and `split_computing_server.py` (cloud).
   - Use the `moop_solver.py` to perform the **Offline Phase** optimization.
   - Run `evaluate.py` to evaluate baselines or dynamically adapt to QoS requirements.
