# Charmed-XGBoost: High-Performance C++ MLOps on MicroK8s

**Charmed-XGBoost** is a cloud-native MLOps pipeline designed to demonstrate the bridge between high-performance "bare-metal" engineering and modern Kubernetes orchestration. This project takes a custom-built XGBoost engine (written in **C++** with **pybind11** wrappers) and deploys it as a scalable, tracked, and containerized workload on **Canonical MicroK8s**.

---

## Project Goals

The primary objective of this repository is to showcase a "Day 2 Operations" mindset for machine learning—moving beyond local notebooks into a production-grade environment.

*   **Hybrid Language Engineering:** Containerizing a C++/Python application using **multi-stage Docker builds**. This ensures a small, secure production image by separating the build-time compilers (gcc/cmake) from the runtime environment.
*   **Cloud-Native Orchestration:** Deploying and managing ML workloads on **MicroK8s** utilizing **Juju-inspired** operational patterns.
*   **Automated Experiment Tracking:** Integrating **MLflow** within the Kubernetes cluster to log hyperparameters, metrics, and performance data automatically via internal cluster networking.
*   **Hybrid Cloud Storage:** Bridging private infrastructure (K8s) with public cloud (Azure) by implementing **Azure Blob Storage** as a remote artifact repository for model weights.
*   **Infrastructure as Code (IaC):** Moving from imperative manual setup to declarative YAML-based deployments, utilizing **Kubernetes Secrets** for secure Azure credential management.

---

## The Stack

*   **Core Engine:** C++17 with `pybind11` for Python bindings.
*   **Orchestration:** [MicroK8s](https://microk8s.io/) (Canonical’s lightweight, zero-ops Kubernetes distribution).
*   **Automation:** [Juju](https://juju.is/) (for deploying the Charmed MLflow stack).
*   **Tracking:** MLflow (hosted in-cluster).
*   **Artifact Store:** Azure Blob Storage (Microsoft Cloud).
*   **Containers:** Docker (Multi-stage), Kubernetes Jobs, and YAML Manifests.

---

## Architecture & Workflow

The system is designed around the concept of decoupling the management layer from the workload layer:

1.  **Build Phase (Multi-Stage):** A `builder` container uses `cmake` and `g++` to compile the C++ source. The resulting shared object library (`.so`) is then copied into a slim Python image.
2.  **Deployment:** A Kubernetes `Job` is triggered via `kubectl`, spinning up a Pod to execute the training run.
3.  **Logging:** The Python wrapper utilizes internal Kubernetes DNS (`mlflow-server.default.svc.cluster.local`) to communicate with the MLflow service.
4.  **Persistence:** The final trained model is pushed securely to **Azure Blob Storage** using credentials injected via a Kubernetes Secret.

---

## Repository Structure

```text
├── src/
│   ├── cpp/               # Custom C++ XGBoost implementation
│   ├── bindings/          # pybind11 integration code
│   └── python/            # MLflow wrapper and training script
├── deploy/
│   ├── dockerfile         # Multi-stage build definition
│   ├── job.yaml           # Kubernetes Job manifest
│   └── secret.yaml        # Kubernetes Secret for Azure keys (Template)
├── CMakeLists.txt         # C++ Build configuration
├── requirements.txt       # Python dependencies
└── README.md
```

--- 

## Using Multipass

MLFlow init
```
mlflow server \
  --backend-store-uri sqlite:////Users/jasonliang/Dev/AIML/mlflow/mlflow.db \
  --default-artifact-root /Users/jasonliang/Dev/AIML/mlflow/artifacts/ \
  --host 0.0.0.0 \
  --port 1234 \
  --serve-artifacts
```

Launch VM
```
multipass launch --name k8s-single-node --cpus 4 --memory 6G --disk 40G --cloud-init microk8s-node.yaml 24.04
multipass exec k8s-single-node -- cloud-init status --wait
```

Transfer and Load Engine 
``` 
multipass transfer xgboost-local-test.tar k8s-single-node:/home/ubuntu/ # Copy the tarball from your Mac to the VM
multipass exec k8s-single-node -- sudo /snap/bin/microk8s ctr images import /home/ubuntu/xgboost-local-test.tar # Import the image into the MicroK8s container runtime
```

Mount Workspace and Run Job
```
multipass mount $(pwd) k8s-single-node:/workspace # Mount your current project directory to /workspace inside the VM
multipass exec k8s-single-node -- /snap/bin/microk8s kubectl apply -f /workspace/deploy/training-job.yaml # Command Kubernetes to execute your training job
```

Monitor Pipeline
``` 
multipass exec k8s-single-node -- /snap/bin/microk8s kubectl logs -f job/cpp-xgboost-training-job
```