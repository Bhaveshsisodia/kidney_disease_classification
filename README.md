# Kidney Disease Classification

## Overview

This project implements an end-to-end machine learning pipeline for classifying kidney diseases using Convolutional Neural Networks (CNNs). The system leverages transfer learning with VGG16 architecture to classify CT scan images into Normal and Tumor categories. The project follows MLOps best practices with experiment tracking via MLflow, data versioning with DVC, and modular code structure for scalability and maintainability.

## Features

- **Data Ingestion**: Automated download and extraction of kidney CT scan datasets from Google Drive
- **Model Preparation**: Transfer learning setup using pre-trained VGG16 model with custom classification layers
- **Training Pipeline**: Configurable training with data augmentation, early stopping, and hyperparameter tuning
- **Model Evaluation**: Comprehensive evaluation metrics with validation on unseen data
- **MLflow Integration**: Complete experiment tracking, model versioning, and artifact logging to DagsHub
- **Modular Architecture**: Clean separation of concerns with reusable components
- **Configuration Management**: YAML-based configuration for easy parameter tuning
- **Logging**: Structured logging throughout the pipeline for debugging and monitoring

## Technology Stack

- **Programming Language**: Python 3.9+
- **Deep Learning Framework**: TensorFlow 2.x with Keras
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC (Data Version Control)
- **Configuration Management**: Python Box, PyYAML
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Flask (for potential API deployment)
- **Package Management**: pip, setuptools

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git
- DVC (for data versioning)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bhaveshsisodia/kidney_disease_classification.git
   cd kidney_disease_classification
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

5. **Set up MLflow tracking credentials** (for DagsHub integration):
   ```bash
   export MLFLOW_TRACKING_USERNAME=your_dagshub_username
   export MLFLOW_TRACKING_PASSWORD=your_dagshub_access_token
   ```

## Usage

### Running the Complete Pipeline

Execute the entire ML pipeline from data ingestion to model evaluation:

```bash
python main.py
```

This will run all stages sequentially:
1. Data Ingestion
2. Base Model Preparation
3. Model Training
4. Model Evaluation with MLflow logging

### Running Individual Stages

You can also run specific pipeline stages:

```bash
# Data Ingestion
python -m kidney_disease_classifier.pipeline.stage_01_data_ingestion

# Base Model Preparation
python -m kidney_disease_classifier.pipeline.stage_02_prepare_base_model

# Model Training
python -m kidney_disease_classifier.pipeline.stage_03_model_training

# Model Evaluation
python -m kidney_disease_classifier.pipeline.stage_04_model_evaluation_mlflow
```

### Configuration

Modify hyperparameters and settings in:
- `config/config.yaml`: Pipeline configuration
- `params.yaml`: Model hyperparameters

## Project Structure

```
kidney_disease_classification/
├── artifacts/                    # Generated artifacts and models
│   ├── data_ingestion/          # Downloaded and extracted data
│   ├── prepare_base_model/      # Base and updated models
│   └── training/                # Trained models
├── config/                      # Configuration files
│   └── config.yaml
├── logs/                        # Application logs
├── mlruns/                      # MLflow experiment runs
├── research/                    # Jupyter notebooks for experimentation
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation_with_mlflow.ipynb
├── src/kidney_disease_classifier/  # Source code
│   ├── components/               # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py
│   │   ├── model_training.py
│   │   └── mode_evaluation_mlflow.py
│   ├── config/                  # Configuration management
│   ├── constants/               # Project constants
│   ├── entity/                  # Data entities and configurations
│   ├── pipeline/                # Pipeline orchestration
│   └── utils/                   # Utility functions
├── templates/                   # Web templates (for Flask app)
├── tests/                       # Unit and integration tests
├── dvc.yaml                     # DVC pipeline configuration
├── main.py                      # Main entry point
├── params.yaml                  # Model parameters
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # Project documentation
```

## MLflow Integration

The project integrates with MLflow for comprehensive experiment tracking:

- **Experiment Management**: Automatic experiment creation and run tracking
- **Parameter Logging**: Hyperparameters and configuration parameters
- **Metric Tracking**: Training and validation metrics
- **Model Versioning**: Automatic model artifact storage and versioning
- **Remote Tracking**: Integration with DagsHub for collaborative ML development

### Viewing Experiments

Access your experiments at: https://dagshub.com/bhaveshsisodia2/kidney_disease_classification.mlflow

## Model Architecture

- **Base Model**: VGG16 pre-trained on ImageNet
- **Input Size**: 224x224x3 pixels
- **Output Classes**: 2 (Normal, Tumor)
- **Training Strategy**: Transfer learning with fine-tuning
- **Data Augmentation**: Random rotations, flips, and scaling
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam with configurable learning rate

## Data

The project uses kidney CT scan images categorized into:
- Normal kidney scans
- Kidney scans with tumors

Data is automatically downloaded from Google Drive during the ingestion stage.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive unit tests
- Update documentation for new features
- Ensure all tests pass before submitting PR

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- VGG16 model architecture from the Visual Geometry Group, University of Oxford
- TensorFlow and Keras for the deep learning framework
- MLflow for experiment tracking capabilities
- DagsHub for MLflow hosting and collaboration

## Contact

Bhavesh Kumar Lohar - bhaveshsisodia2@gmail.com

Project Link: [https://github.com/Bhaveshsisodia/kidney_disease_classification](https://github.com/Bhaveshsisodia/kidney_disease_classification)