```
# Regression Model 

This project demonstrates how to build and run a regression model inside a Docker container.

## Overview

The project contains the following files:

- `Dockerfile`: Defines the Docker image for the project, including instructions to install dependencies and specify the command to run.
- `requirements.txt`: Lists all Python dependencies required for the project.
- `main.py`: Python script containing the regression model code.
- `data/`: Directory containing dataset file(s) used for training and testing the model.

## Usage

1. **Clone the Repository**: 
   ```bash
   git clone git@github.com:NacerKH/EstimationDeliveryModel.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd EstimationDeliveryModel
   ```

3. **Build the Docker Image**:
   ```bash
   docker build -t regression-model .
   ```

4. **Run the Docker Container**:
   ```bash
   docker run regression-model
   ```

5. **Get the Container Name**:
 - docker ps -a.

6. **Enter to Container**:  
  - docker exec -it $CONTAINER_NAME bash

7. **Run the Python Script**:
   - Run All Algorithms:
   ```bash
   python main.py
   ```
   - Get the best Model 
   ```bash
   python detectTheBestModel.py
   ```

## Notes

- Make sure you have Docker installed on your system before running the Docker commands.
- Update `main.py` with your actual regression model code.
- Replace `data/dataset.csv` with your dataset file.
```
