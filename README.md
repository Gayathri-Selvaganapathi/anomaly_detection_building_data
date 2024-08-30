# Anomaly Detection in Building Data

## Introduction
This project focuses on anomaly detection in buildings using machine learning. Buildings generate vast amounts of data, from temperature and humidity levels to energy consumption and occupancy patterns. Analyzing this data can provide invaluable insights into the health and efficiency of a building.

Anomalies—deviations from expected behavior in time series data—are crucial for effective building management. Detecting these anomalies helps identify issues such as unusual spikes in energy use, unexpected drops in temperature, or occupancy patterns that deviate from the norm. These anomalies can indicate potential problems like malfunctioning equipment, inefficient systems, or even security breaches.

Time series prediction, the process of forecasting future values based on historical data, is often intertwined with anomaly detection. By modeling expected behavior, we can highlight deviations as anomalies. This enables proactive maintenance, optimizing energy use by catching inefficiencies early on.

Effectively utilizing anomaly detection in time series data empowers building owners and managers to transform their spaces into intelligent environments—anticipating and responding to changes to ensure comfort, safety, and cost-effectiveness.

## Project Overview
This project demonstrates the application of various machine learning algorithms for anomaly detection in building data. The analysis is performed on an open-source dataset from the U.S. Department of Energy and the Lawrence Berkeley National Laboratory, which contains multi-year data on indoor CO2 levels and miscellaneous electrical consumption from building sockets.

## Key Areas of Focus

1. Proactive Maintenance:

    * Early identification of equipment malfunctions or inefficiencies through accurate anomaly detection.
    * Enables proactive maintenance actions, minimizing downtime, reducing repair costs, and extending equipment lifespan.

2.Energy Efficiency:

    * Pinpointing energy consumption anomalies to facilitate targeted interventions.
    * Optimizing building energy efficiency by identifying and addressing energy wastage, reducing operational costs, lowering carbon emissions, and enhancing sustainability.

3. Occupant Comfort:

    * Detecting anomalies in environmental conditions such as temperature or air quality.
    * Ensuring prompt actions to maintain comfortable and safe indoor environments, contributing to improved tenant satisfaction and offering a healthier indoor space.

## Algorithms Used
The project explores multiple algorithms for anomaly detection, demonstrating their effectiveness using visualizations:

1. Angle-Based Outlier Detection (ABOD):

    * Clusters points based on the angle from the origin to each point.

2. Gaussian Mixture Model (GMM):

    * Assumes a Gaussian distribution and estimates the mean and covariance to identify outliers.

3. Isolation Forest:

    * A well-known algorithm using decision trees to find boundaries within distributions.

4. Cluster-Based Local Outlier Factor (CBLOF):

    * Calculates the local outlier factor based on the density of neighboring clusters.

5. Histogram-Based Outlier Detection (HBOS):

    * Sorts points into bins and categorizes the bins as anomalous or not.

6. K-Nearest Neighbors (KNN):

    * Simple and effective, identifies outliers by calculating the distance to neighboring points.

7. Principal Component Analysis (PCA):

    * Transforms the data and classifies points as outliers if they lie off the principal components.

8. Support Vector Machine (SVM):

    * Finds a high-dimensional boundary that separates different distributions.

## Notebooks and Scripts:
1. unsupervised-time-series-methods.ipynb: Contains implementations of unsupervised anomaly detection techniques such as Z-Score, Isolation Forest, and Local Outlier Factor.

2. supervised-time-series-methods.ipynb: Focuses on supervised methods like Gradient Boosted Trees and Long Short-Term Memory (LSTM) networks for anomaly detection.

3. supervised-time-series-setup.ipynb: Sets up the environment and data preprocessing steps for supervised methods.

4. outlier-detection-demo.ipynb: Demonstrates a comprehensive walkthrough of detecting anomalies in the building dataset, including both supervised and unsupervised techniques.

5. utils.py: Contains utility functions used across the notebooks for data processing, visualization, and evaluation.

## Dataset

The dataset used in this project is from the U.S. Department of Energy and the Lawrence Berkeley National Laboratory. It includes multi-year data on indoor CO2 levels and miscellaneous electrical consumption, which are key indicators of building occupancy and energy usage patterns.

## Results and Conclusion
The results from applying these algorithms reveal interesting insights into building behavior. For example, the Isolation Forest and Z-Score methods identified certain thresholds for anomalous peaks in CO2 levels, while Local Outlier Factor highlighted different outliers, including lower bounds of CO2.

In supervised methods, the LSTM model detected more anomalies compared to Gradient Boosted Trees, suggesting a potential over-sensitivity of the neural network model. Further research is needed to understand the relationship between model performance and the number of detected anomalies, as this could inform adjustments to detection thresholds and improve model accuracy.

## Getting Started
To explore the project, clone the repository and install the required dependencies. The notebooks are well-commented and organized to guide you through the analysis.

```bash
git clone https://github.com/Gayathri-Selvaganapathi/anomaly_detection_building_data.git
cd anomaly-detection-building-data
pip install -r requirements.txt
```

## Running the Notebooks
1. Data Preprocessing: Start with the supervised-time-series-setup.ipynb to set up the environment and preprocess the data.

2. Unsupervised Methods: Explore unsupervised-time-series-methods.ipynb to apply various unsupervised anomaly detection algorithms.

3. Supervised Methods: Move on to supervised-time-series-methods.ipynb for supervised learning approaches.

4. Demo: Finally, check out the outlier-detection-demo.ipynb for a complete walkthrough of the project.