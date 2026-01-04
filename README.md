# Satellite-Imagery-Based-Property-Valuation-
A Real Estate Analytics firm aims to improve its valuation framework by developing a Multimodal Regression Pipeline that predicts property market value using both tabular data and satellite imagery.

# Project Overview
This project implements a hybrid deep learning model to predict property values in King County. It combines traditional tabular features (square footage, location, etc.) with high-resolution satellite imagery from Mapbox to capture visual value drivers like greenery, neighborhood density, and waterfront access.

# Key Features
1. Multimodal Architecture: A dual-branch network using a CNN for image processing and an MLP for tabular data.

2. Bias Initialization: Final layer bias initialized to the dataset mean log-price to accelerate convergence and improve stability.

3. Feature Engineering: Includes one-hot encoding for 70 zipcodes and normalization of physical property specs.

4. Explainability: High correlation with visual cues like waterfront access and tree canopy.

# Setup Instructions
1. Prerequisites
Ensure you have Python 3.10+ installed. It is recommended to run this in Google Colab using a T4 GPU for faster training.

### 2. Installation
Clone the repository and install the required libraries:

```bash
git clone (https://github.com/ayu853/Satellite-Imagery-Based-Property-Valuation.git)
cd Satellite-Imagery-Based-Property-Valuation
```
3. Data Preparation
Images: Place satellite images in /content/dataset/train_images and /content/dataset/test_images.

CSV Files: Ensure train.csv and test (2).csv are in the root directory.

# Project Structure
1. data_fetcher.py: Script to download satellite images via the Mapbox API.

2. preprocessing&model_training.ipynb: Data cleaning, feature engineering,zipcode encoding, the multimodal model architecture, bias initialization, and training loop.
   
3. multimodal_property_model.keras: The final trained model weights.

4. property_scaler.pkl: The saved scaler used to normalize input features.

# Running the Prediction
To generate predictions on the test set using the saved model:

```bash
import tensorflow as tf
import joblib
import numpy as np

# Load model and scaler
model = tf.keras.models.load_model('multimodal_property_model.keras')
scaler = joblib.load('property_scaler.pkl')

# Run prediction (ensure data is preprocessed and scaled)
# preds_log = model.predict(test_generator)
# final_prices = np.expm1(preds_log)
```




