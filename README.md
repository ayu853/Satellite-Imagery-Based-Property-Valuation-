# Satellite-Imagery-Based-Property-Valuation-
A Real Estate Analytics firm aims to improve its valuation framework by developing a Multimodal Regression Pipeline that predicts property market value using both tabular data and satellite imagery.

# Project Overview
This project implements a hybrid deep learning model to predict property values. It combines traditional tabular features (square footage, location, etc.) with high-resolution satellite imagery from Mapbox to capture visual value drivers like greenery, neighborhood density, and waterfront access.

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
git clone https://github.com/ayu853/Satellite-Imagery-Based-Property-Valuation.git
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

5. 24117032_final: Final predictions on test(2).csv
6. 24117032_Report: Final project report.
7. Property_Valuation_without_images.ipynb: Model without using images that is using lat and long as numerical data.

# Running the Prediction
To replicate this project or train the model from scratch, follow these steps in order:

1. Data Acquisition

    a. First, run the data_fetcher.py script to acquire the necessary satellite imagery.

    b. The script interfaces with the Mapbox Static Images API to download high-resolution tiles based on property coordinates.

    c. It ensures a 1:1 mapping between property IDs and image files, verifying a total of 21,506 unique images.


Prerequisite: You will need a Mapbox API Token.

2. Model Training & Evaluation

    a. Once the images are secured in the designated directory, open and run the preprocessing&model_training.ipynb notebook.


    b. Data Preprocessing: This step cleans the tabular data, performs One-Hot Encoding on zip codes, and applies log-transformation to the property prices.



    c. Training: The notebook builds the dual-branch Multi-Modal Neural Network and trains it using both the images and tabular features.



    d. Result: This process generates the final multimodal_property_model.keras and the property_scaler.pkl file.




