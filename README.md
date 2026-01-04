# Satellite-Imagery-Based-Property-Valuation-
A Real Estate Analytics firm aims to improve its valuation framework by developing a Multimodal Regression Pipeline that predicts property market value using both tabular data and satellite imagery.

# Project Overview
This project implements a hybrid deep learning model to predict property values in King County. It combines traditional tabular features (square footage, location, etc.) with high-resolution satellite imagery from Mapbox to capture visual value drivers like greenery, neighborhood density, and waterfront access.

# Key Features
1. Multimodal Architecture: A dual-branch network using a CNN for image processing and an MLP for tabular data.

2. Bias Initialization: Final layer bias initialized to the dataset mean log-price to accelerate convergence and improve stability.

3. Feature Engineering: Includes one-hot encoding for 70 zipcodes and normalization of physical property specs.

4. Explainability: High correlation with visual cues like waterfront access and tree canopy.
