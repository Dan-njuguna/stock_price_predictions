#!/usr/bin/env python3

'''
AUTHOR: DAN NJUGUNA
DATE: 2025-04-26

This module implements data loading and preprocessing for the training dataset.
'''

import sys
import pandas as pd
import numpy as np
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs('../logs', exist_ok=True)
file_handler = logging.FileHandler('../logs/preprocess_dataset.log')
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info("Starting preprocessing dataset...")

# Setting constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SUB_DIRS = ["indices", "stocks"]


# TODO: Load all tthe data files inside the train/indices directory for preprocessing
# NOTE: The reason to use indices is to have a general overview of the market performance
def load_indices_data():
    """
    Load and preprocess indices data from the train/indices directory.
    """
    indices_files = glob.glob(os.path.join(TRAIN_DIR, SUB_DIRS[0], '*.csv'))
    indices_data = []
    
    for file in indices_files:
        try:
            df = pd.read_csv(file)
            # Extract ticker from filename
            ticker = os.path.splitext(os.path.basename(file))[0]
            df['Ticker'] = ticker
            
            # Process required columns
            df['Date'] = pd.to_datetime(df['Date'])
            df['Returns'] = df['Adjusted'].pct_change()
            
            # Set standard column order
            columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 
                      'Adjusted', 'Returns', 'Volume']
            df = df[columns]
            
            df.set_index('Date', inplace=True)
            indices_data.append(df)
            
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not indices_data:
        logger.warning("No indices data found.")
        return None
    
    combined_indices_data = pd.concat(indices_data, axis=0)
    combined_indices_data.sort_index(inplace=True)
    logger.info(f"Loaded and combined indices data with shape {combined_indices_data.shape}")
    return combined_indices_data

if __name__ == "__main__":
    # Load indices data
    indices_data = load_indices_data()
    
    if indices_data is not None:
        # Save the preprocessed data to a CSV file
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        output_file = os.path.join(PROCESSED_DIR, 'preprocessed_indices_data.csv')
        indices_data.to_csv(output_file)
        logger.info(f"Preprocessed indices data saved to {output_file}")
    else:
        logger.error("Failed to load any indices data.")