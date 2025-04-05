import pandas as pd
import pickle
import os
import logging
from story_recommender import StoryRecommender

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def train_and_save_model(csv_file_path, pickle_file_path):
    """
    Train the recommendation model and save it to a pickle file.

    Args:
        csv_file_path: Path to the CSV file containing story data
        pickle_file_path: Path to save the pickle file
    """
    logger.info(f"Training model from {csv_file_path}")

    # Create and train the recommender
    recommender = StoryRecommender()
    recommender.load_data(csv_file_path)
    recommender.prepare_recommendation_system()

    # Save the model to a pickle file
    logger.info(f"Saving model to {pickle_file_path}")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(recommender, f)

    logger.info("Model saved successfully")
    return recommender


if __name__ == "__main__":
    # File paths
    csv_file_path = 'Stories  - Sheet1 (1).csv'
    pickle_file_path = 'story_recommender_model.pkl'

    # Train and save the model
    train_and_save_model(csv_file_path, pickle_file_path)