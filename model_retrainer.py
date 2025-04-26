import time
import logging
import pickle
import os
import pandas as pd
import requests
from pymongo import MongoClient
from story_recommender import StoryRecommender

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
MONGO_URI = "mongodb+srv://user1234:user1234@cluster0.yl4cb.mongodb.net/"
DB_NAME = "sr_echo_tales"  # Your MongoDB database name
COLLECTION_NAME = "stories"  # Your collection name for stories
MODEL_PATH = "story_recommender_model.pkl"  # Path to save the model
RETRAIN_THRESHOLD = 20  # Number of new entries to trigger retraining
CHECK_INTERVAL = 3600  # Check for updates every hour (in seconds)

class ModelTrainer:
    def __init__(self, mongo_uri, db_name, collection_name, model_path, retrain_threshold):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.model_path = model_path
        self.retrain_threshold = retrain_threshold
        self.last_count = 0
        self.client = None
        self.db = None
        self.collection = None

    def connect_to_mongodb(self):
        """Establish connection to MongoDB Atlas."""
        try:
            # Connect to MongoDB Atlas with a timeout to prevent hanging
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            # Test the connection
            self.client.server_info()

            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info(f"Connected to MongoDB Atlas: {self.db_name}.{self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            return False

    def check_for_updates(self):
        """Check if there are enough new entries to trigger retraining."""
        try:
            current_count = self.collection.count_documents({})

            if self.last_count == 0:
                # First run, just store the count
                self.last_count = current_count
                logger.info(f"Initial collection count: {current_count}")
                return False

            new_entries = current_count - self.last_count
            logger.info(f"Found {new_entries} new entries since last check")

            if new_entries >= self.retrain_threshold:
                self.last_count = current_count
                return True

            return False
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return False

    def convert_mongo_to_dataframe(self):
        """Convert MongoDB collection to pandas DataFrame."""
        try:
            # Get all documents from collection
            cursor = self.collection.find({})

            # Convert cursor to list of dictionaries
            data = list(cursor)

            if not data:
                logger.warning("No data found in MongoDB collection")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)
            # Drop rows where 'Story content' is missing or empty
            """
            df = df.dropna(subset=['Story content'])
            df = df[df['Story content'].str.strip() != ""]

            if len(df) == 0:
                logger.error("No valid 'Story content' found after cleaning. Aborting retraining.")
                return None
            """
            # MongoDB _id is not serializable, so we'll drop it
            if '_id' in df.columns:
                df['ID'] = df['_id'].astype(str)  # Convert MongoDB _id to string ID
                df = df.drop('_id', axis=1)

            # Check for column name differences and standardize
            # Mapping of possible MongoDB field names to required column names
            field_mapping = {
                'title': 'Title',
                'storyTitle': 'Title',
                'category': 'Category',
                'storyCategory': 'Category',
                'subcategory': 'Subcategory',
                'storySubcategory': 'Subcategory',
                'content': 'Story content',
                'storyContent': 'Story content',
                'text': 'Story content',
                'story': 'Story content',
                'script': 'Story content'
            }

            # Rename columns if needed
            for mongo_field, std_column in field_mapping.items():
                if mongo_field in df.columns and std_column not in df.columns:
                    df.rename(columns={mongo_field: std_column}, inplace=True)

            # Ensure required columns exist
            required_columns = ['Title', 'Category', 'Subcategory', 'Story content']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.warning(f"Missing required columns in MongoDB data: {missing_columns}")
                # Create empty columns for missing fields
                for col in missing_columns:
                    df[col] = ""

            logger.info(f"Successfully converted {len(df)} documents to DataFrame")
            return df
        except Exception as e:
            logger.error(f"Error converting MongoDB to DataFrame: {e}")
            return None
    #
    # def retrain_model(self):
    #     """Retrain the recommendation model with new data."""
    #     try:
    #         # Convert MongoDB data to DataFrame
    #         df = self.convert_mongo_to_dataframe()
    #
    #         if df is None or len(df) == 0:
    #             logger.error("Cannot retrain model: No valid data available")
    #             return False
    #
    #         # Initialize and train recommender
    #         recommender = StoryRecommender()
    #
    #         # We'll manually set the DataFrame instead of loading from CSV
    #         recommender.df = df
    #
    #         # Prepare the recommendation system
    #         recommender.prepare_recommendation_system()
    #
    #         # Save the trained model - Use a temporary file for safety
    #         temp_model_path = f"{self.model_path}.temp"
    #         with open(temp_model_path, 'wb') as f:
    #             pickle.dump(recommender, f)
    #
    #         # Replace the old model file with the new one
    #         if os.path.exists(self.model_path):
    #             os.remove(self.model_path)
    #         os.rename(temp_model_path, self.model_path)
    #
    #         logger.info(f"Model successfully retrained and saved to {self.model_path}")
    #
    #         # Optional: Save DataFrame to CSV as backup
    #         csv_backup_path = f"stories_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    #         df.to_csv(csv_backup_path, index=False)
    #         logger.info(f"Backup CSV saved to {csv_backup_path}")
    #
    #         return True
    #     except Exception as e:
    #         logger.error(f"Error retraining model: {e}")
    #         return False

    def retrain_model(self):
        """Retrain the recommendation model with new data."""
        try:
            # Convert MongoDB data to DataFrame
            df = self.convert_mongo_to_dataframe()

            if df is None or len(df) == 0:
                logger.error("Cannot retrain model: No valid data available")
                return False

            # Initialize and train recommender
            recommender = StoryRecommender()

            # We'll manually set the DataFrame instead of loading from CSV
            recommender.df = df

            # Prepare the recommendation system
            recommender.prepare_recommendation_system()

            # Save the trained model - Use a temporary file for safety
            temp_model_path = f"{self.model_path}.temp"
            with open(temp_model_path, 'wb') as f:
                pickle.dump(recommender, f)

            # Replace the old model file with the new one
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            os.rename(temp_model_path, self.model_path)

            logger.info(f"Model successfully retrained and saved to {self.model_path}")

            # Optional: Save DataFrame to CSV as backup
            csv_backup_path = f"stories_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_backup_path, index=False)
            logger.info(f"Backup CSV saved to {csv_backup_path}")

            # Trigger model reload in API
            try:
                import requests
                api_url = "https://ec2-15-206-194-179.ap-south-1.compute.amazonaws.com:5000/api/reload"  # Adjust URL if needed
                response = requests.post(api_url)
                if response.status_code == 200:
                    logger.info("API successfully reloaded the model after retraining")
                else:
                    logger.warning(f"API reload returned non-200: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Failed to trigger model reload via API: {e}")
                logger.info("API reload request failed, but model retraining was successful")

            return True
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return False

    def run_monitoring_loop(self):
        """Main loop to monitor database and retrain as needed."""
        logger.info("Starting MongoDB Atlas monitoring and model retraining service")

        if not self.connect_to_mongodb():
            logger.error("Failed to connect to MongoDB Atlas. Exiting.")
            return

        while True:
            try:
                # Check if we need to retrain
                if self.check_for_updates():
                    logger.info(f"Detected {self.retrain_threshold}+ new entries. Retraining model...")
                    success = self.retrain_model()
                    if success:
                        logger.info("Model retraining completed successfully")
                    else:
                        logger.error("Model retraining failed")

                # Wait before checking again
                logger.info(f"Sleeping for {CHECK_INTERVAL} seconds before next check")
                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                logger.info("Monitoring interrupted. Exiting gracefully.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in monitoring loop: {e}")
                # If connection to MongoDB is lost, try to reconnect
                try:
                    self.connect_to_mongodb()
                except:
                    pass
                logger.info(f"Trying again in {CHECK_INTERVAL} seconds")
                time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    trainer = ModelTrainer(
        mongo_uri=MONGO_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        model_path=MODEL_PATH,
        retrain_threshold=RETRAIN_THRESHOLD
    )

    trainer.run_monitoring_loop()