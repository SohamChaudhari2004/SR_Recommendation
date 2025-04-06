from flask import Flask, request, jsonify
import pickle
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(_name_)

# Create Flask app
app = Flask(_name_)

# Global recommender instance
recommender = None


@app.route('/api/recommend', methods=['GET', 'POST'])
def recommend():
    """API endpoint to get story recommendations."""
    if request.method == 'POST':
        # Handle POST request with JSON data
        if request.is_json:
            data = request.get_json()
            query = data.get('query', '')
        # Handle POST request with form data
        else:
            query = request.form.get('query', '')
    else:
        # Keep the original GET parameter handling as fallback
        query = request.args.get('query', '')

    top_n = int(request.args.get('top_n', 5))

    if not query:
        return jsonify({
            "status": "error",
            "message": "Missing 'query' parameter",
            "recommendations": []
        }), 400

    if not recommender:
        return jsonify({
            "status": "error",
            "message": "Recommendation system not initialized",
            "recommendations": []
        }), 500

    result = recommender.api_recommend_stories(query, top_n)

    if result["status"] == "error":
        return jsonify(result), 500
    elif result["status"] == "not_found":
        return jsonify(result), 404
    else:
        return jsonify(result), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint to check if the service is running."""
    if recommender is None:
        return jsonify({"status": "error", "message": "Recommendation system not initialized"}), 500

    return jsonify({
        "status": "ok",
        "message": "Service is running",
        "stories_count": len(recommender.df) if recommender.df is not None else 0,
        "model_type": type(recommender)._name_
    }), 200


def load_recommender(pickle_file_path):
    """Load the recommendation system from a pickle file."""
    global recommender

    try:
        logger.info(f"Loading recommendation model from {pickle_file_path}")
        with open(pickle_file_path, 'rb') as f:
            recommender = pickle.load(f)
        logger.info("Recommendation system loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading recommendation system: {e}")
        return False


def train_recommender(csv_file_path):
    """Train the recommendation system from scratch."""
    global recommender

    try:
        from story_recommender import StoryRecommender

        logger.info(f"Training recommendation model from {csv_file_path}")
        recommender = StoryRecommender()
        recommender.load_data(csv_file_path)
        recommender.prepare_recommendation_system()
        logger.info("Recommendation system trained successfully")
        return True
    except Exception as e:
        logger.error(f"Error training recommendation system: {e}")
        return False


def run_api(pickle_file_path=None, csv_file_path=None, host='0.0.0.0', port=5000):
    """
    Run the Flask API server.

    Args:
        pickle_file_path: Path to the pickle file containing the trained model
        csv_file_path: Path to the CSV file (as fallback if pickle loading fails)
        host: Host address to bind the server to
        port: Port to run the server on
    """
    # Try to load model from pickle first
    if pickle_file_path and os.path.exists(pickle_file_path):
        success = load_recommender(pickle_file_path)
        if not success and csv_file_path:
            # Fall back to training from CSV if pickle loading fails
            logger.warning("Falling back to training from CSV")
            success = train_recommender(csv_file_path)
    elif csv_file_path:
        # Train from CSV if no pickle file specified
        success = train_recommender(csv_file_path)
    else:
        logger.error("Neither pickle file nor CSV file provided")
        success = False

    if success:
        # Run Flask app
        logger.info(f"Starting API server on {host}:{port}")
        app.run(host=host, port=port)
    else:
        logger.error("Failed to initialize recommendation system. API not started.")


if _name_ == "_main_":
    # File paths - modify these to match your setup
    pickle_file_path = 'story_recommender_model.pkl'
    csv_file_path = 'Stories  - Sheet1 (1).csv'  # Fallback if pickle loading fails

    # Run the API
    run_api(pickle_file_path, csv_file_path)
