# story_recommender.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import string
import logging
import time
from typing import List, Dict, Tuple, Optional, Union
import difflib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class StoryRecommender:
    """A recommendation system for stories based on content similarity and other features."""

    def __init__(self, csv_file_path: str = None):
        """Initialize the recommendation system.

        Args:
            csv_file_path: Path to the CSV file containing story data
        """
        self.df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.tfidf_vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.knn_model = None

        # Download NLTK resources
        self._download_nltk_resources()

        # Load data if path is provided
        if csv_file_path:
            self.load_data(csv_file_path)

    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        logger.info("Downloading NLTK resources...")
        nltk_resources = ['punkt', 'stopwords', 'wordnet']

        for resource in nltk_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else resource)
                logger.info(f"Resource '{resource}' already downloaded.")
            except LookupError:
                logger.info(f"Downloading '{resource}'...")
                nltk.download(resource, quiet=True)

    def load_data(self, csv_file_path: str) -> pd.DataFrame:
        """Load story data from CSV file.

        Args:
            csv_file_path: Path to the CSV file

        Returns:
            DataFrame containing story data
        """
        try:
            start_time = time.time()
            df = pd.read_csv(csv_file_path)
            load_time = time.time() - start_time
            logger.info(f"Successfully loaded {len(df)} stories from {csv_file_path} in {load_time:.2f} seconds")

            # Verify and standardize column names
            self._standardize_columns(df)

            # Ensure required columns exist
            required_columns = ['Title', 'Category', 'Subcategory', 'Story content']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                logger.info(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Remove duplicates if any
            original_rows = len(df)
            df = df.drop_duplicates(subset=['Title', 'Story content'])
            if len(df) < original_rows:
                logger.info(f"Removed {original_rows - len(df)} duplicate rows.")

            # Handle missing values
            df['Story content'] = df['Story content'].fillna('')
            df['Category'] = df['Category'].fillna('Uncategorized')
            df['Subcategory'] = df['Subcategory'].fillna('Uncategorized')

            # Generate URLs for stories with empty URLs
            self._generate_urls(df)

            # Create an ID column if not present
            if 'ID' not in df.columns:
                df['ID'] = range(1, len(df) + 1)

            self.df = df
            return df

        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def _standardize_columns(self, df: pd.DataFrame) -> None:
        """Standardize column names in the DataFrame.

        Args:
            df: DataFrame to standardize
        """
        # Define expected columns and potential variations
        column_mapping = {
            'title': 'Title',
            'story title': 'Title',
            'name': 'Title',
            'category': 'Category',
            'genre': 'Category',
            'type': 'Category',
            'subcategory': 'Subcategory',
            'subgenre': 'Subcategory',
            'subtype': 'Subcategory',
            'url': 'Url',
            'link': 'Url',
            'language': 'Language',
            'lang': 'Language',
            'story content': 'Story content',
            'content': 'Story content',
            'text': 'Story content',
            'story': 'Story content',
            'story text': 'Story content',
            'id': 'ID',
            'story_id': 'ID',
            'story id': 'ID'
        }

        # Standardize column names (case-insensitive)
        rename_dict = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in column_mapping:
                rename_dict[col] = column_mapping[col_lower]

        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)
            logger.info(f"Standardized column names: {rename_dict}")

    def _generate_urls(self, df: pd.DataFrame) -> None:
        """Generate URLs from story titles if URL column is empty.

        Args:
            df: DataFrame containing story data
        """
        if 'Url' not in df.columns:
            # Create Url column as string type to avoid dtype issues
            df['Url'] = ''
        else:
            # Ensure Url column is string type
            df['Url'] = df['Url'].astype(str)

        # Find rows with empty URLs
        empty_urls = (df['Url'].isna()) | (df['Url'] == '') | (df['Url'] == 'nan')

        if empty_urls.any():
            logger.info(f"Generating URLs for {empty_urls.sum()} stories with empty URLs...")

            # Convert titles to URL-friendly format
            for idx in df[empty_urls].index:
                title = df.loc[idx, 'Title']
                if isinstance(title, str):
                    # Remove special characters and convert spaces to hyphens
                    url = re.sub(r'[^\w\s-]', '', title.lower())
                    url = re.sub(r'[\s]+', '-', url)
                    df.loc[idx, 'Url'] = url

            logger.info("URLs generated successfully")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data.

        Args:
            text: Text to preprocess

        Returns:
            Preprocessed text
        """
        # Handle NaN or non-string values
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback to simple splitting
            tokens = text.split()
            logger.warning("Using fallback tokenization method")

        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        except LookupError:
            # Fallback: use a small set of common stopwords
            common_stopwords = {'the', 'a', 'an', 'and', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was',
                                'were'}
            tokens = [word for word in tokens if word not in common_stopwords]
            logger.warning("Using fallback stopwords list")

        # Lemmatization (better than stemming)
        try:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}")
            # Skip lemmatization if there's an issue

        return ' '.join(tokens)

    def extract_features(self, max_features: int = 5000) -> None:
        """Extract features from preprocessed text using TF-IDF.

        Args:
            max_features: Maximum number of features to extract
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Preprocessing text data...")
        self.df['processed_content'] = self.df['Story content'].apply(self.preprocess_text)

        logger.info(f"Extracting features using TF-IDF (max_features={max_features})...")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['processed_content'])

        logger.info(f"Created TF-IDF matrix with shape {self.tfidf_matrix.shape}")

        # Create a reverse mapping of titles and indices
        self.indices = pd.Series(self.df.index, index=self.df['Title'])

        # Extract keywords
        self._extract_keywords()

        # Build KNN model for faster similarity search
        self._build_knn_model()

    def _build_knn_model(self, n_neighbors: int = 15) -> None:
        """Build a K-nearest neighbors model for faster similarity search.

        Args:
            n_neighbors: Number of neighbors to find
        """
        logger.info(f"Building KNN model with {n_neighbors} neighbors...")
        self.knn_model = NearestNeighbors(
            n_neighbors=min(n_neighbors + 1, len(self.df)),  # +1 to account for self-similarity
            algorithm='auto',
            metric='cosine'
        )
        self.knn_model.fit(self.tfidf_matrix)
        logger.info("KNN model built successfully")

    def compute_similarity_matrix(self) -> None:
        """Compute cosine similarity matrix between all stories."""
        if self.tfidf_matrix is None:
            raise ValueError("No features extracted. Call extract_features() first.")

        logger.info("Computing cosine similarity matrix...")
        start_time = time.time()

        # For larger datasets, we'll use the KNN model instead of full similarity matrix
        if len(self.df) > 1000:
            logger.info("Large dataset detected, skipping full similarity matrix computation")
            # We'll compute similarities on-demand using the KNN model
            self.cosine_sim = None
        else:
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        computation_time = time.time() - start_time
        logger.info(f"Similarity computation completed in {computation_time:.2f} seconds")

    def _extract_keywords(self, top_n: int = 10) -> None:
        """Extract top keywords from each story using TF-IDF scores.

        Args:
            top_n: Number of top keywords to extract
        """
        logger.info(f"Extracting top {top_n} keywords from each story...")

        # Get feature names from the TF-IDF vectorizer
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        # Extract keywords for each story
        keywords_list = []

        for i in range(len(self.df)):
            # Get TF-IDF scores for this document
            tfidf_scores = self.tfidf_matrix[i].toarray()[0]

            # Create a dictionary of words and their TF-IDF scores
            word_scores = {feature_names[j]: tfidf_scores[j] for j in range(len(feature_names)) if tfidf_scores[j] > 0}

            # Sort words by TF-IDF score
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

            # Get top N keywords
            top_keywords = [word for word, score in sorted_words[:top_n]]
            keywords_list.append(top_keywords)

        self.df['keywords'] = keywords_list
        logger.info("Keyword extraction completed")

    def get_hybrid_recommendations(self, title: str, weight_content: float = 0.6,
                                   weight_category: float = 0.3, weight_keyword: float = 0.1,
                                   top_n: int = 5) -> pd.DataFrame:
        """Get recommendations based on a hybrid approach combining multiple signals.

        Args:
            title: Title of the story to get recommendations for
            weight_content: Weight for content similarity
            weight_category: Weight for category matching
            weight_keyword: Weight for keyword similarity
            top_n: Number of recommendations to return

        Returns:
            DataFrame containing recommended stories with hybrid scores
        """
        if title not in self.indices:
            return pd.DataFrame({"Error": ["Story not found in the database."]})

        # Get index of story that matches title
        idx = self.indices[title]

        # Get content similarity scores
        if self.cosine_sim is not None:
            content_sim_scores = list(enumerate(self.cosine_sim[idx]))
        else:
            # Use KNN model
            distances, indices = self.knn_model.kneighbors(self.tfidf_matrix[idx].reshape(1, -1))
            similarities = 1 - distances.flatten()
            # Create a list of tuples (index, similarity)
            content_sim_scores = [(indices.flatten()[i], similarities[i]) for i in range(len(indices.flatten()))]

        # Get category and subcategory of the selected story
        selected_category = self.df.loc[idx, 'Category']
        selected_subcategory = self.df.loc[idx, 'Subcategory']

        # Get keywords of the selected story
        story_keywords = set(self.df.loc[idx, 'keywords'])

        # Calculate hybrid scores
        hybrid_scores = []
        for i in range(len(self.df)):
            if i == idx:  # Skip the story itself
                continue

            # Content similarity
            content_score = next((score for index, score in content_sim_scores if index == i), 0)

            # Category similarity
            if self.df.loc[i, 'Subcategory'] == selected_subcategory:
                category_score = 1.0  # Exact subcategory match
            elif self.df.loc[i, 'Category'] == selected_category:
                category_score = 0.7  # Category match
            else:
                category_score = 0.0  # No match

            # Keyword similarity
            other_keywords = set(self.df.loc[i, 'keywords'])
            intersection = len(story_keywords.intersection(other_keywords))
            union = len(story_keywords.union(other_keywords))
            keyword_score = intersection / union if union > 0 else 0

            # Calculate hybrid score
            hybrid_score = (
                    weight_content * content_score +
                    weight_category * category_score +
                    weight_keyword * keyword_score
            )

            hybrid_scores.append((i, hybrid_score))

        # Sort by hybrid score
        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)

        # Get top N
        hybrid_scores = hybrid_scores[:top_n]

        # Get story indices and scores
        story_indices = [i[0] for i in hybrid_scores]
        hybrid_values = [i[1] for i in hybrid_scores]

        # Create result dataframe
        result = self.df.iloc[story_indices][['ID', 'Title', 'Category', 'Subcategory']].copy()
        result['Hybrid Score'] = hybrid_values

        return result

    def find_closest_story(self, query: str, cutoff: float = 0.4) -> Optional[str]:
        """Find the closest matching story based on query, searching through titles,
        categories, subcategories, and story content.

        Args:
            query: Search query (title, category, subcategory, or keywords in content)
            cutoff: Minimum similarity score to consider a match (0-1)

        Returns:
            Closest matching story title or None if no match found
        """
        if self.df is None or len(self.df) == 0:
            return None

        # Get all story titles
        all_titles = self.df['Title'].tolist()

        # First check for exact match in titles
        exact_matches = [title for title in all_titles if query.lower() == title.lower()]
        if exact_matches:
            return exact_matches[0]

        # Check for contains match in titles
        contains_matches = [title for title in all_titles if query.lower() in title.lower()]
        if contains_matches:
            return contains_matches[0]  # Return first containing match

        # Check for category/subcategory match
        category_matches = self.df[
            (self.df['Category'].str.lower() == query.lower()) |
            (self.df['Subcategory'].str.lower() == query.lower())
            ]

        if not category_matches.empty:
            # Return the first story from the matching category
            return category_matches['Title'].iloc[0]

        # Check for partial category/subcategory match
        partial_category_matches = self.df[
            (self.df['Category'].str.lower().str.contains(query.lower())) |
            (self.df['Subcategory'].str.lower().str.contains(query.lower()))
            ]

        if not partial_category_matches.empty:
            # Return the first story from the partially matching category
            return partial_category_matches['Title'].iloc[0]

        # Check for content match
        content_matches = self.df[self.df['Story content'].str.lower().str.contains(query.lower(), na=False)]

        if not content_matches.empty:
            # Return the first story with matching content
            return content_matches['Title'].iloc[0]

        # Use TF-IDF vector matching
        if hasattr(self, 'tfidf_vectorizer') and hasattr(self, 'tfidf_matrix') and self.tfidf_vectorizer is not None:
            # Preprocess the query the same way we preprocessed story content
            processed_query = self.preprocess_text(query)

            # Transform the query to the same vector space as the stories
            query_vector = self.tfidf_vectorizer.transform([processed_query])

            # Calculate similarity between query and all stories
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

            # Get the index of the most similar story
            most_similar_idx = similarities.argmax()

            # Only consider it a match if similarity is above threshold
            if similarities[most_similar_idx] > cutoff:
                return self.df.iloc[most_similar_idx]['Title']

        # Use fuzzy matching for approximate matches in titles
        matches = difflib.get_close_matches(query, all_titles, n=1, cutoff=cutoff)
        if matches:
            return matches[0]

        return None

    def api_recommend_stories(self, query: str, top_n: int = 5) -> Dict:
        """API endpoint to get story recommendations based on a search query.

        Args:
            query: Search query or partial title
            top_n: Number of recommendations to return

        Returns:
            Dictionary with recommendations and status information
        """
        if self.df is None:
            return {
                "status": "error",
                "message": "No data loaded",
                "recommendations": []
            }

        # Find the closest story title match
        closest_title = self.find_closest_story(query)

        if not closest_title:
            return {
                "status": "not_found",
                "message": f"No story found matching '{query}'",
                "recommendations": []
            }

        # Get recommendations using the hybrid method
        recommendations = self.get_hybrid_recommendations(closest_title, top_n=top_n)

        if 'Error' in recommendations.columns:
            return {
                "status": "error",
                "message": recommendations['Error'].iloc[0],
                "recommendations": []
            }

        # Format results as dictionary
        result = {
            "status": "success",
            "query": query,
            "matched_story": closest_title,
            "matched_story_id": int(self.df.loc[self.indices[closest_title], 'ID']),
            "recommendations": []
        }

        # Add recommendations
        for _, row in recommendations.iterrows():
            rec = {
                "id": int(row['ID']),
                "title": row['Title'],
                "category": row['Category'],
                "subcategory": row['Subcategory'],
                "score": float(row['Hybrid Score'])
            }
            result["recommendations"].append(rec)

        return result

    def prepare_recommendation_system(self) -> None:
        """Prepare the recommendation system by extracting features and computing similarities."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Extract features and compute similarities
        self.extract_features()
        self.compute_similarity_matrix()

        logger.info("Recommendation system is ready to use")