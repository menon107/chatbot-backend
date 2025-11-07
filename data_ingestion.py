"""
Real-time data ingestion module for storing updates in MongoDB
"""
from datetime import datetime
from database import db
from config import Config
import logging

logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self):
        self.collection = db.get_collection(Config.COLLECTION_DATA)
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for efficient querying"""
        try:
            # Index on timestamp for time-based queries
            self.collection.create_index([("timestamp", -1)])
            # Index on data_type for filtering
            self.collection.create_index([("data_type", 1)])
            # Text index for search
            self.collection.create_index([("content", "text")])
            logger.info("Indexes created successfully")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    def store_data(self, data_type, content, metadata=None):
        """
        Store real-time data update in MongoDB
        
        Args:
            data_type: Type of data (e.g., 'news', 'update', 'document', etc.)
            content: The actual data content
            metadata: Additional metadata dictionary
        """
        try:
            document = {
                "data_type": data_type,
                "content": content,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {},
                "processed": False  # Flag for embedding processing
            }
            
            result = self.collection.insert_one(document)
            logger.info(f"Data stored successfully with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            raise
    
    def get_latest_data(self, limit=100, data_type=None):
        """Get the latest data updates"""
        try:
            query = {}
            if data_type:
                query["data_type"] = data_type
            
            cursor = self.collection.find(query).sort("timestamp", -1).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error retrieving latest data: {e}")
            return []
    
    def search_data(self, search_query, limit=10):
        """Search data using text search"""
        try:
            results = self.collection.find(
                {"$text": {"$search": search_query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            return list(results)
        except Exception as e:
            logger.error(f"Error searching data: {e}")
            return []

