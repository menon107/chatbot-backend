"""
Training script to process existing MongoDB Atlas data
This script will:
1. Connect to your existing MongoDB Atlas database
2. Read all data from your existing collections
3. Create indexes for efficient searching
4. Prepare the data for the chatbot
"""
from database import db
from config import Config
import logging
from pymongo.errors import OperationFailure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_collections():
    """Get list of all collections in the database"""
    try:
        database = db.get_database()
        collections = database.list_collection_names()
        logger.info(f"Found collections: {collections}")
        return collections
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        return []

def create_search_indexes(collection_name):
    """Create text search indexes on the collection"""
    try:
        collection = db.get_collection(collection_name)
        
        # Get sample document to understand structure
        sample = collection.find_one()
        if not sample:
            logger.warning(f"No documents found in collection: {collection_name}")
            return
        
        logger.info(f"Sample document structure: {list(sample.keys())[:10]}")
        
        # Try to create text index on common text fields
        text_fields = []
        for field in sample.keys():
            if isinstance(sample[field], str) and len(sample[field]) > 10:
                text_fields.append(field)
        
        if text_fields:
            # Create text index on multiple fields
            index_dict = {}
            for field in text_fields[:5]:  # Limit to first 5 text fields
                index_dict[field] = "text"
            
            try:
                collection.create_index(list(index_dict.items()))
                logger.info(f"Created text index on {collection_name} for fields: {text_fields[:5]}")
            except OperationFailure as e:
                logger.warning(f"Could not create text index: {e}")
                # Try individual text index
                try:
                    collection.create_index([(text_fields[0], "text")])
                    logger.info(f"Created text index on {collection_name} for field: {text_fields[0]}")
                except:
                    logger.warning(f"Could not create text index on {text_fields[0]}")
        
        # Create index on common fields like timestamp, date, _id
        for field in ['timestamp', 'created_at', 'date', 'updated_at']:
            if field in sample:
                try:
                    collection.create_index([(field, -1)])
                    logger.info(f"Created index on {field}")
                except:
                    pass
        
        logger.info(f"Indexes created successfully for {collection_name}")
        
    except Exception as e:
        logger.error(f"Error creating indexes for {collection_name}: {e}")

def get_collection_stats(collection_name):
    """Get statistics about a collection"""
    try:
        collection = db.get_collection(collection_name)
        count = collection.count_documents({})
        logger.info(f"Collection '{collection_name}': {count} documents")
        return count
    except Exception as e:
        logger.error(f"Error getting stats for {collection_name}: {e}")
        return 0

def train_on_existing_database(collection_name=None):
    """
    Train the chatbot on existing MongoDB data
    
    Args:
        collection_name: Specific collection to use. If None, uses all collections.
    """
    try:
        logger.info("Starting training on existing MongoDB Atlas database...")
        
        # Test connection
        db.connect()
        logger.info("✓ Successfully connected to MongoDB Atlas")
        
        # Get collections
        collections = get_all_collections()
        
        if not collections:
            logger.error("No collections found in the database!")
            return
        
        # Use specified collection or all collections
        if collection_name:
            if collection_name not in collections:
                logger.error(f"Collection '{collection_name}' not found!")
                logger.info(f"Available collections: {collections}")
                return
            collections_to_process = [collection_name]
        else:
            collections_to_process = collections
        
        # Process each collection
        total_documents = 0
        for coll_name in collections_to_process:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing collection: {coll_name}")
            logger.info(f"{'='*50}")
            
            # Get stats
            doc_count = get_collection_stats(coll_name)
            total_documents += doc_count
            
            # Create indexes
            create_search_indexes(coll_name)
            
            logger.info(f"✓ Collection '{coll_name}' processed successfully")
        
        logger.info(f"\n{'='*50}")
        logger.info("Training completed successfully!")
        logger.info(f"Total documents processed: {total_documents}")
        logger.info(f"Collections processed: {len(collections_to_process)}")
        logger.info(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    # Allow specifying collection name as argument
    collection_name = None
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
        logger.info(f"Processing specific collection: {collection_name}")
    
    train_on_existing_database(collection_name)

