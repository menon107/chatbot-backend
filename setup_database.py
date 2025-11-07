"""
Setup script to configure and test MongoDB Atlas connection
Run this first to verify your connection works!
"""
import os
from dotenv import load_dotenv
from database import db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    """Test MongoDB Atlas connection"""
    try:
        logger.info("Testing MongoDB Atlas connection...")
        logger.info("Make sure you have created a .env file with your credentials!")
        
        # Load environment variables
        load_dotenv()
        
        # Test connection
        client = db.connect()
        
        # Get database info
        database = db.get_database()
        db_name = database.name
        collections = database.list_collection_names()
        
        logger.info("✓ Connection successful!")
        logger.info(f"✓ Database: {db_name}")
        logger.info(f"✓ Collections found: {len(collections)}")
        
        if collections:
            logger.info(f"Collections: {', '.join(collections)}")
        else:
            logger.warning("No collections found. This is okay if you're starting fresh.")
        
        # Show sample data from first collection if available
        if collections:
            first_collection = collections[0]
            collection = db.get_collection(first_collection)
            sample = collection.find_one()
            
            if sample:
                logger.info(f"\nSample document from '{first_collection}':")
                logger.info(f"Fields: {', '.join(list(sample.keys())[:10])}")
                if len(sample.keys()) > 10:
                    logger.info(f"... and {len(sample.keys()) - 10} more fields")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")
        logger.error("\nPlease check:")
        logger.error("1. Your MONGODB_URI in .env file is correct")
        logger.error("2. Your username and password are correct")
        logger.error("3. Your IP address is whitelisted in MongoDB Atlas")
        logger.error("4. Your network allows connections to MongoDB Atlas")
        return False

if __name__ == "__main__":
    print("="*60)
    print("MongoDB Atlas Connection Test")
    print("="*60)
    
    if not os.path.exists('.env'):
        print("\n⚠ WARNING: .env file not found!")
        print("Please create a .env file with your MongoDB credentials.")
        print("You can copy .env.example to .env and fill in your details.")
        print("\nExample .env file should contain:")
        print("MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/...")
        print("DATABASE_NAME=your_database_name")
        print("EXISTING_COLLECTION_NAME=your_collection_name")
    else:
        success = test_connection()
        if success:
            print("\n" + "="*60)
            print("✓ Setup complete! You can now run train_on_existing_db.py")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("✗ Setup failed. Please fix the connection issues above.")
            print("="*60)

