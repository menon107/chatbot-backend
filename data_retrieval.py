"""
Flexible data retrieval module that works with existing MongoDB collections
"""
from database import db
from config import Config
from date_parser import extract_date_from_query, parse_natural_date
from query_parser import extract_health_metric
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DataRetrieval:
    def __init__(self, collection_name=None):
        """
        Initialize data retrieval with existing collection
        
        Args:
            collection_name: Name of your existing collection. If None, uses config default.
        """
        self.collection_name = collection_name or Config.COLLECTION_DATA
        self.collection = db.get_collection(self.collection_name)
        logger.info(f"Using collection: {self.collection_name}")
    
    def get_all_data(self, limit=None, filter_query=None):
        """
        Get all data from the collection
        
        Args:
            limit: Maximum number of documents to return
            filter_query: MongoDB query filter (e.g., {"status": "active"})
        """
        try:
            query = filter_query or {}
            cursor = self.collection.find(query)
            
            if limit:
                cursor = cursor.limit(limit)
            
            return list(cursor)
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return []
    
    def get_latest_data(self, limit=100, sort_field=None):
        """
        Get latest data updates
        
        Args:
            limit: Maximum number of documents
            sort_field: Field to sort by (default: tries to find timestamp/date fields)
        """
        try:
            # Try to find common timestamp fields
            if not sort_field:
                sample = self.collection.find_one()
                if sample:
                    for field in ['timestamp', 'created_at', 'date', 'updated_at', '_id']:
                        if field in sample:
                            sort_field = field
                            break
            
            if sort_field:
                cursor = self.collection.find({}).sort(sort_field, -1).limit(limit)
            else:
                cursor = self.collection.find({}).limit(limit)
            
            return list(cursor)
        except Exception as e:
            logger.error(f"Error retrieving latest data: {e}")
            return []
    
    def search_data(self, search_query, limit=10, search_fields=None):
        """
        Search data in the collection
        
        Args:
            search_query: Search text
            limit: Maximum results
            search_fields: List of fields to search in (if None, uses text index)
        """
        try:
            # First, try to parse date-based queries (e.g., "systolic on date 2025-11-05")
            date_query = self._parse_date_query(search_query)
            if date_query:
                results = self.collection.find(date_query).limit(limit)
                result_list = list(results)
                if result_list:
                    return result_list
            
            # Try text search first
            try:
                results = self.collection.find(
                    {"$text": {"$search": search_query}},
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]).limit(limit)
                return list(results)
            except:
                # Fallback to regex search on common text fields
                if search_fields:
                    query = {"$or": [{field: {"$regex": search_query, "$options": "i"}} 
                                   for field in search_fields]}
                else:
                    # Try to find text fields dynamically
                    sample = self.collection.find_one()
                    if sample:
                        text_fields = [k for k, v in sample.items() 
                                     if isinstance(v, str) and len(v) > 10]
                        if text_fields:
                            query = {"$or": [{field: {"$regex": search_query, "$options": "i"}} 
                                           for field in text_fields[:5]]}
                        else:
                            query = {"$text": {"$search": search_query}}  # Will fail but try
                    else:
                        query = {"$text": {"$search": search_query}}
                
                results = self.collection.find(query).limit(limit)
                return list(results)
                
        except Exception as e:
            logger.warning(f"Text search failed, trying regex: {e}")
            # Last resort: search in _id or any string field
            try:
                results = self.collection.find({
                    "$or": [
                        {"_id": {"$regex": search_query, "$options": "i"}}
                    ]
                }).limit(limit)
                return list(results)
            except Exception as e2:
                logger.error(f"Error searching data: {e2}")
                return []
    
    def _parse_date_query(self, search_query: str):
        """
        Parse date-based queries like "systolic on date 2025-11-05" or "systolic on 1st november 2025"
        Returns MongoDB query or None
        """
        import re
        
        query_lower = search_query.lower()
        
        # Extract date in any format (natural language or structured)
        date_str = extract_date_from_query(search_query)
        
        if not date_str:
            return None
        
        # Find which metric is being asked for using natural language parser
        metric_field = extract_health_metric(search_query)
        
        # Build query for nested bpReadings array
        # Search for documents where bpReadings array contains an element with matching date
        if metric_field:
            # Query for the specific metric on the specific date
            query = {
                "bpReadings": {
                    "$elemMatch": {
                        "date": date_str,
                        metric_field: {"$exists": True}
                    }
                }
            }
        else:
            # Just search for the date
            query = {
                "bpReadings": {
                    "$elemMatch": {
                        "date": date_str
                    }
                }
            }
        
        logger.info(f"Parsed date query: {query}")
        return query
    
    def query_by_date_and_metric(self, date: str, metric: str = None, email: str = None):
        """
        Query data by specific date and optionally by metric, filtered by patient email if provided
        
        Args:
            date: Date in any format (will be parsed to YYYY-MM-DD)
            metric: Optional metric name (systolic, diastolic, etc.)
            email: Optional patient email to filter results
        """
        try:
            # Parse date to YYYY-MM-DD format if needed
            if date:
                parsed_date = parse_natural_date(date)
                if parsed_date:
                    date = parsed_date
                else:
                    logger.warning(f"Could not parse date: {date}")
                    return []
            metric_map = {
                'systolic': 'systolic',
                'diastolic': 'diastolic',
                'pulse': 'pulse',
                'bmi': 'bmi',
                'blood sugar': 'fastingBloodSugar',
                'bloodsugar': 'fastingBloodSugar',
                'cholesterol': 'totalCholesterol',
                'waist': 'waistCircumference',
                'sleep': 'sleepQuality',
                'stress': 'stressLevel'
            }
            
            metric_field = None
            if metric:
                metric_field = metric_map.get(metric.lower())
            
            if metric_field:
                query = {
                    "bpReadings": {
                        "$elemMatch": {
                            "date": date,
                            metric_field: {"$exists": True}
                        }
                    }
                }
            else:
                query = {
                    "bpReadings": {
                        "$elemMatch": {
                            "date": date
                        }
                    }
                }
            
            # Add email filter if provided
            if email:
                query["email"] = email
                logger.info(f"Filtering date query by email: {email}")
            
            results = self.collection.find(query)
            return list(results)
        except Exception as e:
            logger.error(f"Error querying by date and metric: {e}")
            return []
    
    def get_collection_info(self):
        """Get information about the collection structure"""
        try:
            sample = self.collection.find_one()
            if sample:
                return {
                    "collection_name": self.collection_name,
                    "sample_fields": list(sample.keys()),
                    "total_documents": self.collection.count_documents({})
                }
            return None
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None

