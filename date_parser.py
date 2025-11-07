"""
Natural language date parsing utility
Converts various date formats to YYYY-MM-DD format
"""
import re
from datetime import datetime
from dateutil import parser as date_parser
import logging

logger = logging.getLogger(__name__)

def parse_natural_date(date_string: str) -> str:
    """
    Parse natural language date to YYYY-MM-DD format
    
    Supports formats like:
    - "1 november 2025"
    - "1st november 2025"
    - "november 1 2025"
    - "nov 1 2025"
    - "11/1/2025"
    - "2025-11-01"
    - "1-11-2025"
    - "01/11/2025"
    - etc.
    
    Returns:
        Date string in YYYY-MM-DD format or None if parsing fails
    """
    if not date_string:
        return None
    
    try:
        # First try structured format (YYYY-MM-DD)
        structured_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
        match = re.search(structured_pattern, date_string)
        if match:
            date_str = match.group(1)
            # Validate it's a valid date
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
    except:
        pass
    
    try:
        # Remove ordinal suffixes (1st, 2nd, 3rd, 4th, etc.)
        date_clean = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', date_string)
        
        # Try parsing with dateutil (handles many formats)
        parsed_date = date_parser.parse(date_clean, dayfirst=False, yearfirst=False)
        
        # Convert to YYYY-MM-DD format
        return parsed_date.strftime('%Y-%m-%d')
    except Exception as e:
        logger.warning(f"Could not parse date '{date_string}': {e}")
        return None

def extract_date_from_query(query: str) -> str:
    """
    Extract and parse date from a query string
    
    Looks for dates in various formats and converts them to YYYY-MM-DD
    
    Args:
        query: The query string that may contain a date
        
    Returns:
        Date string in YYYY-MM-DD format or None
    """
    if not query:
        return None
    
    # Common date patterns
    patterns = [
        # Structured format
        r'\b(\d{4}-\d{2}-\d{2})\b',
        # Date with month name (1 november 2025, 1st november 2025)
        r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})\b',
        # Month name with date (november 1 2025, nov 1 2025)
        r'\b((?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4})\b',
        # Slash format (11/1/2025, 1/11/2025)
        r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
        # Dash format (1-11-2025, 01-11-2025)
        r'\b(\d{1,2}-\d{1,2}-\d{4})\b',
        # Date with "on" keyword (on 1 november 2025)
        r'\bon\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})\b',
    ]
    
    # Try each pattern
    for pattern in patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            date_str = match.group(1) if match.groups() else match.group(0)
            parsed = parse_natural_date(date_str)
            if parsed:
                logger.info(f"Extracted date '{date_str}' -> '{parsed}' from query: {query}")
                return parsed
    
    return None

