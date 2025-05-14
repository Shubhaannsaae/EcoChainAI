"""
Configuration settings for the application.
Handles environment variables, API configurations, and constants.
"""

import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# API Keys
CLIMATIQ_API_KEY = os.getenv("CLIMATIQ_API_KEY")
CARBON_INTERFACE_API_KEY = os.getenv("CARBON_INTERFACE_API_KEY")
DITCHCARBON_API_KEY = os.getenv("DITCHCARBON_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API Endpoints
CLIMATIQ_BASE_URL = "https://api.climatiq.io"
EPA_BASE_URL = "https://data.epa.gov/efservice"
CARBON_INTERFACE_BASE_URL = "https://www.carboninterface.com/api/v1"
DITCHCARBON_BASE_URL = "https://api.ditchcarbon.com"
GLEIF_API_BASE_URL = "https://api.gleif.org/api/v1"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")  # Add this line



# Application Settings
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./supplychain.db")

# Constants
EMISSION_SCOPES = {
    "scope1": "Direct emissions from owned or controlled sources",
    "scope2": "Indirect emissions from purchased electricity, steam, heating, and cooling",
    "scope3": "All other indirect emissions in the value chain"
}

TRANSPORT_MODES = {
    "road": "Road transport (trucks, vans)",
    "rail": "Rail transport",
    "sea": "Sea transport (ships, boats)",
    "air": "Air transport (planes)"
}

INDUSTRY_SECTORS = [
    "Manufacturing", "Retail", "Food & Beverage", "Textiles", 
    "Electronics", "Construction", "Automotive", "Healthcare"
]

REGULATION_FRAMEWORKS = {
    "EU": ["REACH", "CSRD", "EU Green Deal"],
    "US": ["EPA", "TSCA", "Clean Air Act"],
    "Global": ["GHG Protocol", "ISO 14001", "Paris Agreement"]
}

# Setup logging
def get_logger(name):
    """Create a logger with the given name"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler('app.log')
        console_handler = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Set formatters
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
