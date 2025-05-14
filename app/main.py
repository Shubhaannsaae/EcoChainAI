import streamlit as st
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import dashboard
from app.dashboard.dashboard import setup_dashboard

def load_css(file_name):
    """Load a CSS file into the Streamlit app"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    """Main entry point for the Streamlit application"""

    st.set_page_config(
        page_title="AI Supply Chain Optimizer",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check if API keys are set
    required_apis = ["CLIMATIQ_API_KEY", "ANTHROPIC_API_KEY"]
    missing_apis = [api for api in required_apis if not os.getenv(api)]
    
    if missing_apis:
        st.error(f"Missing API keys: {', '.join(missing_apis)}. Please check your .env file.")
        st.info("You can still explore the app with sample data, but some features will be limited.")
    
    # Load custom CSS
    css_file = os.path.join(project_root, "app", "assets", "style.css")
    if os.path.exists(css_file):
        load_css(css_file)
        logger.info("Custom CSS loaded successfully.")
    else:
        logger.warning(f"Custom CSS file not found at {css_file}")

    # Load and display the dashboard
    setup_dashboard()

    logger.info("Application started successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
