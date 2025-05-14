"""
API client wrapper functions for external services used by the Supply Chain Optimizer
"""

import requests
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from app.config import (
    CLIMATIQ_API_KEY, CARBON_INTERFACE_API_KEY,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY, CLIMATIQ_BASE_URL, EPA_BASE_URL,
    CARBON_INTERFACE_BASE_URL, OPENAI_BASE_URL,
    DITCHCARBON_API_KEY, DITCHCARBON_BASE_URL, GLEIF_API_BASE_URL,
    get_logger
)

logger = get_logger(__name__)

class APIError(Exception):
    """Custom exception for API errors"""
    pass

class RateLimitError(APIError):
    """Exception raised when API rate limit is reached"""
    pass

def _handle_response(response, service_name):
    """
    Handle API response and raise appropriate exceptions
    
    Args:
        response: requests.Response object
        service_name: Name of the API service for logging
        
    Returns:
        dict: Parsed JSON response
        
    Raises:
        RateLimitError: If rate limit is exceeded
        APIError: For other API errors
    """
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            logger.warning(f"{service_name} API rate limit exceeded")
            raise RateLimitError(f"{service_name} rate limit exceeded")
        
        error_msg = f"{service_name} API error: {str(e)}"
        try:
            error_detail = response.json()
            error_msg += f" - Details: {json.dumps(error_detail)}"
        except:
            error_msg += f" - Status code: {response.status_code}"
        
        logger.error(error_msg)
        raise APIError(error_msg)
    except requests.exceptions.RequestException as e:
        logger.error(f"{service_name} API request failed: {str(e)}")
        raise APIError(f"{service_name} request failed: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"{service_name} API returned invalid JSON: {str(e)}")
        raise APIError(f"{service_name} returned invalid JSON: {str(e)}")

# CLIMATIQ API FUNCTIONS
def calculate_emission(activity_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate emissions using the Climatiq API
    
    Args:
        activity_type: Type of activity (e.g., 'transportation', 'electricity')
        params: Activity parameters (e.g., {'distance': 100, 'distance_unit': 'km'})
        
    Returns:
        dict: Emission calculation result
    """
    if not CLIMATIQ_API_KEY:
        logger.warning("CLIMATIQ_API_KEY not set, using mock data")
        # Return mock data for demo purposes
        return {
            "co2e": round(params.get('distance', 100) * 0.12, 2),
            "co2e_unit": "kg",
            "co2e_calculation_method": "mock",
            "co2e_calculation_origin": "mock_data",
            "emission_factor": {
                "activity_id": activity_type,
                "source": "Climatiq Mock",
                "year": datetime.now().year
            }
        }
    
    url = f"{CLIMATIQ_BASE_URL}/estimate"
    headers = {
        "Authorization": f"Bearer {CLIMATIQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "emission_factor": {
            "activity_id": activity_type
        },
        "parameters": params
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        return _handle_response(response, "Climatiq")
    except Exception as e:
        logger.error(f"Error calculating emissions: {str(e)}")
        raise

def get_emission_factors(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get emission factors from Climatiq API
    
    Args:
        category: Optional category filter
        
    Returns:
        list: List of emission factors
    """
    if not CLIMATIQ_API_KEY:
        logger.warning("CLIMATIQ_API_KEY not set, using mock data")
        # Return mock data for demo purposes
        return [
            {
                "activity_id": "transportation_road_hgv",
                "name": "Heavy Goods Vehicle (HGV)",
                "category": "transport",
                "unit": "km",
                "co2e_factor": 0.12
            },
            {
                "activity_id": "transportation_sea_cargo",
                "name": "Sea Freight",
                "category": "transport",
                "unit": "km",
                "co2e_factor": 0.02
            },
            {
                "activity_id": "transportation_air_cargo",
                "name": "Air Freight",
                "category": "transport",
                "unit": "km",
                "co2e_factor": 0.65
            }
        ]
    
    url = f"{CLIMATIQ_BASE_URL}/emission-factors"
    if category:
        url += f"?category={category}"
        
    headers = {
        "Authorization": f"Bearer {CLIMATIQ_API_KEY}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        result = _handle_response(response, "Climatiq")
        return result.get("results", [])
    except Exception as e:
        logger.error(f"Error getting emission factors: {str(e)}")
        raise

# CARBON INTERFACE API FUNCTIONS
def estimate_shipping_emissions(weight_kg: float, distance_km: float, transport_mode: str) -> Dict[str, Any]:
    """
    Estimate shipping emissions using Carbon Interface API
    
    Args:
        weight_kg: Weight of shipment in kg
        distance_km: Distance in km
        transport_mode: Mode of transport (air, sea, rail, truck)
        
    Returns:
        dict: Emissions estimate
    """
    if not CARBON_INTERFACE_API_KEY:
        logger.warning("CARBON_INTERFACE_API_KEY not set, using mock data")
        # Return mock data for demo purposes
        emission_factors = {
            "air": 0.8,
            "sea": 0.02,
            "rail": 0.03,
            "truck": 0.1
        }
        factor = emission_factors.get(transport_mode.lower(), 0.1)
        
        return {
            "data": {
                "id": "mock-id",
                "type": "shipping_estimate",
                "attributes": {
                    "weight_value": weight_kg,
                    "weight_unit": "kg",
                    "distance_value": distance_km,
                    "distance_unit": "km",
                    "transport_method": transport_mode,
                    "carbon_g": round(weight_kg * distance_km * factor),
                    "carbon_lb": round(weight_kg * distance_km * factor * 0.0022, 2),
                    "carbon_kg": round(weight_kg * distance_km * factor / 1000, 2),
                    "carbon_mt": round(weight_kg * distance_km * factor / 1000000, 4)
                }
            }
        }
    
    # Updated endpoint URL - changed from /shipping_estimates to /estimates
    url = f"{CARBON_INTERFACE_BASE_URL}/estimates"
    headers = {
        "Authorization": f"Bearer {CARBON_INTERFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Map transport mode to Carbon Interface terminology
    transport_method_map = {
        "air": "air",
        "plane": "air",
        "sea": "sea",
        "ship": "sea",
        "ocean": "sea",
        "rail": "rail",
        "train": "rail",
        "truck": "truck",
        "road": "truck"
    }
    
    transport_method = transport_method_map.get(transport_mode.lower(), "truck")
    
    # Updated payload structure with "type" at the root level
    payload = {
        "type": "shipping",
        "shipping_method": transport_method,  # Changed from transport_method to shipping_method
        "weight_value": weight_kg,
        "weight_unit": "kg",
        "distance_value": distance_km,
        "distance_unit": "km"
    }
    
    try:
        # Add retry logic
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                response = requests.post(url, headers=headers, json=payload)
                return _handle_response(response, "Carbon Interface")
            except APIError as e:
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(f"Retrying Carbon Interface API request ({retry_count}/{max_retries})")
                    time.sleep(1)  # Wait before retrying
                else:
                    raise e
                    
    except Exception as e:
        logger.error(f"Error estimating shipping emissions: {str(e)}")
        # Use fallback calculation
        logger.warning("Using default emission factors for shipping calculation")
        emission_factors = {
            "air": 0.8,
            "sea": 0.02,
            "rail": 0.03,
            "truck": 0.1
        }
        factor = emission_factors.get(transport_mode.lower(), 0.1)
        
        return {
            "data": {
                "id": "mock-id",
                "type": "shipping_estimate",
                "attributes": {
                    "weight_value": weight_kg,
                    "weight_unit": "kg",
                    "distance_value": distance_km,
                    "distance_unit": "km",
                    "transport_method": transport_mode,
                    "carbon_g": round(weight_kg * distance_km * factor),
                    "carbon_lb": round(weight_kg * distance_km * factor * 0.0022, 2),
                    "carbon_kg": round(weight_kg * distance_km * factor / 1000, 2),
                    "carbon_mt": round(weight_kg * distance_km * factor / 1000000, 4)
                }
            }
        }

    
    url = f"{CARBON_INTERFACE_BASE_URL}/shipping_estimates"
    headers = {
        "Authorization": f"Bearer {CARBON_INTERFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Map transport mode to Carbon Interface terminology
    transport_method_map = {
        "air": "air",
        "plane": "air",
        "sea": "sea",
        "ship": "sea",
        "ocean": "sea",
        "rail": "rail",
        "train": "rail",
        "truck": "truck",
        "road": "truck"
    }
    
    transport_method = transport_method_map.get(transport_mode.lower(), "truck")
    
    # Corrected payload structure
    payload = {
        "type": "shipping",
        "weight_value": weight_kg,
        "weight_unit": "kg",
        "distance_value": distance_km,
        "distance_unit": "km",
        "transport_method": transport_method  # Ensure there are no extra quotes
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        return _handle_response(response, "Carbon Interface")
    except Exception as e:
        logger.error(f"Error estimating shipping emissions: {str(e)}")
        raise

# EPA ENVIROFACTS API FUNCTIONS
def get_facility_compliance(zip_code: str) -> List[Dict[str, Any]]:
    """
    Get facility compliance data from EPA Envirofacts API
    """
    # Using correct RESTful URL structure based on EPA documentation
    url = f"{EPA_BASE_URL}/lookups.mv_new_geo_best_picks/postal_code/beginsWith/{zip_code}/json"
    
    try:
        response = requests.get(url)
        return _handle_response(response, "EPA Envirofacts")
    except Exception as e:
        logger.error(f"Error getting facility compliance data: {str(e)}")
        # Return mock data if API fails
        return [
            {
                "facility_name": "Mock Manufacturing Inc",
                "registry_id": "M00001",
                "location_address": f"{zip_code} Main St",
                "city_name": "Anytown",
                "state_code": "CA",
                "postal_code": zip_code,
                "compliance_status": "In Compliance",
                "last_inspection_date": "2023-06-15"
            }
        ]

    
    url = f"{EPA_BASE_URL}/get_facilities"
    params = {
        "zip_code": zip_code,
        "output": "JSON",
        # "api_key": EPA_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        return _handle_response(response, "EPA Envirofacts")
    except Exception as e:
        logger.error(f"Error getting facility compliance data: {str(e)}")
        raise

# OPENCORPORATES API FUNCTIONS

def get_company_sustainability_data(company_name: str) -> Dict[str, Any]:
    """
    Get company sustainability data from DitchCarbon SBTI API
    
    Args:
        company_name: Name of the company
        
    Returns:
        dict: Company sustainability details
    """
    try:
        if not DITCHCARBON_API_KEY:
            logger.warning("DITCHCARBON_API_KEY not set, using mock data")
            # Return mock data
            return {
                "company_name": company_name,
                "has_science_based_target": True,
                "target_status": "Committed",
                "scope_coverage": "1, 2, 3",
                "target_year": 2030,
                "sustainability_score": 7.5
            }
        
        url = f"{DITCHCARBON_BASE_URL}/sbti_companies"
        headers = {
            "Authorization": f"Bearer {DITCHCARBON_API_KEY}",
            "accept": "application/json"
        }
        
        # Filter by company name in the query
        params = {"company_name": company_name}
        
        response = requests.get(url, headers=headers, params=params)
        return _handle_response(response, "DitchCarbon")
        
    except Exception as e:
        logger.error(f"Error getting sustainability data: {str(e)}")
        raise

def get_company_details(company_name: str, jurisdiction_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Get company details using GLEIF API (free, no authentication required)
    
    Args:
        company_name: Name of the company
        jurisdiction_code: Optional jurisdiction code
        
    Returns:
        dict: Company details
    """
    try:
        # GLEIF API endpoint for company search
        url = f"{GLEIF_API_BASE_URL}/lei-records"
        params = {"filter[entity.legalName]": company_name}
        
        response = requests.get(url, params=params)
        
        # Handle error if the request fails
        if response.status_code != 200:
            logger.warning(f"GLEIF API request failed with status {response.status_code}, using mock data")
            return _get_mock_company_details(company_name, jurisdiction_code)
        
        data = response.json()
        
        # Check if we got any results
        if not data.get("data"):
            logger.warning(f"No company data found for {company_name}, using mock data")
            return _get_mock_company_details(company_name, jurisdiction_code)
        
        company_data = data["data"][0]
        
        # Format the response to match expected structure
        return {
            "company": {
                "name": company_data["attributes"]["entity"]["legalName"]["name"],
                "company_number": company_data["id"],
                "jurisdiction_code": company_data["attributes"]["entity"]["legalAddress"]["country"],
                "status": company_data["attributes"]["entity"]["status"],
                "registration_date": company_data["attributes"]["registration"]["initialRegistrationDate"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting company details: {str(e)}")
        return _get_mock_company_details(company_name, jurisdiction_code)

def _get_mock_company_details(company_name: str, jurisdiction_code: Optional[str] = None) -> Dict[str, Any]:
    """Helper function to return mock company data"""
    return {
        "company": {
            "name": company_name,
            "company_number": "12345678",
            "jurisdiction_code": jurisdiction_code or "us_de",
            "incorporation_date": "2010-03-15",
            "company_type": "Private Limited Company",
            "status": "Active"
        }
    }

# AI MODEL API FUNCTIONS
def query_claude(prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """
    Query Claude AI model via Anthropic API
    
    Args:
        prompt: The prompt text
        temperature: Sampling temperature
        max_tokens: Maximum tokens in the response
        
    Returns:
        str: Claude's response
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set, using mock data")
        # Return mock data for demo purposes
        mock_responses = {
            "emissions": "Based on the data provided, your company's total emissions are approximately 25,000 kg CO2e. The largest contributors are logistics (65%), followed by electricity usage (25%) and raw materials (10%).",
            "optimize": "To reduce your carbon footprint, I recommend: 1) Consolidate shipments to reduce transportation emissions, 2) Switch to rail transport for long-distance freight, 3) Source materials from local suppliers to reduce shipping distances.",
            "compliance": "Your current operations are compliant with EPA standards. However, upcoming EU regulations will require additional reporting for Scope 3 emissions starting next year. I recommend preparing by implementing a more comprehensive data collection system.",
            "risk": "Supplier A has high sustainability risk due to location in a water-stressed region and lack of environmental certifications. Consider diversifying your supply chain with alternative vendors who have better sustainability profiles."
        }
        
        for key, response in mock_responses.items():
            if key in prompt.lower():
                return response
        
        return "I've analyzed your supply chain data and found several opportunities for sustainability improvements. The most significant impact would come from optimizing your logistics network and working with suppliers who have strong environmental practices."
    
    try:
        from anthropic import Anthropic
        
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are an AI assistant specializing in sustainable supply chain analysis. Provide concise, actionable insights based on supply chain data.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
    except ImportError:
        logger.error("anthropic package not installed. Please install with 'pip install anthropic'")
        raise
    except Exception as e:
        logger.error(f"Error querying Claude: {str(e)}")
        raise

def query_openai(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000, stream: bool = False) -> str:
    """
    Query OpenAI model using the newer client approach
    
    Args:
        prompt: The prompt text
        model: OpenAI model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in the response
        stream: Whether to return a streaming response
        
    Returns:
        str: OpenAI's response
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set, using mock data")
        # Return mock data for demo purposes
        return "Based on your supply chain data, I recommend optimizing your logistics routes to reduce carbon emissions by an estimated 15-20%. Additionally, consider sourcing from suppliers with better sustainability certifications, as this could significantly improve your company's environmental impact."
    
    try:
        from openai import OpenAI
        
        # Initialize the client with the forwarding API
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        
        # Create messages array
        messages = [
            {"role": "system", "content": "You are an AI assistant specializing in sustainable supply chain analysis."},
            {"role": "user", "content": prompt}
        ]
        
        # Handle streaming responses
        if stream:
            response_text = ""
            stream_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Collect the streaming content
            for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response_text += content
            
            return response_text
        else:
            # Handle non-streaming response
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return completion.choices[0].message.content
            
    except ImportError:
        logger.error("openai package not installed. Please install with 'pip install openai'")
        raise
    except Exception as e:
        logger.error(f"Error querying OpenAI: {str(e)}")
        raise

