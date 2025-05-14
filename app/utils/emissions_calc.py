"""
Emissions calculation utilities for the Supply Chain Optimizer.
Provides functions to calculate carbon footprint for various supply chain activities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime

from app.utils.api_clients import calculate_emission, estimate_shipping_emissions
from app.config import EMISSION_SCOPES, TRANSPORT_MODES, get_logger

logger = get_logger(__name__)

# Standard emission factors (fallback values when API calls are not possible)
EMISSION_FACTORS = {
    # Transport emission factors (kg CO2e per tonne-km)
    "transport": {
        "road": {
            "truck_small": 0.185,  # Small truck (<7.5t)
            "truck_medium": 0.125,  # Medium truck (7.5t-16t)
            "truck_large": 0.095,  # Large truck (>16t)
            "van": 0.250,  # Delivery van
        },
        "rail": 0.028,  # Rail freight
        "sea": {
            "container_ship": 0.015,  # Container ship
            "bulk_carrier": 0.008,  # Bulk carrier
            "tanker": 0.012,  # Tanker
        },
        "air": {
            "short_haul": 1.100,  # Short haul (<1000km)
            "medium_haul": 0.800,  # Medium haul (1000km-3700km)
            "long_haul": 0.600,  # Long haul (>3700km)
        }
    },
    
    # Electricity emission factors (kg CO2e per kWh) by country
    "electricity": {
        "USA": 0.380,
        "China": 0.555,
        "India": 0.708,
        "EU": 0.275,
        "UK": 0.233,
        "Germany": 0.350,
        "France": 0.056,
        "Global_average": 0.475
    },
    
    # Materials emission factors (kg CO2e per kg of material)
    "materials": {
        "steel": 1.85,
        "aluminum": 8.14,
        "plastic": 3.10,
        "paper": 1.09,
        "glass": 0.85,
        "concrete": 0.11,
        "textiles": 15.0,
        "electronics": 30.0
    }
}

def calculate_logistics_emissions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calculate emissions for logistics data
    
    Args:
        df: pandas.DataFrame containing logistics data
            Required columns: transport_mode, weight_kg, distance_km
            
    Returns:
        Tuple containing:
        - DataFrame with added emissions columns
        - Dictionary with summary statistics
    """
    if df is None or len(df) == 0:
        logger.warning("Empty dataframe provided to calculate_logistics_emissions")
        return pd.DataFrame(), {}
    
    # Check required columns
    required_cols = ['transport_mode', 'weight_kg', 'distance_km']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    # Handle missing columns with approximate calculations
    df_calc = df.copy()
    
    if 'weight_kg' not in df_calc.columns and 'weight' in df_calc.columns:
        df_calc['weight_kg'] = df_calc['weight']
    elif 'weight_kg' not in df_calc.columns:
        logger.warning("No weight column found, assuming default weight of 1000 kg")
        df_calc['weight_kg'] = 1000
        
    if 'distance_km' not in df_calc.columns and 'distance' in df_calc.columns:
        df_calc['distance_km'] = df_calc['distance']
    elif 'distance_km' not in df_calc.columns:
        logger.warning("No distance column found, estimating from origin/destination if available")
        if 'origin' in df_calc.columns and 'destination' in df_calc.columns:
            # Very simplified distance estimation (mock for demo)
            df_calc['distance_km'] = np.random.randint(500, 10000, size=len(df_calc))
        else:
            df_calc['distance_km'] = 1000  # Default assumption
    
    if 'transport_mode' not in df_calc.columns:
        if any(col in df_calc.columns for col in ['mode', 'shipping_method']):
            mode_col = next(col for col in ['mode', 'shipping_method'] if col in df_calc.columns)
            df_calc['transport_mode'] = df_calc[mode_col]
        else:
            logger.warning("No transport mode column found, assuming truck transport")
            df_calc['transport_mode'] = 'Truck'
    
    # Calculate tonne-kilometers
    df_calc['tonne_km'] = df_calc['weight_kg'] / 1000 * df_calc['distance_km']
    
    # Initialize emissions columns
    df_calc['emissions_kg_co2e'] = 0.0
    df_calc['emissions_g_co2e_per_tonne_km'] = 0.0
    
    # Process each shipment
    for idx, row in df_calc.iterrows():
        try:
            # Try to get emissions from API
            mode = str(row['transport_mode']).lower()
            
            # Map to standard modes
            if 'truck' in mode or 'road' in mode or 'van' in mode:
                std_mode = 'truck'
            elif 'ship' in mode or 'sea' in mode or 'ocean' in mode or 'boat' in mode:
                std_mode = 'sea'
            elif 'air' in mode or 'plane' in mode:
                std_mode = 'air'
            elif 'rail' in mode or 'train' in mode:
                std_mode = 'rail'
            else:
                std_mode = 'truck'  # Default
            
            try:
                # Try to get emissions from Carbon Interface API
                result = estimate_shipping_emissions(
                    weight_kg=row['weight_kg'],
                    distance_km=row['distance_km'],
                    transport_mode=std_mode
                )
                
                if 'data' in result and 'attributes' in result['data']:
                    attrs = result['data']['attributes']
                    emissions_kg = attrs.get('carbon_kg', 0)
                    df_calc.at[idx, 'emissions_kg_co2e'] = emissions_kg
                    
                    # Calculate emission factor (g CO2e per tonne-km)
                    if row['tonne_km'] > 0:
                        emission_factor = (emissions_kg * 1000) / row['tonne_km']
                        df_calc.at[idx, 'emissions_g_co2e_per_tonne_km'] = emission_factor
                    
                else:
                    raise ValueError("Invalid response format from Carbon Interface API")
                    
            except Exception as e:
                logger.warning(f"Could not calculate emissions via API: {str(e)}. Using default factors.")
                
                # Fallback to standard emission factors
                if std_mode == 'truck':
                    if row['weight_kg'] < 3000:
                        factor = EMISSION_FACTORS['transport']['road']['truck_small']
                    elif row['weight_kg'] < 10000:
                        factor = EMISSION_FACTORS['transport']['road']['truck_medium']
                    else:
                        factor = EMISSION_FACTORS['transport']['road']['truck_large']
                elif std_mode == 'sea':
                    factor = EMISSION_FACTORS['transport']['sea']['container_ship']
                elif std_mode == 'air':
                    if row['distance_km'] < 1000:
                        factor = EMISSION_FACTORS['transport']['air']['short_haul']
                    elif row['distance_km'] < 3700:
                        factor = EMISSION_FACTORS['transport']['air']['medium_haul']
                    else:
                        factor = EMISSION_FACTORS['transport']['air']['long_haul']
                elif std_mode == 'rail':
                    factor = EMISSION_FACTORS['transport']['rail']
                else:
                    factor = EMISSION_FACTORS['transport']['road']['truck_medium']
                
                # Calculate emissions based on tonne-km
                emissions_kg = row['tonne_km'] * factor
                df_calc.at[idx, 'emissions_kg_co2e'] = emissions_kg
                df_calc.at[idx, 'emissions_g_co2e_per_tonne_km'] = factor * 1000
                
        except Exception as e:
            logger.error(f"Error calculating emissions for row {idx}: {str(e)}")
            # Set default values
            df_calc.at[idx, 'emissions_kg_co2e'] = row['tonne_km'] * 0.1  # Default factor
            df_calc.at[idx, 'emissions_g_co2e_per_tonne_km'] = 100  # Default factor
    
    # Calculate summary statistics
    total_emissions = df_calc['emissions_kg_co2e'].sum()
    total_tonne_km = df_calc['tonne_km'].sum()
    
    emissions_by_mode = df_calc.groupby('transport_mode')['emissions_kg_co2e'].sum().to_dict()
    
    # Calculate average emission factor
    avg_emission_factor = 0
    if total_tonne_km > 0:
        avg_emission_factor = total_emissions / total_tonne_km
    
    # Calculate intensity metrics
    if 'cost' in df_calc.columns:
        emissions_per_cost = total_emissions / df_calc['cost'].sum() if df_calc['cost'].sum() > 0 else 0
    else:
        emissions_per_cost = None
    
    summary = {
        'total_emissions_kg': total_emissions,
        'total_tonne_km': total_tonne_km,
        'avg_emission_factor': avg_emission_factor,
        'emissions_by_mode': emissions_by_mode,
        'emissions_per_cost': emissions_per_cost,
        'scope': 'Scope 3 (Upstream transportation)'
    }
    
    return df_calc, summary

def calculate_procurement_emissions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calculate emissions for procurement data
    
    Args:
        df: pandas.DataFrame containing procurement data
            Required columns: product_category, quantity
            
    Returns:
        Tuple containing:
        - DataFrame with added emissions columns
        - Dictionary with summary statistics
    """
    if df is None or len(df) == 0:
        logger.warning("Empty dataframe provided to calculate_procurement_emissions")
        return pd.DataFrame(), {}
    
    # Check required columns and handle alternatives
    df_calc = df.copy()
    
    if 'product_category' not in df_calc.columns:
        if 'product' in df_calc.columns:
            df_calc['product_category'] = df_calc['product']
        elif 'category' in df_calc.columns:
            df_calc['product_category'] = df_calc['category']
        elif 'item' in df_calc.columns:
            df_calc['product_category'] = df_calc['item']
        else:
            logger.warning("No product category column found, using 'Unknown' as default")
            df_calc['product_category'] = 'Unknown'
    
    if 'quantity' not in df_calc.columns:
        if 'qty' in df_calc.columns:
            df_calc['quantity'] = df_calc['qty']
        else:
            logger.warning("No quantity column found, assuming quantity of 1")
            df_calc['quantity'] = 1
    
    # Add unit weight if not present
    if 'weight_kg' not in df_calc.columns:
        # Estimate weight based on product category (mock for demo)
        category_weights = {
            'Electronics': 2,
            'Packaging': 0.5,
            'Raw Materials': 10,
            'Office Supplies': 0.3,
            'Chemicals': 5
        }
        
        df_calc['weight_kg'] = df_calc['product_category'].map(
            lambda x: category_weights.get(str(x), 1)
        )
    
    # Initialize emissions columns
    df_calc['emissions_kg_co2e'] = 0.0
    df_calc['emissions_kg_co2e_per_unit'] = 0.0
    
    # Map product categories to material types for emission factors
    category_to_material = {
        'Electronics': 'electronics',
        'Packaging': 'paper',
        'Raw Materials': 'steel',
        'Office Supplies': 'paper',
        'Chemicals': 'plastic'
    }
    
    # Process each procurement line
    for idx, row in df_calc.iterrows():
        try:
            category = str(row['product_category'])
            material = category_to_material.get(category, 'steel')  # Default to steel
            
            # Get emission factor for material
            emission_factor = EMISSION_FACTORS['materials'].get(material, 1.0)
            
            # Calculate emissions
            unit_emissions = row['weight_kg'] * emission_factor
            total_emissions = unit_emissions * row['quantity']
            
            df_calc.at[idx, 'emissions_kg_co2e'] = total_emissions
            df_calc.at[idx, 'emissions_kg_co2e_per_unit'] = unit_emissions
            
        except Exception as e:
            logger.error(f"Error calculating procurement emissions for row {idx}: {str(e)}")
            # Set default values
            df_calc.at[idx, 'emissions_kg_co2e'] = row['quantity'] * 10  # Default assumption
            df_calc.at[idx, 'emissions_kg_co2e_per_unit'] = 10  # Default assumption
    
    # Calculate summary statistics
    total_emissions = df_calc['emissions_kg_co2e'].sum()
    total_units = df_calc['quantity'].sum()
    
    emissions_by_category = df_calc.groupby('product_category')['emissions_kg_co2e'].sum().to_dict()
    
    # Calculate intensity metrics
    if 'total_cost' in df_calc.columns:
        emissions_per_cost = total_emissions / df_calc['total_cost'].sum() if df_calc['total_cost'].sum() > 0 else 0
    elif 'unit_cost' in df_calc.columns and 'quantity' in df_calc.columns:
        df_calc['line_cost'] = df_calc['unit_cost'] * df_calc['quantity']
        emissions_per_cost = total_emissions / df_calc['line_cost'].sum() if df_calc['line_cost'].sum() > 0 else 0
    else:
        emissions_per_cost = None
    
    summary = {
        'total_emissions_kg': total_emissions,
        'emissions_per_unit': total_emissions / total_units if total_units > 0 else 0,
        'emissions_by_category': emissions_by_category,
        'emissions_per_cost': emissions_per_cost,
        'scope': 'Scope 3 (Purchased goods and services)'
    }
    
    return df_calc, summary

def calculate_supplier_emissions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calculate and assess emissions for supplier data
    
    Args:
        df: pandas.DataFrame containing supplier data
            
    Returns:
        Tuple containing:
        - DataFrame with added emissions assessment columns
        - Dictionary with summary statistics
    """
    if df is None or len(df) == 0:
        logger.warning("Empty dataframe provided to calculate_supplier_emissions")
        return pd.DataFrame(), {}
    
    # Make a copy of the dataframe
    df_calc = df.copy()
    
    # Check if carbon footprint data already exists
    if 'carbon_footprint_tons' in df_calc.columns:
        # Convert to kg
        df_calc['carbon_footprint_kg'] = df_calc['carbon_footprint_tons'] * 1000
    else:
        # Estimate carbon footprint based on company size and industry
        if 'annual_revenue_usd' not in df_calc.columns:
            logger.warning("No revenue data found, estimating based on employee count if available")
            if 'employee_count' in df_calc.columns:
                # Rough estimate: $100,000 revenue per employee
                df_calc['annual_revenue_usd'] = df_calc['employee_count'] * 100000
            else:
                # Default value
                df_calc['annual_revenue_usd'] = 1000000
        
        # Get industry category
        if 'category' not in df_calc.columns:
            logger.warning("No category column found, using 'Manufacturing' as default")
            df_calc['category'] = 'Manufacturing'
        
        # Industry-specific emission factors (kg CO2e per $1000 revenue)
        industry_factors = {
            'Manufacturing': 150,
            'Electronics': 100,
            'Packaging': 120,
            'Raw Materials': 200,
            'Office Supplies': 80,
            'Chemicals': 250,
            'Retail': 70,
            'Services': 50
        }
        
        # Calculate estimated carbon footprint
        df_calc['carbon_footprint_kg'] = df_calc.apply(
            lambda row: (row['annual_revenue_usd'] / 1000) * 
                        industry_factors.get(str(row['category']), 150),
            axis=1
        )
    
    # Calculate emissions intensity (carbon footprint per revenue)
    if 'annual_revenue_usd' in df_calc.columns and df_calc['annual_revenue_usd'].sum() > 0:
        df_calc['emissions_intensity'] = df_calc['carbon_footprint_kg'] / df_calc['annual_revenue_usd']
    else:
        df_calc['emissions_intensity'] = np.nan
    
    # Assess sustainability based on certifications
    if 'sustainability_certification' in df_calc.columns:
        # Score certifications (higher is better)
        cert_scores = {
            'ISO 14001': 5,
            'B Corp': 5,
            'Fair Trade': 4,
            'LEED': 3,
            'None': 0,
            'Unknown': 0
        }
        
        df_calc['certification_score'] = df_calc['sustainability_certification'].map(
            lambda x: cert_scores.get(str(x), 0)
        )
    else:
        df_calc['certification_score'] = 0
    
    # Create a sustainability score (lower emissions intensity and higher certification score is better)
    # Normalize emissions intensity to 0-10 scale (inversely related to sustainability)
    if not df_calc['emissions_intensity'].isna().all():
        max_intensity = df_calc['emissions_intensity'].max()
        min_intensity = df_calc['emissions_intensity'].min()
        
        if max_intensity > min_intensity:
            df_calc['emissions_score'] = 10 - 10 * (
                (df_calc['emissions_intensity'] - min_intensity) / (max_intensity - min_intensity)
            )
        else:
            df_calc['emissions_score'] = 5  # Default mid-value
    else:
        df_calc['emissions_score'] = 5  # Default mid-value
    
    # Calculate overall sustainability score (weighted average)
    df_calc['sustainability_score'] = 0.7 * df_calc['emissions_score'] + 0.3 * df_calc['certification_score']
    
    # Risk assessment
    df_calc['sustainability_risk'] = 10 - df_calc['sustainability_score']
    
    # Calculate summary statistics
    total_footprint = df_calc['carbon_footprint_kg'].sum()
    avg_sustainability_score = df_calc['sustainability_score'].mean()
    
    emissions_by_category = df_calc.groupby('category')['carbon_footprint_kg'].sum().to_dict() if 'category' in df_calc.columns else {}
    
    # Identify high-risk suppliers
    high_risk_threshold = 7
    high_risk_suppliers = df_calc[df_calc['sustainability_risk'] >= high_risk_threshold]
    
    summary = {
        'total_supplier_footprint_kg': total_footprint,
        'avg_sustainability_score': avg_sustainability_score,
        'emissions_by_category': emissions_by_category,
        'high_risk_supplier_count': len(high_risk_suppliers),
        'high_risk_suppliers': high_risk_suppliers['supplier_name'].tolist() if 'supplier_name' in high_risk_suppliers.columns else [],
        'scope': 'Scope 3 (Upstream)'
    }
    
    return df_calc, summary

def calculate_total_emissions(
    logistics_summary: Dict[str, Any] = None,
    procurement_summary: Dict[str, Any] = None,
    supplier_summary: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculate total emissions and metrics based on all data types
    
    Args:
        logistics_summary: Summary dict from logistics calculations
        procurement_summary: Summary dict from procurement calculations
        supplier_summary: Summary dict from supplier calculations
        
    Returns:
        dict: Consolidated emissions summary
    """
    # Initialize with zeros
    total_emissions = 0
    emissions_by_scope = {
        'scope1': 0,
        'scope2': 0,
        'scope3': 0
    }
    emissions_by_category = {}
    
    # Add logistics emissions if available
    if logistics_summary:
        total_emissions += logistics_summary.get('total_emissions_kg', 0)
        emissions_by_scope['scope3'] += logistics_summary.get('total_emissions_kg', 0)
        emissions_by_category['Logistics'] = logistics_summary.get('total_emissions_kg', 0)
        
        # Add transport mode breakdown
        if 'emissions_by_mode' in logistics_summary:
            for mode, value in logistics_summary['emissions_by_mode'].items():
                emissions_by_category[f'Logistics - {mode}'] = value
    
    # Add procurement emissions if available
    if procurement_summary:
        total_emissions += procurement_summary.get('total_emissions_kg', 0)
        emissions_by_scope['scope3'] += procurement_summary.get('total_emissions_kg', 0)
        emissions_by_category['Procurement'] = procurement_summary.get('total_emissions_kg', 0)
        
        # Add category breakdown
        if 'emissions_by_category' in procurement_summary:
            for category, value in procurement_summary['emissions_by_category'].items():
                emissions_by_category[f'Procurement - {category}'] = value
    
    # Add supplier emissions if available (avoid double counting with procurement)
    # Only include if no procurement data to avoid double counting
    if supplier_summary and not procurement_summary:
        supplier_emissions = supplier_summary.get('total_supplier_footprint_kg', 0)
        # Scale down to avoid overestimation (assume only portion is attributable)
        attributable_supplier_emissions = supplier_emissions * 0.1  # Assume 10% attribution
        
        total_emissions += attributable_supplier_emissions
        emissions_by_scope['scope3'] += attributable_supplier_emissions
        emissions_by_category['Supplier'] = attributable_supplier_emissions
        
        # Add category breakdown
        if 'emissions_by_category' in supplier_summary:
            for category, value in supplier_summary['emissions_by_category'].items():
                emissions_by_category[f'Supplier - {category}'] = value * 0.1  # Assume 10% attribution
    
    # Calculate metrics
    total_co2e_tonnes = total_emissions / 1000
    
    # Summary dictionary
    result = {
        'total_emissions_kg': total_emissions,
        'total_emissions_tonnes': total_co2e_tonnes,
        'emissions_by_scope': emissions_by_scope,
        'emissions_by_category': emissions_by_category,
        'scope_percentages': {
            'scope1': (emissions_by_scope['scope1'] / total_emissions * 100) if total_emissions > 0 else 0,
            'scope2': (emissions_by_scope['scope2'] / total_emissions * 100) if total_emissions > 0 else 0,
            'scope3': (emissions_by_scope['scope3'] / total_emissions * 100) if total_emissions > 0 else 0
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return result

def generate_optimization_targets(
    total_summary: Dict[str, Any],
    logistics_df: Optional[pd.DataFrame] = None,
    procurement_df: Optional[pd.DataFrame] = None,
    supplier_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Generate optimization targets and recommendations based on emissions data
    
    Args:
        total_summary: Summary dict from calculate_total_emissions
        logistics_df: Logistics dataframe with emissions calculations
        procurement_df: Procurement dataframe with emissions calculations
        supplier_df: Supplier dataframe with emissions calculations
        
    Returns:
        dict: Optimization targets and recommendations
    """
    recommendations = []
    targets = {}
    
    # Set overall reduction target (industry standard is 5-15% per year)
    total_emissions = total_summary.get('total_emissions_kg', 0)
    if total_emissions > 0:
        targets['overall'] = {
            'current_emissions_kg': total_emissions,
            'reduction_target_percent': 10,  # Standard target
            'reduction_target_kg': total_emissions * 0.1,
            'target_emissions_kg': total_emissions * 0.9
        }
    
    # Logistics optimization targets
    if logistics_df is not None and not logistics_df.empty:
        logistics_emissions = total_summary.get('emissions_by_category', {}).get('Logistics', 0)
        
        if logistics_emissions > 0:
            # Analyze transport modes
            if 'transport_mode' in logistics_df.columns:
                mode_emissions = logistics_df.groupby('transport_mode')['emissions_kg_co2e'].sum()
                highest_mode = mode_emissions.idxmax()
                highest_emissions = mode_emissions.max()
                
                # Target highest-emission transport mode
                if 'Air' in str(highest_mode) or 'air' in str(highest_mode) or 'plane' in str(highest_mode):
                    # Air freight is high-emission, significant reduction possible by modal shift
                    recommendations.append({
                        'category': 'Logistics',
                        'title': 'Shift Air Freight to Sea or Rail',
                        'description': f'Shifting air freight to sea or rail could reduce emissions by up to 80%.',
                        'potential_reduction_kg': highest_emissions * 0.8,
                        'difficulty': 'Medium',
                        'timeframe': 'Medium-term (3-6 months)'
                    })
                elif 'Truck' in str(highest_mode) or 'truck' in str(highest_mode) or 'road' in str(highest_mode):
                    # Road freight optimization
                    recommendations.append({
                        'category': 'Logistics',
                        'title': 'Optimize Road Transport',
                        'description': 'Consolidate shipments and optimize routes to reduce emissions by 15-20%.',
                        'potential_reduction_kg': highest_emissions * 0.15,
                        'difficulty': 'Low',
                        'timeframe': 'Short-term (1-3 months)'
                    })
            
            # Check for long-distance shipments
            if 'distance_km' in logistics_df.columns:
                long_distance_threshold = 2000  # km
                long_distance_shipments = logistics_df[logistics_df['distance_km'] > long_distance_threshold]
                
                if not long_distance_shipments.empty:
                    long_emissions = long_distance_shipments['emissions_kg_co2e'].sum()
                    
                    recommendations.append({
                        'category': 'Logistics',
                        'title': 'Source Locally for Long-Distance Materials',
                        'description': f'Identified {len(long_distance_shipments)} long-distance shipments. Consider local sourcing to reduce transportation emissions.',
                        'potential_reduction_kg': long_emissions * 0.5,  # Assuming 50% reduction from local sourcing
                        'difficulty': 'Medium',
                        'timeframe': 'Medium-term (3-6 months)'
                    })
            
            # Set logistics target
            targets['logistics'] = {
                'current_emissions_kg': logistics_emissions,
                'reduction_target_percent': 15,  # Slightly higher than overall
                'reduction_target_kg': logistics_emissions * 0.15,
                'target_emissions_kg': logistics_emissions * 0.85
            }
    
    # Procurement optimization targets
    if procurement_df is not None and not procurement_df.empty:
        procurement_emissions = total_summary.get('emissions_by_category', {}).get('Procurement', 0)
        
        if procurement_emissions > 0:
            # Analyze product categories
            if 'product_category' in procurement_df.columns and 'emissions_kg_co2e' in procurement_df.columns:
                category_emissions = procurement_df.groupby('product_category')['emissions_kg_co2e'].sum()
                highest_category = category_emissions.idxmax()
                highest_cat_emissions = category_emissions.max()
                
                # Target highest-emission category
                recommendations.append({
                    'category': 'Procurement',
                    'title': f'Sustainable Alternatives for {highest_category}',
                    'description': f'{highest_category} represents your highest-emission product category. Research sustainable alternatives or suppliers with better environmental practices.',
                    'potential_reduction_kg': highest_cat_emissions * 0.3,  # Assuming 30% reduction potential
                    'difficulty': 'Medium',
                    'timeframe': 'Medium-term (3-6 months)'
                })
            
            # Set procurement target
            targets['procurement'] = {
                'current_emissions_kg': procurement_emissions,
                'reduction_target_percent': 12,
                'reduction_target_kg': procurement_emissions * 0.12,
                'target_emissions_kg': procurement_emissions * 0.88
            }
    
    # Supplier optimization targets
    if supplier_df is not None and not supplier_df.empty:
        # Check for high-risk suppliers
        if 'sustainability_risk' in supplier_df.columns:
            high_risk_threshold = 7
            high_risk_suppliers = supplier_df[supplier_df['sustainability_risk'] >= high_risk_threshold]
            
            if not high_risk_suppliers.empty:
                if 'supplier_name' in high_risk_suppliers.columns:
                    supplier_names = ", ".join(high_risk_suppliers['supplier_name'].head(3).astype(str))
                    
                    recommendations.append({
                        'category': 'Supplier Management',
                        'title': 'Engage with High-Risk Suppliers',
                        'description': f'Identified {len(high_risk_suppliers)} high-sustainability-risk suppliers (e.g., {supplier_names}). Engage them on sustainability practices or consider alternatives.',
                        'potential_reduction_kg': total_emissions * 0.05,  # Rough estimate
                        'difficulty': 'Medium',
                        'timeframe': 'Medium-term (3-6 months)'
                    })
                else:
                    recommendations.append({
                        'category': 'Supplier Management',
                        'title': 'Address High-Risk Suppliers',
                        'description': f'Identified {len(high_risk_suppliers)} suppliers with high sustainability risk scores. Consider sustainability criteria in supplier selection.',
                        'potential_reduction_kg': total_emissions * 0.05,  # Rough estimate
                        'difficulty': 'Medium',
                        'timeframe': 'Medium-term (3-6 months)'
                    })
    
    # General recommendations based on overall emissions profile
    if total_emissions > 0:
        scope_percentages = total_summary.get('scope_percentages', {})
        scope3_percentage = scope_percentages.get('scope3', 0)
        
        if scope3_percentage > 80:
            recommendations.append({
                'category': 'Strategy',
                'title': 'Supplier Engagement Program',
                'description': 'Over 80% of your emissions are Scope 3 (supply chain). Implement a comprehensive supplier engagement program focusing on emissions reduction and reporting.',
                'potential_reduction_kg': total_emissions * 0.1,
                'difficulty': 'High',
                'timeframe': 'Long-term (6-12 months)'
            })
        
        # Add circular economy recommendation
        recommendations.append({
            'category': 'Strategy',
            'title': 'Circular Economy Initiative',
            'description': 'Implement circular economy principles to reduce virgin material use and associated emissions through product redesign, take-back programs, or recycled content.',
            'potential_reduction_kg': total_emissions * 0.08,
            'difficulty': 'High',
            'timeframe': 'Long-term (6-12 months)'
        })
    
    # Calculate total potential reduction
    total_potential_reduction = sum(r['potential_reduction_kg'] for r in recommendations)
    
    return {
        'targets': targets,
        'recommendations': recommendations,
        'total_potential_reduction_kg': total_potential_reduction,
        'potential_reduction_percent': (total_potential_reduction / total_emissions * 100) if total_emissions > 0 else 0
    }
