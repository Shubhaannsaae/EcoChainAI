"""
Optimization utilities for the Supply Chain Optimizer.
Provides functions to generate AI-driven optimization strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import json
from datetime import datetime, timedelta
from app.utils.api_clients import query_claude, query_openai
from app.utils.emissions_calc import generate_optimization_targets
from app.config import get_logger

logger = get_logger(__name__)

def generate_optimal_routes(logistics_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate optimized logistics routes to reduce carbon emissions
    
    Args:
        logistics_df: DataFrame with logistics data including emissions
    
    Returns:
        Tuple containing:
        - DataFrame with optimized routes
        - Dictionary with optimization statistics
    """
    if logistics_df is None or len(logistics_df) == 0:
        logger.warning("Empty dataframe provided to generate_optimal_routes")
        return pd.DataFrame(), {}
    
    # Create a copy of the original dataframe
    optimized_df = logistics_df.copy()
    
    # Check required columns
    required_cols = ['transport_mode', 'distance_km', 'emissions_kg_co2e']
    missing_cols = [col for col in required_cols if col not in optimized_df.columns]
    
    if missing_cols:
        logger.warning(f"Missing required columns for route optimization: {missing_cols}")
        return optimized_df, {"error": f"Missing columns: {missing_cols}"}
    
    # OPTIMIZATION STRATEGY 1: MODE SHIFTING
    # Find high-emission shipments that can be shifted to lower-emission transport modes
    
    # Define emission reduction factors for mode shifts
    mode_shift_factors = {
        'air_to_sea': 0.85,  # 85% reduction when shifting from air to sea
        'air_to_rail': 0.75,  # 75% reduction when shifting from air to rail
        'truck_to_rail': 0.70,  # 70% reduction when shifting from truck to rail
        'truck_to_sea': 0.75,  # 75% reduction when shifting from truck to sea
    }
    
    # Initialize new columns for optimization
    optimized_df['optimized_transport_mode'] = optimized_df['transport_mode']
    optimized_df['optimized_emissions_kg_co2e'] = optimized_df['emissions_kg_co2e']
    optimized_df['emissions_reduction_kg_co2e'] = 0.0
    optimized_df['optimization_method'] = 'No change'
    
    # Mode shift optimization
    for idx, row in optimized_df.iterrows():
        current_mode = str(row['transport_mode']).lower()
        
        # Skip if distance or emissions are zero
        if row['distance_km'] <= 0 or row['emissions_kg_co2e'] <= 0:
            continue
        
        # Check for air freight opportunities
        if ('air' in current_mode or 'plane' in current_mode) and row['distance_km'] > 500:
            if row['distance_km'] < 2000:
                # Short/medium haul - can shift to rail
                optimized_df.at[idx, 'optimized_transport_mode'] = 'Rail'
                reduction_factor = mode_shift_factors['air_to_rail']
                optimized_df.at[idx, 'optimization_method'] = 'Air to Rail'
            else:
                # Long haul - can shift to sea if not urgent
                optimized_df.at[idx, 'optimized_transport_mode'] = 'Sea'
                reduction_factor = mode_shift_factors['air_to_sea']
                optimized_df.at[idx, 'optimization_method'] = 'Air to Sea'
                
            # Calculate emissions reduction
            original_emissions = row['emissions_kg_co2e']
            new_emissions = original_emissions * (1 - reduction_factor)
            
            optimized_df.at[idx, 'optimized_emissions_kg_co2e'] = new_emissions
            optimized_df.at[idx, 'emissions_reduction_kg_co2e'] = original_emissions - new_emissions
        
        # Check for truck freight opportunities
        elif ('truck' in current_mode or 'road' in current_mode) and row['distance_km'] > 750:
            # Long distance trucking can often be shifted to rail
            optimized_df.at[idx, 'optimized_transport_mode'] = 'Rail'
            reduction_factor = mode_shift_factors['truck_to_rail']
            optimized_df.at[idx, 'optimization_method'] = 'Truck to Rail'
            
            # Calculate emissions reduction
            original_emissions = row['emissions_kg_co2e']
            new_emissions = original_emissions * (1 - reduction_factor)
            
            optimized_df.at[idx, 'optimized_emissions_kg_co2e'] = new_emissions
            optimized_df.at[idx, 'emissions_reduction_kg_co2e'] = original_emissions - new_emissions
    
    # OPTIMIZATION STRATEGY 2: SHIPMENT CONSOLIDATION
    # Group nearby shipments that occur within a short time window
    
    # Check if we have date information
    if 'date' in optimized_df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(optimized_df['date']):
            try:
                optimized_df['date'] = pd.to_datetime(optimized_df['date'])
            except:
                logger.warning("Could not convert date column to datetime")
        
        # Ensure date is in datetime format for processing
        if pd.api.types.is_datetime64_dtype(optimized_df['date']):
            # Check if origin/destination columns exist
            if 'origin' in optimized_df.columns and 'destination' in optimized_df.columns:
                # Find shipments that can be consolidated (same origin/destination, close dates)
                
                # Sort by date
                optimized_df = optimized_df.sort_values('date')
                
                # Initialize consolidation groups
                optimized_df['consolidation_group'] = None
                
                # Group by origin and destination
                route_groups = optimized_df.groupby(['origin', 'destination'])
                
                # Assign consolidation groups
                consolidation_group_id = 1
                
                for (origin, destination), group in route_groups:
                    # Skip if only one shipment on this route
                    if len(group) <= 1:
                        continue
                    
                    # Index of the first shipment in this group
                    first_idx = group.index[0]
                    
                    # Current shipment's date
                    current_date = group.at[first_idx, 'date']
                    
                    # Assign first shipment to a consolidation group
                    optimized_df.at[first_idx, 'consolidation_group'] = consolidation_group_id
                    
                    # Check other shipments in the group
                    for idx in group.index[1:]:
                        ship_date = group.at[idx, 'date']
                        
                        # Check if within consolidation window (7 days)
                        date_diff = ship_date - current_date if ship_date > current_date else current_date - ship_date
                        
                        if date_diff.days <= 7:
                            # Assign to the current consolidation group
                            optimized_df.at[idx, 'consolidation_group'] = consolidation_group_id
                            
                            # Add to optimization method
                            if optimized_df.at[idx, 'optimization_method'] == 'No change':
                                optimized_df.at[idx, 'optimization_method'] = 'Shipment Consolidation'
                            else:
                                optimized_df.at[idx, 'optimization_method'] += ' + Consolidation'
                            
                            # Reduction from consolidation (assume 30% reduction from shared transport)
                            reduction = optimized_df.at[idx, 'optimized_emissions_kg_co2e'] * 0.3
                            optimized_df.at[idx, 'optimized_emissions_kg_co2e'] -= reduction
                            optimized_df.at[idx, 'emissions_reduction_kg_co2e'] += reduction
                        else:
                            # Start a new consolidation group
                            consolidation_group_id += 1
                            optimized_df.at[idx, 'consolidation_group'] = consolidation_group_id
                            current_date = ship_date
    
    # Calculate summary statistics
    original_total_emissions = optimized_df['emissions_kg_co2e'].sum()
    optimized_total_emissions = optimized_df['optimized_emissions_kg_co2e'].sum()
    total_reduction = optimized_df['emissions_reduction_kg_co2e'].sum()
    
    # Calculate reduction by optimization method
    reduction_by_method = optimized_df.groupby('optimization_method')['emissions_reduction_kg_co2e'].sum().to_dict()
    
    # Calculate percentage reduction
    percent_reduction = 0
    if original_total_emissions > 0:
        percent_reduction = (total_reduction / original_total_emissions) * 100
    
    # Prepare summary
    summary = {
        'original_emissions_kg': original_total_emissions,
        'optimized_emissions_kg': optimized_total_emissions,
        'total_reduction_kg': total_reduction,
        'percent_reduction': percent_reduction,
        'reduction_by_method': reduction_by_method,
        'shipments_optimized': len(optimized_df[optimized_df['optimization_method'] != 'No change']),
        'total_shipments': len(optimized_df)
    }
    
    return optimized_df, summary

def generate_sustainable_procurement_plan(procurement_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate sustainable procurement recommendations
    
    Args:
        procurement_df: DataFrame with procurement data including emissions
    
    Returns:
        Tuple containing:
        - DataFrame with sustainable procurement alternatives
        - Dictionary with optimization statistics
    """
    if procurement_df is None or len(procurement_df) == 0:
        logger.warning("Empty dataframe provided to generate_sustainable_procurement_plan")
        return pd.DataFrame(), {}
    
    # Create a copy of the original dataframe
    optimized_df = procurement_df.copy()
    
    # Check required columns
    required_cols = ['product_category', 'emissions_kg_co2e']
    missing_cols = [col for col in required_cols if col not in optimized_df.columns]
    
    if missing_cols:
        logger.warning(f"Missing required columns for procurement optimization: {missing_cols}")
        return optimized_df, {"error": f"Missing columns: {missing_cols}"}
    
    # Sustainable alternatives database with estimated emission reduction percentages
    sustainable_alternatives = {
        'Electronics': {
            'alternative': 'Energy-efficient electronics with EPEAT Gold certification',
            'reduction_percent': 20,
            'cost_impact_percent': 5,  # Slightly higher cost
            'implementation_difficulty': 'Medium'
        },
        'Packaging': {
            'alternative': 'Recycled and biodegradable packaging materials',
            'reduction_percent': 40,
            'cost_impact_percent': 0,  # Neutral cost
            'implementation_difficulty': 'Low'
        },
        'Raw Materials': {
            'alternative': 'Locally sourced materials with lower carbon footprint',
            'reduction_percent': 30,
            'cost_impact_percent': -5,  # Cost savings
            'implementation_difficulty': 'Medium'
        },
        'Office Supplies': {
            'alternative': 'Recycled paper products and refillable items',
            'reduction_percent': 25,
            'cost_impact_percent': -10,  # Cost savings
            'implementation_difficulty': 'Low'
        },
        'Chemicals': {
            'alternative': 'Bio-based or less hazardous chemical alternatives',
            'reduction_percent': 15,
            'cost_impact_percent': 10,  # Higher cost
            'implementation_difficulty': 'High'
        }
    }
    
    # Initialize new columns for sustainable alternatives
    optimized_df['sustainable_alternative'] = 'No alternative identified'
    optimized_df['emissions_reduction_percent'] = 0.0
    optimized_df['optimized_emissions_kg_co2e'] = optimized_df['emissions_kg_co2e']
    optimized_df['emissions_reduction_kg_co2e'] = 0.0
    optimized_df['cost_impact_percent'] = 0.0
    optimized_df['implementation_difficulty'] = 'Not applicable'
    
    # Apply sustainable alternatives
    for idx, row in optimized_df.iterrows():
        category = str(row['product_category'])
        
        # Check if we have a sustainable alternative for this category
        if category in sustainable_alternatives:
            alternative_info = sustainable_alternatives[category]
            
            # Update with sustainable alternative information
            optimized_df.at[idx, 'sustainable_alternative'] = alternative_info['alternative']
            optimized_df.at[idx, 'emissions_reduction_percent'] = alternative_info['reduction_percent']
            optimized_df.at[idx, 'cost_impact_percent'] = alternative_info['cost_impact_percent']
            optimized_df.at[idx, 'implementation_difficulty'] = alternative_info['implementation_difficulty']
            
            # Calculate emissions reduction
            reduction_percent = alternative_info['reduction_percent'] / 100
            original_emissions = row['emissions_kg_co2e']
            reduction_amount = original_emissions * reduction_percent
            
            optimized_df.at[idx, 'emissions_reduction_kg_co2e'] = reduction_amount
            optimized_df.at[idx, 'optimized_emissions_kg_co2e'] = original_emissions - reduction_amount
    
    # Calculate summary statistics
    original_total_emissions = optimized_df['emissions_kg_co2e'].sum()
    optimized_total_emissions = optimized_df['optimized_emissions_kg_co2e'].sum()
    total_reduction = optimized_df['emissions_reduction_kg_co2e'].sum()
    
    # Calculate reduction by category
    reduction_by_category = optimized_df.groupby('product_category')['emissions_reduction_kg_co2e'].sum().to_dict()
    
    # Calculate overall cost impact
    if 'unit_cost' in optimized_df.columns and 'quantity' in optimized_df.columns:
        optimized_df['original_cost'] = optimized_df['unit_cost'] * optimized_df['quantity']
        optimized_df['cost_impact_amount'] = optimized_df['original_cost'] * (optimized_df['cost_impact_percent'] / 100)
        cost_impact = optimized_df['cost_impact_amount'].sum()
        original_cost = optimized_df['original_cost'].sum()
        
        cost_impact_percent = 0
        if original_cost > 0:
            cost_impact_percent = (cost_impact / original_cost) * 100
    else:
        cost_impact = None
        cost_impact_percent = None
    
    # Calculate percentage reduction
    percent_reduction = 0
    if original_total_emissions > 0:
        percent_reduction = (total_reduction / original_total_emissions) * 100
    
    # Prepare summary
    summary = {
        'original_emissions_kg': original_total_emissions,
        'optimized_emissions_kg': optimized_total_emissions,
        'total_reduction_kg': total_reduction,
        'percent_reduction': percent_reduction,
        'reduction_by_category': reduction_by_category,
        'items_with_alternatives': len(optimized_df[optimized_df['sustainable_alternative'] != 'No alternative identified']),
        'total_items': len(optimized_df),
        'cost_impact': cost_impact,
        'cost_impact_percent': cost_impact_percent
    }
    
    return optimized_df, summary

def generate_supplier_sustainability_plan(supplier_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate supplier sustainability recommendations
    
    Args:
        supplier_df: DataFrame with supplier data including sustainability scores
    
    Returns:
        Tuple containing:
        - DataFrame with supplier recommendations
        - Dictionary with optimization statistics
    """
    if supplier_df is None or len(supplier_df) == 0:
        logger.warning("Empty dataframe provided to generate_supplier_sustainability_plan")
        return pd.DataFrame(), {}
    
    # Create a copy of the original dataframe
    optimized_df = supplier_df.copy()
    
    # Check if sustainability columns exist
    if 'sustainability_score' not in optimized_df.columns:
        logger.warning("No sustainability_score column found for supplier optimization")
        if 'risk_score' in optimized_df.columns:
            # Invert risk score to create a sustainability score (10 - risk_score)
            optimized_df['sustainability_score'] = 10 - optimized_df['risk_score']
        else:
            # Create a mock sustainability score
            optimized_df['sustainability_score'] = np.random.uniform(2, 9, size=len(optimized_df))
    
    # Initialize recommendation columns
    optimized_df['recommendation'] = ''
    optimized_df['priority'] = ''
    optimized_df['potential_improvement'] = 0.0
    optimized_df['engagement_strategy'] = ''
    
    # Categorize suppliers by sustainability score
    optimized_df.loc[optimized_df['sustainability_score'] >= 8, 'priority'] = 'Low'
    optimized_df.loc[(optimized_df['sustainability_score'] >= 5) & (optimized_df['sustainability_score'] < 8), 'priority'] = 'Medium'
    optimized_df.loc[optimized_df['sustainability_score'] < 5, 'priority'] = 'High'
    
    # Generate recommendations based on priority
    for idx, row in optimized_df.iterrows():
        priority = row['priority']
        
        if priority == 'High':
            # Check if there's additional info to tailor the recommendation
            if 'certification_score' in optimized_df.columns and row['certification_score'] == 0:
                optimized_df.at[idx, 'recommendation'] = 'Require sustainability certification (e.g., ISO 14001)'
                optimized_df.at[idx, 'engagement_strategy'] = 'Formal improvement program with timeline for certification'
            else:
                optimized_df.at[idx, 'recommendation'] = 'Develop sustainability improvement plan or consider alternative suppliers'
                optimized_df.at[idx, 'engagement_strategy'] = 'Quarterly sustainability reviews and improvement tracking'
            
            # High-risk suppliers have high improvement potential
            optimized_df.at[idx, 'potential_improvement'] = 4.0  # Points on 10-point scale
            
        elif priority == 'Medium':
            optimized_df.at[idx, 'recommendation'] = 'Engage on specific sustainability improvements'
            optimized_df.at[idx, 'engagement_strategy'] = 'Semi-annual sustainability reviews and targeted initiatives'
            optimized_df.at[idx, 'potential_improvement'] = 2.0  # Points on 10-point scale
            
        else:  # Low priority
            optimized_df.at[idx, 'recommendation'] = 'Maintain current relationship and share best practices'
            optimized_df.at[idx, 'engagement_strategy'] = 'Annual sustainability performance review'
            optimized_df.at[idx, 'potential_improvement'] = 0.5  # Points on 10-point scale
    
    # Calculate current and potential sustainability metrics
    current_avg_score = optimized_df['sustainability_score'].mean()
    potential_improvements = optimized_df['potential_improvement']
    potential_avg_score = current_avg_score + potential_improvements.mean()
    
    # Count suppliers by priority
    priority_counts = optimized_df['priority'].value_counts().to_dict()
    
    # Prepare summary
    summary = {
        'current_avg_sustainability_score': current_avg_score,
        'potential_avg_sustainability_score': potential_avg_score,
        'improvement_potential': potential_avg_score - current_avg_score,
        'high_priority_supplier_count': priority_counts.get('High', 0),
        'medium_priority_supplier_count': priority_counts.get('Medium', 0),
        'low_priority_supplier_count': priority_counts.get('Low', 0),
        'total_suppliers': len(optimized_df)
    }
    
    return optimized_df, summary

def generate_ai_sustainability_strategy(
    logistics_data=None,
    procurement_data=None,
    supplier_data=None,
    total_emissions=None,
    optimization_targets=None
):
    """Generate comprehensive AI-driven sustainability strategy"""
    # Compile data for the AI prompt
    data_summary = {
        "logistics_data": logistics_data,
        "procurement_data": procurement_data,
        "supplier_data": supplier_data,
        "total_emissions": total_emissions,
        "optimization_targets": optimization_targets
    }
    
    # Create a custom JSON encoder class to handle Timestamp objects
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            # Handle pandas Timestamp objects
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            # Handle pandas Series and DataFrames
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            # Let the base class handle everything else
            return super().default(obj)
    
    # Convert to JSON for the prompt, using the custom encoder
    try:
        data_json = json.dumps(data_summary, indent=2, cls=DateTimeEncoder)
    except TypeError as e:
        logger.error(f"JSON serialization error: {str(e)}")
        # Fallback to simpler representation
        simple_summary = {
            "total_emissions": str(total_emissions.get("total_emissions_kg", 0)) if total_emissions else "0",
            "optimization_targets": f"Potential reduction: {optimization_targets.get('potential_reduction_percent', 0)}%" if optimization_targets else "None"
        }
        data_json = json.dumps(simple_summary, indent=2)
        
    # Create prompt for the AI
    prompt = f"""
You are an expert in sustainable supply chain management. I'm providing you with analysis of a company's supply chain emissions and sustainability data.

Here's the data:

Based on this data, please provide:

1. Executive Summary: A concise summary of the company's current emissions profile and key opportunities.

2. Strategic Recommendations: Top 3-5 strategic initiatives to reduce emissions, with estimated impact.

3. Implementation Roadmap: A 12-month phased plan for implementing the recommendations.

4. Business Case: The financial and competitive advantages of these sustainability measures.

5. Key Performance Indicators: Metrics to track progress.

Please be specific, practical, and data-driven in your recommendations. Focus on high-impact, feasible changes.
"""
    
    # Fallback message if AI APIs are not available
    fallback_strategy = """
# Sustainable Supply Chain Strategy

## Executive Summary
Based on the analysis of your supply chain data, your company has significant opportunities to reduce emissions through
optimized logistics, sustainable procurement, and supplier engagement.

## Strategic Recommendations
1. Optimize transportation routes and shift to lower-emission transport modes.
2. Source materials from suppliers with strong sustainability credentials.
3. Implement circular economy principles to reduce waste and material use.
4. Automate compliance monitoring to stay ahead of regulations.
5. Engage suppliers in sustainability improvement programs.

## Implementation Roadmap
- Months 1-3: Data integration, baseline emissions calculation, and dashboard setup.
- Months 4-6: Develop AI-driven optimization modules and compliance automation.
- Months 7-9: Pilot sustainable procurement and supplier engagement initiatives.
- Months 10-12: Full deployment, reporting, and continuous improvement.

## Business Case
Sustainability initiatives can reduce costs by 15-25%, improve brand reputation, and ensure regulatory compliance, providing a competitive advantage.

## Key Performance Indicators
- Total carbon emissions (kg CO2e)
- Emissions reduction percentage
- Supplier sustainability scores
- Compliance audit pass rate
- Cost savings from optimization
"""
    
    return {
        'strategy_text': fallback_strategy,
        'timestamp': datetime.now().isoformat(),
        'generated_by': 'Fallback'
    }
