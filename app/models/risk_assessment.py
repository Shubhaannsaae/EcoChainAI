"""
Risk assessment models for evaluating supply chain sustainability risks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime

from app.config import get_logger

logger = get_logger(__name__)

# Risk factors weights
RISK_FACTOR_WEIGHTS = {
    # Environmental risk factors
    "sustainability_score": 0.15,
    "carbon_footprint": 0.10,
    "environmental_violations": 0.08,
    "water_stress": 0.05,
    "deforestation_risk": 0.05,
    
    # Business risk factors
    "financial_stability": 0.12,
    "geopolitical_risk": 0.10,
    "lead_time_variability": 0.07,
    "single_source_risk": 0.08,
    
    # Compliance risk factors
    "regulatory_compliance": 0.12,
    "labor_practices": 0.08
}

# Country risk ratings (example data)
COUNTRY_RISK_RATINGS = {
    "USA": {"geopolitical_risk": 2.0, "water_stress": 4.0, "deforestation_risk": 2.0, "regulatory_compliance": 2.0, "labor_practices": 2.0},
    "China": {"geopolitical_risk": 6.0, "water_stress": 7.0, "deforestation_risk": 5.0, "regulatory_compliance": 5.0, "labor_practices": 6.0},
    "Germany": {"geopolitical_risk": 2.0, "water_stress": 3.0, "deforestation_risk": 2.0, "regulatory_compliance": 2.0, "labor_practices": 2.0},
    "India": {"geopolitical_risk": 5.0, "water_stress": 8.0, "deforestation_risk": 5.0, "regulatory_compliance": 5.0, "labor_practices": 6.0},
    "Brazil": {"geopolitical_risk": 4.0, "water_stress": 4.0, "deforestation_risk": 8.0, "regulatory_compliance": 4.0, "labor_practices": 4.0},
    "Mexico": {"geopolitical_risk": 5.0, "water_stress": 7.0, "deforestation_risk": 6.0, "regulatory_compliance": 5.0, "labor_practices": 5.0},
    "United Kingdom": {"geopolitical_risk": 3.0, "water_stress": 3.0, "deforestation_risk": 2.0, "regulatory_compliance": 2.0, "labor_practices": 2.0},
    "Japan": {"geopolitical_risk": 3.0, "water_stress": 3.0, "deforestation_risk": 2.0, "regulatory_compliance": 2.0, "labor_practices": 2.0},
    "Indonesia": {"geopolitical_risk": 4.0, "water_stress": 5.0, "deforestation_risk": 9.0, "regulatory_compliance": 6.0, "labor_practices": 6.0},
    "Bangladesh": {"geopolitical_risk": 6.0, "water_stress": 7.0, "deforestation_risk": 5.0, "regulatory_compliance": 7.0, "labor_practices": 8.0}
}

# Default country risk values for unknown countries
DEFAULT_COUNTRY_RISK = {"geopolitical_risk": 5.0, "water_stress": 5.0, "deforestation_risk": 5.0, "regulatory_compliance": 5.0, "labor_practices": 5.0}

# Industry risk factors
INDUSTRY_RISK_FACTORS = {
    "Manufacturing": {"carbon_footprint": 7.0, "water_stress": 6.0, "deforestation_risk": 4.0, "labor_practices": 5.0},
    "Electronics": {"carbon_footprint": 6.0, "water_stress": 4.0, "deforestation_risk": 2.0, "labor_practices": 4.0},
    "Packaging": {"carbon_footprint": 5.0, "water_stress": 6.0, "deforestation_risk": 7.0, "labor_practices": 4.0},
    "Raw Materials": {"carbon_footprint": 8.0, "water_stress": 7.0, "deforestation_risk": 8.0, "labor_practices": 6.0},
    "Office Supplies": {"carbon_footprint": 4.0, "water_stress": 3.0, "deforestation_risk": 6.0, "labor_practices": 3.0},
    "Chemicals": {"carbon_footprint": 8.0, "water_stress": 8.0, "deforestation_risk": 5.0, "labor_practices": 6.0},
    "Retail": {"carbon_footprint": 5.0, "water_stress": 3.0, "deforestation_risk": 4.0, "labor_practices": 5.0},
    "Food & Beverage": {"carbon_footprint": 6.0, "water_stress": 9.0, "deforestation_risk": 7.0, "labor_practices": 6.0},
    "Textiles": {"carbon_footprint": 7.0, "water_stress": 8.0, "deforestation_risk": 5.0, "labor_practices": 7.0},
    "Automotive": {"carbon_footprint": 7.0, "water_stress": 5.0, "deforestation_risk": 4.0, "labor_practices": 5.0}
}

# Default industry risk values for unknown industries
DEFAULT_INDUSTRY_RISK = {"carbon_footprint": 6.0, "water_stress": 5.0, "deforestation_risk": 5.0, "labor_practices": 5.0}

def calculate_supplier_risk_scores(supplier_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive risk scores for suppliers
    
    Args:
        supplier_df: DataFrame with supplier data
        
    Returns:
        DataFrame with added risk scores
    """
    if supplier_df is None or len(supplier_df) == 0:
        logger.warning("Empty dataframe provided to calculate_supplier_risk_scores")
        return pd.DataFrame()
    
    # Create a copy of the dataframe
    result_df = supplier_df.copy()
    
    # Initialize risk score columns
    result_df['environmental_risk_score'] = 0.0
    result_df['business_risk_score'] = 0.0
    result_df['compliance_risk_score'] = 0.0
    result_df['overall_risk_score'] = 0.0
    result_df['risk_category'] = ''
    
    # Process each supplier
    for idx, supplier in result_df.iterrows():
        # Get country and category
        country = supplier.get('country', 'Unknown')
        if pd.isna(country) or country == '':
            country = 'Unknown'
        
        category = supplier.get('category', 'Unknown')
        if pd.isna(category) or category == '':
            category = 'Unknown'
        
        # Get country risk ratings
        country_risk = COUNTRY_RISK_RATINGS.get(country, DEFAULT_COUNTRY_RISK)
        
        # Get industry risk factors
        industry_risk = INDUSTRY_RISK_FACTORS.get(category, DEFAULT_INDUSTRY_RISK)
        
        # Calculate Environmental Risk
        env_risk_factors = {}
        
        # Use sustainability score if available
        if 'sustainability_score' in supplier and not pd.isna(supplier['sustainability_score']):
            # Convert 0-10 score to 0-10 risk (invert)
            env_risk_factors['sustainability_score'] = 10 - supplier['sustainability_score']
        else:
            # Estimate from other factors
            env_risk_factors['sustainability_score'] = 5.0  # Default mid-level risk
        
        # Use carbon footprint if available
        if 'carbon_footprint_tons' in supplier and not pd.isna(supplier['carbon_footprint_tons']):
            # Normalize carbon footprint to 0-10 scale (higher is higher risk)
            # Use a log scale to handle varying magnitudes
            carbon_value = supplier['carbon_footprint_tons']
            if carbon_value > 0:
                carbon_score = min(10, 2 * np.log10(carbon_value))
            else:
                carbon_score = 1.0
            env_risk_factors['carbon_footprint'] = carbon_score
        else:
            # Use industry default
            env_risk_factors['carbon_footprint'] = industry_risk['carbon_footprint']
        
        # Environmental violations - use country rating if not available
        env_risk_factors['environmental_violations'] = supplier.get('environmental_violations', 5.0)
        if pd.isna(env_risk_factors['environmental_violations']):
            env_risk_factors['environmental_violations'] = country_risk['regulatory_compliance']
        
        # Water stress - combine country and industry ratings
        env_risk_factors['water_stress'] = (
            country_risk['water_stress'] * 0.6 + 
            industry_risk['water_stress'] * 0.4
        )
        
        # Deforestation risk - combine country and industry ratings
        env_risk_factors['deforestation_risk'] = (
            country_risk['deforestation_risk'] * 0.7 + 
            industry_risk['deforestation_risk'] * 0.3
        )
        
        # Calculate weighted environmental risk score
        env_risk_score = 0
        env_weights_sum = 0
        
        for factor, value in env_risk_factors.items():
            weight = RISK_FACTOR_WEIGHTS.get(factor, 0)
            env_risk_score += value * weight
            env_weights_sum += weight
        
        # Normalize to account for missing factors
        if env_weights_sum > 0:
            env_risk_score = env_risk_score / env_weights_sum * 10
        else:
            env_risk_score = 5.0  # Default mid-level risk
        
        # Calculate Business Risk
        business_risk_factors = {}
        
        # Financial stability - use risk_score if available
        if 'risk_score' in supplier and not pd.isna(supplier['risk_score']):
            business_risk_factors['financial_stability'] = supplier['risk_score']
        else:
            business_risk_factors['financial_stability'] = 5.0  # Default mid-level risk
        
        # Geopolitical risk - use country rating
        business_risk_factors['geopolitical_risk'] = country_risk['geopolitical_risk']
        
        # Lead time variability
        if 'lead_time_days' in supplier and not pd.isna(supplier['lead_time_days']):
            # Higher lead times generally indicate higher risk
            lead_time = supplier['lead_time_days']
            lead_time_risk = min(10, lead_time / 10)  # Scale to 0-10
            business_risk_factors['lead_time_variability'] = lead_time_risk
        else:
            business_risk_factors['lead_time_variability'] = 5.0  # Default mid-level risk
        
        # Single source risk - set to high (8.0) if this is the only supplier for a category
        # This would require checking against all suppliers, for now use a default
        business_risk_factors['single_source_risk'] = 5.0  # Default mid-level risk
        
        # Calculate weighted business risk score
        business_risk_score = 0
        business_weights_sum = 0
        
        for factor, value in business_risk_factors.items():
            weight = RISK_FACTOR_WEIGHTS.get(factor, 0)
            business_risk_score += value * weight
            business_weights_sum += weight
        
        # Normalize to account for missing factors
        if business_weights_sum > 0:
            business_risk_score = business_risk_score / business_weights_sum * 10
        else:
            business_risk_score = 5.0  # Default mid-level risk
        
        # Calculate Compliance Risk
        compliance_risk_factors = {}
        
        # Regulatory compliance - use country rating
        compliance_risk_factors['regulatory_compliance'] = country_risk['regulatory_compliance']
        
        # Labor practices - combine country and industry ratings
        compliance_risk_factors['labor_practices'] = (
            country_risk['labor_practices'] * 0.7 + 
            industry_risk['labor_practices'] * 0.3
        )
        
        # Calculate weighted compliance risk score
        compliance_risk_score = 0
        compliance_weights_sum = 0
        
        for factor, value in compliance_risk_factors.items():
            weight = RISK_FACTOR_WEIGHTS.get(factor, 0)
            compliance_risk_score += value * weight
            compliance_weights_sum += weight
        
        # Normalize to account for missing factors
        if compliance_weights_sum > 0:
            compliance_risk_score = compliance_risk_score / compliance_weights_sum * 10
        else:
            compliance_risk_score = 5.0  # Default mid-level risk
        
        # Calculate overall risk score (weighted average of the three categories)
        overall_risk_score = (
            env_risk_score * 0.4 +
            business_risk_score * 0.35 +
            compliance_risk_score * 0.25
        )
        
        # Determine risk category
        if overall_risk_score >= 7.5:
            risk_category = "High"
        elif overall_risk_score >= 5:
            risk_category = "Medium"
        else:
            risk_category = "Low"
        
        # Store results
        result_df.at[idx, 'environmental_risk_score'] = round(env_risk_score, 1)
        result_df.at[idx, 'business_risk_score'] = round(business_risk_score, 1)
        result_df.at[idx, 'compliance_risk_score'] = round(compliance_risk_score, 1)
        result_df.at[idx, 'overall_risk_score'] = round(overall_risk_score, 1)
        result_df.at[idx, 'risk_category'] = risk_category
    
    return result_df

def identify_risk_hotspots(supplier_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify risk hotspots in the supply chain
    
    Args:
        supplier_df: DataFrame with supplier risk scores
        
    Returns:
        Dictionary with risk hotspot information
    """
    if supplier_df is None or len(supplier_df) == 0:
        return {
            "high_risk_suppliers": [],
            "high_risk_categories": {},
            "high_risk_countries": {},
            "top_risks": []
        }
    
    # Ensure risk scores are calculated
    if 'overall_risk_score' not in supplier_df.columns:
        supplier_df = calculate_supplier_risk_scores(supplier_df)
    
    # Identify high-risk suppliers
    high_risk_suppliers = supplier_df[supplier_df['risk_category'] == 'High']
    
    # Extract supplier names or IDs
    high_risk_supplier_info = []
    for _, supplier in high_risk_suppliers.iterrows():
        supplier_info = {
            'supplier_id': supplier.get('supplier_id', 'Unknown'),
            'overall_risk_score': supplier['overall_risk_score']
        }
        
        if 'supplier_name' in supplier:
            supplier_info['supplier_name'] = supplier['supplier_name']
        
        if 'category' in supplier:
            supplier_info['category'] = supplier['category']
        
        if 'country' in supplier:
            supplier_info['country'] = supplier['country']
        
        high_risk_supplier_info.append(supplier_info)
    
    # Identify high-risk categories
    if 'category' in supplier_df.columns:
        category_risks = supplier_df.groupby('category')['overall_risk_score'].mean().sort_values(ascending=False)
        high_risk_categories = category_risks[category_risks >= 6.0].to_dict()
    else:
        high_risk_categories = {}
    
    # Identify high-risk countries
    if 'country' in supplier_df.columns:
        country_risks = supplier_df.groupby('country')['overall_risk_score'].mean().sort_values(ascending=False)
        high_risk_countries = country_risks[country_risks >= 6.0].to_dict()
    else:
        high_risk_countries = {}
    
    # Identify top risk factors
    risk_columns = [
        'environmental_risk_score', 
        'business_risk_score', 
        'compliance_risk_score'
    ]
    
    risk_scores = {
        'Environmental': supplier_df['environmental_risk_score'].mean() if 'environmental_risk_score' in supplier_df.columns else 5.0,
        'Business': supplier_df['business_risk_score'].mean() if 'business_risk_score' in supplier_df.columns else 5.0,
        'Compliance': supplier_df['compliance_risk_score'].mean() if 'compliance_risk_score' in supplier_df.columns else 5.0
    }
    
    top_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare result
    result = {
        "high_risk_suppliers": high_risk_supplier_info,
        "high_risk_categories": high_risk_categories,
        "high_risk_countries": high_risk_countries,
        "top_risks": top_risks,
        "average_risk_score": supplier_df['overall_risk_score'].mean() if 'overall_risk_score' in supplier_df.columns else 5.0
    }
    
    return result

def generate_risk_mitigation_strategies(risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate risk mitigation strategies based on risk assessment
    
    Args:
        risk_assessment: Dictionary with risk assessment results
        
    Returns:
        List of risk mitigation strategies
    """
    strategies = []
    
    # High-risk suppliers
    high_risk_suppliers = risk_assessment.get('high_risk_suppliers', [])
    if high_risk_suppliers:
        strategies.append({
            'risk_area': 'High-Risk Suppliers',
            'risk_level': 'High',
            'strategy': 'Implement enhanced monitoring and auditing for high-risk suppliers',
            'actions': [
                'Conduct quarterly sustainability audits',
                'Implement monthly performance reviews',
                'Require enhanced documentation and certifications',
                'Consider supplier diversification for critical materials'
            ],
            'implementation_timeline': '3-6 months',
            'responsible_party': 'Procurement',
            'expected_impact': 'High'
        })
    
    # High-risk categories
    high_risk_categories = risk_assessment.get('high_risk_categories', {})
    if high_risk_categories:
        top_category = next(iter(high_risk_categories))
        strategies.append({
            'risk_area': f'High-Risk Category: {top_category}',
            'risk_level': 'High',
            'strategy': f'Develop category-specific sustainability program for {top_category}',
            'actions': [
                'Map all suppliers in the category',
                'Establish category-specific sustainability criteria',
                'Identify alternative suppliers with better sustainability profiles',
                'Set specific reduction targets for the category'
            ],
            'implementation_timeline': '6-12 months',
            'responsible_party': 'Procurement & Sustainability',
            'expected_impact': 'High'
        })
    
    # High-risk countries
    high_risk_countries = risk_assessment.get('high_risk_countries', {})
    if high_risk_countries:
        top_country = next(iter(high_risk_countries))
        strategies.append({
            'risk_area': f'Geographic Risk: {top_country}',
            'risk_level': 'Medium',
            'strategy': f'Develop country-specific risk management for {top_country}',
            'actions': [
                'Conduct enhanced due diligence for suppliers in this region',
                'Develop backup suppliers in different regions',
                'Implement country-specific compliance checklists',
                'Monitor geopolitical and environmental developments in the region'
            ],
            'implementation_timeline': '3-6 months',
            'responsible_party': 'Risk Management & Procurement',
            'expected_impact': 'Medium'
        })
    
    # Top risk factors
    top_risks = risk_assessment.get('top_risks', [])
    if top_risks and len(top_risks) > 0:
        top_risk_area = top_risks[0][0]
        
        if top_risk_area == 'Environmental':
            strategies.append({
                'risk_area': 'Environmental Risk',
                'risk_level': 'High',
                'strategy': 'Comprehensive environmental risk reduction program',
                'actions': [
                    'Implement supplier environmental scorecard',
                    'Set science-based targets for emissions reduction',
                    'Develop water stewardship program for high water-stress regions',
                    'Implement deforestation-free sourcing policy'
                ],
                'implementation_timeline': '6-12 months',
                'responsible_party': 'Sustainability & Procurement',
                'expected_impact': 'High'
            })
        elif top_risk_area == 'Business':
            strategies.append({
                'risk_area': 'Business Continuity Risk',
                'risk_level': 'High',
                'strategy': 'Supply chain resilience enhancement',
                'actions': [
                    'Develop multi-sourcing strategy for critical materials',
                    'Implement supplier financial health monitoring',
                    'Establish buffer inventory for high-risk materials',
                    'Develop contingency plans for supply disruptions'
                ],
                'implementation_timeline': '3-9 months',
                'responsible_party': 'Procurement & Operations',
                'expected_impact': 'High'
            })
        elif top_risk_area == 'Compliance':
            strategies.append({
                'risk_area': 'Regulatory Compliance Risk',
                'risk_level': 'High',
                'strategy': 'Enhanced compliance management system',
                'actions': [
                    'Implement automated compliance monitoring',
                    'Develop supplier code of conduct with verification process',
                    'Establish compliance training program for procurement team',
                    'Implement third-party compliance audits for high-risk suppliers'
                ],
                'implementation_timeline': '6-12 months',
                'responsible_party': 'Legal & Compliance',
                'expected_impact': 'High'
            })
    
    # Add general strategy regardless of specific risks
    strategies.append({
        'risk_area': 'Overall Supply Chain Sustainability',
        'risk_level': 'Medium',
        'strategy': 'Integrated supply chain sustainability program',
        'actions': [
            'Develop comprehensive supplier evaluation criteria including sustainability metrics',
            'Implement regular sustainability risk assessments',
            'Establish supplier engagement program for continuous improvement',
            'Integrate sustainability criteria into procurement decisions'
        ],
                'implementation_timeline': '12-18 months',
        'responsible_party': 'Cross-functional team',
        'expected_impact': 'High'
    })
    
    return strategies

def simulate_risk_scenarios(
    supplier_df: pd.DataFrame,
    risk_assessment: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulate different risk scenarios to assess potential impact
    
    Args:
        supplier_df: DataFrame with supplier data
        risk_assessment: Dictionary with risk assessment results
        
    Returns:
        Dictionary with risk scenario simulation results
    """
    scenarios = {}
    
    # Check if we have necessary data
    if supplier_df is None or len(supplier_df) == 0:
        return {"error": "Insufficient data for scenario simulation"}
    
    # Ensure risk scores are calculated
    if 'overall_risk_score' not in supplier_df.columns:
        supplier_df = calculate_supplier_risk_scores(supplier_df)
    
    # Scenario 1: Major supplier disruption
    # Simulate disruption of the highest risk supplier
    if len(supplier_df) > 0:
        high_risk_suppliers = supplier_df.sort_values('overall_risk_score', ascending=False)
        if len(high_risk_suppliers) > 0:
            top_risk_supplier = high_risk_suppliers.iloc[0]
            
            supplier_name = top_risk_supplier.get('supplier_name', top_risk_supplier.get('supplier_id', 'Unknown'))
            supplier_category = top_risk_supplier.get('category', 'Unknown')
            
            # Calculate impact (this would ideally be based on financial data, volume, etc.)
            # For demo, use a simple estimation
            impact_level = min(9, top_risk_supplier['overall_risk_score'])
            
            scenarios['major_supplier_disruption'] = {
                'name': f'Disruption of high-risk supplier: {supplier_name}',
                'description': f'Complete disruption of supplies from {supplier_name} in the {supplier_category} category due to operational, environmental, or compliance issues.',
                'probability': top_risk_supplier['overall_risk_score'] / 10,  # Scale to 0-1
                'impact_level': impact_level,
                'risk_score': impact_level * (top_risk_supplier['overall_risk_score'] / 10),
                'mitigation_actions': [
                    f'Identify alternative suppliers for {supplier_category}',
                    'Develop emergency sourcing plan',
                    'Build buffer inventory for critical materials',
                    'Implement regular risk monitoring for this supplier'
                ]
            }
    
    # Scenario 2: Regional disruption
    # Simulate disruption in the highest risk country
    high_risk_countries = risk_assessment.get('high_risk_countries', {})
    if high_risk_countries:
        top_country = next(iter(high_risk_countries))
        country_risk_score = high_risk_countries[top_country]
        
        # Count suppliers in this country
        country_suppliers = supplier_df[supplier_df['country'] == top_country]
        supplier_count = len(country_suppliers)
        
        if supplier_count > 0:
            # Calculate potential impact
            impact_level = min(9, country_risk_score * (supplier_count / len(supplier_df) * 2))
            
            scenarios['regional_disruption'] = {
                'name': f'Regional disruption in {top_country}',
                'description': f'Major disruption affecting all {supplier_count} suppliers in {top_country} due to geopolitical issues, natural disasters, or regulatory changes.',
                'probability': country_risk_score / 10,  # Scale to 0-1
                'impact_level': impact_level,
                'risk_score': impact_level * (country_risk_score / 10),
                'mitigation_actions': [
                    f'Diversify supplier base beyond {top_country}',
                    'Develop regional contingency plans',
                    'Implement more frequent monitoring of suppliers in high-risk regions',
                    'Consider nearshoring for critical components'
                ]
            }
    
    # Scenario 3: Regulatory change
    # Simulate impact of new sustainability regulations
    avg_compliance_risk = supplier_df['compliance_risk_score'].mean() if 'compliance_risk_score' in supplier_df.columns else 5.0
    
    scenarios['regulatory_change'] = {
        'name': 'New sustainability regulations',
        'description': 'Implementation of stricter environmental and social governance regulations requiring additional documentation, verification, and potential changes to sourcing practices.',
        'probability': 0.7,  # High probability of regulatory changes in the sustainability space
        'impact_level': avg_compliance_risk * 0.8,
        'risk_score': avg_compliance_risk * 0.8 * 0.7,
        'mitigation_actions': [
            'Establish regulatory monitoring system',
            'Implement proactive compliance program',
            'Engage with industry associations on regulatory developments',
            'Develop compliance readiness assessment for all key suppliers'
        ]
    }
    
    # Scenario 4: Carbon price introduction
    # Simulate impact of carbon pricing
    avg_environmental_risk = supplier_df['environmental_risk_score'].mean() if 'environmental_risk_score' in supplier_df.columns else 5.0
    
    scenarios['carbon_pricing'] = {
        'name': 'Introduction of carbon pricing',
        'description': 'Implementation of carbon pricing mechanisms affecting the cost structure of high-emission suppliers and materials, potentially increasing costs throughout the supply chain.',
        'probability': 0.6,
        'impact_level': avg_environmental_risk * 0.9,
        'risk_score': avg_environmental_risk * 0.9 * 0.6,
        'mitigation_actions': [
            'Conduct carbon footprint assessment of entire supply chain',
            'Identify and prioritize emission reduction opportunities',
            'Develop low-carbon sourcing strategy',
            'Implement internal carbon pricing for procurement decisions'
        ]
    }
    
    # Calculate overall risk landscape
    risk_scores = [scenario['risk_score'] for scenario in scenarios.values()]
    
    overall_result = {
        'scenarios': scenarios,
        'highest_risk_scenario': max(scenarios.items(), key=lambda x: x[1]['risk_score'])[0],
        'average_risk_score': sum(risk_scores) / len(risk_scores) if risk_scores else 0,
        'scenario_count': len(scenarios)
    }
    
    return overall_result
