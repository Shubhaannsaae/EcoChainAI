"""
Compliance utilities for the Supply Chain Optimizer.
Provides functions to check regulatory compliance and generate compliance reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import json

from app.utils.api_clients import query_claude, get_facility_compliance
from app.config import REGULATION_FRAMEWORKS, get_logger

logger = get_logger(__name__)

# Global regulations database
REGULATIONS = {
    "GHG Protocol": {
        "description": "Global standard for measuring and managing greenhouse gas emissions",
        "requirements": [
            "Scope 1, 2, and 3 emissions accounting",
            "Annual reporting",
            "Third-party verification recommended",
        ],
        "regions": ["Global"],
        "industry_applicability": ["All"],
        "risk_level": "Medium",
    },
    "EU CSRD": {
        "description": "Corporate Sustainability Reporting Directive requiring detailed sustainability reporting",
        "requirements": [
            "Detailed sustainability reporting",
            "Double materiality assessment",
            "Value chain emissions",
            "Third-party verification required",
        ],
        "regions": ["EU"],
        "industry_applicability": ["All"],
        "compliance_date": "2024-01-01",
        "risk_level": "High",
    },
    "REACH": {
        "description": "Registration, Evaluation, Authorization and Restriction of Chemicals",
        "requirements": [
            "Register chemical substances",
            "Manage substances of very high concern (SVHC)",
            "Provide safety information",
        ],
        "regions": ["EU"],
        "industry_applicability": [
            "Chemicals",
            "Manufacturing",
            "Electronics",
            "Textiles",
        ],
        "risk_level": "High",
    },
    "EPA Clean Air Act": {
        "description": "US regulations for air pollution control",
        "requirements": [
            "Emission limits for hazardous air pollutants",
            "Permits for major emission sources",
            "Monitoring and reporting",
        ],
        "regions": ["US"],
        "industry_applicability": ["Manufacturing", "Chemicals", "Energy"],
        "risk_level": "Medium",
    },
    "California SB 253": {
        "description": "Climate Corporate Data Accountability Act requiring emissions disclosure",
        "requirements": [
            "Annual public disclosure of Scope 1, 2, and 3 emissions",
            "Third-party verification",
            "Applies to companies with >$1B annual revenue doing business in California",
        ],
        "regions": ["US"],
        "industry_applicability": ["All"],
        "compliance_date": "2026-01-01",
        "risk_level": "Medium",
    },
    "UK Modern Slavery Act": {
        "description": "Requires businesses to publish annual statements on steps taken to address modern slavery",
        "requirements": [
            "Annual modern slavery statement",
            "Supply chain due diligence",
            "Board approval of statement",
        ],
        "regions": ["UK"],
        "industry_applicability": ["All"],
        "risk_level": "Medium",
    },
    "ISO 14001": {
        "description": "Environmental management system standard",
        "requirements": [
            "Environmental policy",
            "Environmental aspects identification",
            "Legal compliance",
            "Continuous improvement",
        ],
        "regions": ["Global"],
        "industry_applicability": ["All"],
        "risk_level": "Low",
    },
}


def identify_applicable_regulations(
    company_region: str,
    industry_sector: str,
    revenue_usd: Optional[float] = None,
    employee_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Identify regulations applicable to a company based on region, industry, and size

    Args:
        company_region: Region where the company operates (e.g., 'US', 'EU')
        industry_sector: Industry sector (e.g., 'Manufacturing', 'Retail')
        revenue_usd: Annual revenue in USD (optional)
        employee_count: Number of employees (optional)

    Returns:
        dict: Applicable regulations with compliance assessment
    """
    applicable_regulations = {}

    for reg_name, reg_info in REGULATIONS.items():
        is_applicable = False

        # Check region applicability
        region_match = False
        if "Global" in reg_info["regions"]:
            region_match = True
        elif company_region in reg_info["regions"]:
            region_match = True

        # Check industry applicability
        industry_match = False
        if "All" in reg_info["industry_applicability"]:
            industry_match = True
        elif industry_sector in reg_info["industry_applicability"]:
            industry_match = True

        # Size thresholds (if applicable)
        size_match = True
        if "size_threshold" in reg_info:
            if revenue_usd and "min_revenue" in reg_info["size_threshold"]:
                if revenue_usd < reg_info["size_threshold"]["min_revenue"]:
                    size_match = False

            if employee_count and "min_employees" in reg_info["size_threshold"]:
                if employee_count < reg_info["size_threshold"]["min_employees"]:
                    size_match = False

        is_applicable = region_match and industry_match and size_match

        if is_applicable:
            # Add to applicable regulations
            applicable_regulations[reg_name] = {
                "description": reg_info["description"],
                "requirements": reg_info["requirements"],
                "risk_level": reg_info["risk_level"],
                "compliance_date": reg_info.get("compliance_date", "Current"),
            }

    return applicable_regulations


def check_emissions_disclosure_compliance(
    total_emissions: Dict[str, Any], applicable_regulations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check compliance with emissions disclosure regulations

    Args:
        total_emissions: Dictionary containing emissions data
        applicable_regulations: Dictionary of applicable regulations

    Returns:
        dict: Compliance assessment
    """
    compliance_results = {}

    # Extract emissions data
    emissions_by_scope = total_emissions.get("emissions_by_scope", {})
    scope1 = emissions_by_scope.get("scope1", 0)
    scope2 = emissions_by_scope.get("scope2", 0)
    scope3 = emissions_by_scope.get("scope3", 0)

    # Check GHG Protocol compliance
    if "GHG Protocol" in applicable_regulations:
        ghg_compliance = {"status": "Compliant", "gaps": [], "recommendations": []}

        # Check if all required scopes are reported
        if scope1 == 0 and scope2 == 0:
            ghg_compliance["status"] = "Non-compliant"
            ghg_compliance["gaps"].append("Missing Scope 1 and 2 emissions data")
            ghg_compliance["recommendations"].append(
                "Implement comprehensive emissions tracking for Scope 1 and 2"
            )

        if scope3 == 0:
            ghg_compliance["status"] = "Partially compliant"
            ghg_compliance["gaps"].append("Missing Scope 3 emissions data")
            ghg_compliance["recommendations"].append(
                "Develop Scope 3 emissions tracking methodology"
            )

        compliance_results["GHG Protocol"] = ghg_compliance

    # Check EU CSRD compliance
    if "EU CSRD" in applicable_regulations:
        csrd_compliance = {
            "status": "Unknown",
            "gaps": [
                "CSRD requires detailed sustainability reporting beyond emissions"
            ],
            "recommendations": [
                "Conduct double materiality assessment",
                "Develop comprehensive sustainability reporting framework",
                "Prepare for third-party verification",
            ],
        }

        compliance_results["EU CSRD"] = csrd_compliance

    # Check California SB 253 compliance
    if "California SB 253" in applicable_regulations:
        sb253_compliance = {"status": "Preparing", "gaps": [], "recommendations": []}

        if scope1 == 0 or scope2 == 0:
            sb253_compliance["gaps"].append(
                "Missing complete Scope 1 and 2 emissions data"
            )
            sb253_compliance["recommendations"].append(
                "Implement tracking for all Scope 1 and 2 emission sources"
            )

        if scope3 == 0:
            sb253_compliance["gaps"].append("Missing Scope 3 emissions data")
            sb253_compliance["recommendations"].append(
                "Develop comprehensive Scope 3 emissions inventory"
            )

        sb253_compliance["recommendations"].append(
            "Prepare for third-party verification of emissions data"
        )

        compliance_results["California SB 253"] = sb253_compliance

    return compliance_results


def check_chemical_compliance(
    procurement_df: pd.DataFrame, applicable_regulations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check compliance with chemical regulations like REACH

    Args:
        procurement_df: DataFrame with procurement data
        applicable_regulations: Dictionary of applicable regulations

    Returns:
        dict: Compliance assessment
    """
    compliance_results = {}

    # Check if REACH is applicable
    if "REACH" not in applicable_regulations:
        return compliance_results

    reach_compliance = {"status": "Unknown", "gaps": [], "recommendations": []}

    # Check if we have product categories for chemicals
    if "product_category" in procurement_df.columns:
        chemical_products = procurement_df[
            procurement_df["product_category"] == "Chemicals"
        ]

        if len(chemical_products) > 0:
            reach_compliance["status"] = "Needs assessment"
            reach_compliance["gaps"].append(
                "Potential chemical substances requiring REACH registration"
            )
            reach_compliance["recommendations"].append(
                "Conduct REACH substance inventory for chemical products"
            )
            reach_compliance["recommendations"].append(
                "Verify REACH registration status with suppliers"
            )
        else:
            reach_compliance["status"] = "Likely compliant"
            reach_compliance["recommendations"].append(
                "Verify no REACH substances in non-chemical products"
            )
    else:
        reach_compliance["status"] = "Unknown"
        reach_compliance["gaps"].append("Insufficient data to assess REACH compliance")
        reach_compliance["recommendations"].append(
            "Collect detailed chemical inventory data"
        )

    compliance_results["REACH"] = reach_compliance

    return compliance_results


def check_facility_compliance(zip_codes: List[str]) -> Dict[str, Any]:
    """
    Check compliance status of facilities at provided zip codes using EPA data

    Args:
        zip_codes: List of ZIP codes where facilities are located

    Returns:
        dict: Compliance assessment by facility
    """
    compliance_results = {}

    for zip_code in zip_codes:
        try:
            facility_data = get_facility_compliance(zip_code)

            if facility_data:
                compliance_results[zip_code] = {
                    "facilities": facility_data,
                    "status": "Retrieved",
                    "non_compliant_count": sum(
                        1
                        for f in facility_data
                        if f.get("compliance_status") != "In Compliance"
                    ),
                }
            else:
                compliance_results[zip_code] = {
                    "facilities": [],
                    "status": "No data found",
                    "non_compliant_count": 0,
                }

        except Exception as e:
            logger.error(
                f"Error checking facility compliance for ZIP {zip_code}: {str(e)}"
            )
            compliance_results[zip_code] = {
                "facilities": [],
                "status": f"Error: {str(e)}",
                "non_compliant_count": 0,
            }

    return compliance_results


def generate_compliance_report(
    company_region: str,
    industry_sector: str,
    total_emissions: Dict[str, Any],
    procurement_df: Optional[pd.DataFrame] = None,
    facility_zip_codes: Optional[List[str]] = None,
    revenue_usd: Optional[float] = None,
    employee_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive compliance report

    Args:
        company_region: Region where the company operates
        industry_sector: Industry sector
        total_emissions: Dictionary containing emissions data
        procurement_df: DataFrame with procurement data (optional)
        facility_zip_codes: List of facility ZIP codes (optional)
        revenue_usd: Annual revenue in USD (optional)
        employee_count: Number of employees (optional)

    Returns:
        dict: Comprehensive compliance report
    """
    # Identify applicable regulations
    applicable_regulations = identify_applicable_regulations(
        company_region=company_region,
        industry_sector=industry_sector,
        revenue_usd=revenue_usd,
        employee_count=employee_count,
    )

    # Check emissions disclosure compliance
    emissions_compliance = check_emissions_disclosure_compliance(
        total_emissions=total_emissions, applicable_regulations=applicable_regulations
    )

    # Check chemical compliance if procurement data is available
    chemical_compliance = {}
    if procurement_df is not None:
        chemical_compliance = check_chemical_compliance(
            procurement_df=procurement_df, applicable_regulations=applicable_regulations
        )

    # Check facility compliance if ZIP codes are provided
    facility_compliance = {}
    if facility_zip_codes:
        facility_compliance = check_facility_compliance(zip_codes=facility_zip_codes)

    # Count compliance issues
    compliance_issues = 0
    for regulation, assessment in emissions_compliance.items():
        if assessment["status"] != "Compliant":
            compliance_issues += len(assessment["gaps"])

    for regulation, assessment in chemical_compliance.items():
        if (
            assessment["status"] != "Compliant"
            and assessment["status"] != "Likely compliant"
        ):
            compliance_issues += len(assessment["gaps"])

    # Facility compliance issues
    facility_issues = sum(
        data["non_compliant_count"] for zip_code, data in facility_compliance.items()
    )
    compliance_issues += facility_issues

    # Generate AI insights if available
    compliance_insights = generate_ai_compliance_insights(
        applicable_regulations=applicable_regulations,
        emissions_compliance=emissions_compliance,
        chemical_compliance=chemical_compliance,
        facility_compliance=facility_compliance,
        company_region=company_region,
        industry_sector=industry_sector,
    )

    # Compile overall compliance report
    compliance_report = {
        "timestamp": datetime.now().isoformat(),
        "company_profile": {
            "region": company_region,
            "industry": industry_sector,
            "revenue_usd": revenue_usd,
            "employee_count": employee_count,
        },
        "applicable_regulations": applicable_regulations,
        "emissions_compliance": emissions_compliance,
        "chemical_compliance": chemical_compliance,
        "facility_compliance": facility_compliance,
        "compliance_issues": compliance_issues,
        "compliance_insights": compliance_insights,
        "upcoming_regulations": identify_upcoming_regulations(
            company_region, industry_sector
        ),
    }

    return compliance_report


def identify_upcoming_regulations(
    company_region: str, industry_sector: str
) -> List[Dict[str, Any]]:
    """
    Identify upcoming regulations relevant to the company

    Args:
        company_region: Region where the company operates
        industry_sector: Industry sector

    Returns:
        list: List of upcoming regulations
    """
    upcoming_regulations = []

    for reg_name, reg_info in REGULATIONS.items():
        # Check if regulation has a future compliance date
        if "compliance_date" in reg_info:
            compliance_date = datetime.strptime(reg_info["compliance_date"], "%Y-%m-%d")

            if compliance_date > datetime.now():
                # Check if applicable to company region and industry
                region_applicable = False
                if (
                    "Global" in reg_info["regions"]
                    or company_region in reg_info["regions"]
                ):
                    region_applicable = True

                industry_applicable = False
                if (
                    "All" in reg_info["industry_applicability"]
                    or industry_sector in reg_info["industry_applicability"]
                ):
                    industry_applicable = True

                if region_applicable and industry_applicable:
                    days_until_compliance = (compliance_date - datetime.now()).days

                    upcoming_regulations.append(
                        {
                            "name": reg_name,
                            "description": reg_info["description"],
                            "compliance_date": reg_info["compliance_date"],
                            "days_until_compliance": days_until_compliance,
                            "requirements": reg_info["requirements"],
                            "risk_level": reg_info["risk_level"],
                        }
                    )

    # Sort by compliance date (soonest first)
    upcoming_regulations.sort(key=lambda x: x["days_until_compliance"])

    return upcoming_regulations


def generate_ai_compliance_insights(
    applicable_regulations: Dict[str, Any],
    emissions_compliance: Dict[str, Any],
    chemical_compliance: Dict[str, Any],
    facility_compliance: Dict[str, Any],
    company_region: str,
    industry_sector: str,
) -> str:
    """
    Generate AI insights on compliance status and recommendations

    Args:
        applicable_regulations: Dictionary of applicable regulations
        emissions_compliance: Emissions compliance assessment
        chemical_compliance: Chemical compliance assessment
        facility_compliance: Facility compliance assessment
        company_region: Region where the company operates
        industry_sector: Industry sector

    Returns:
        str: AI-generated compliance insights
    """
    # Compile data for the AI prompt
    compliance_data = {
        "applicable_regulations": applicable_regulations,
        "emissions_compliance": emissions_compliance,
        "chemical_compliance": chemical_compliance,
        "facility_compliance": facility_compliance,
        "company_profile": {"region": company_region, "industry": industry_sector},
    }

    # Convert to JSON for the prompt
    data_json = json.dumps(compliance_data, indent=2)

    # Create prompt for the AI
    prompt = f"""
You are an expert in environmental regulatory compliance. I'm providing you with a compliance assessment for a company in the {industry_sector} industry, operating in {company_region}.

Here's the compliance data:

Please provide:

1. Compliance Summary: A brief overview of the company's current compliance status
2. Key Compliance Risks: The most critical compliance issues that need immediate attention
3. Strategic Recommendations: Specific actions to improve compliance
4. Regulatory Outlook: Insights on how the regulatory landscape is evolving for this industry and region

Keep your response concise and actionable, focusing on practical steps the company can take.
"""

    try:
        # Try to use Claude for insight generation
        insights = query_claude(prompt)
        return insights

    except Exception as e:
        logger.warning(f"Error using AI for compliance insights: {str(e)}")

        # Return basic insights if AI fails
        return f"""
## Compliance Summary
Based on the assessment, your company has several compliance requirements related to operating in {company_region} in the {industry_sector} industry. Key regulations include {', '.join(applicable_regulations.keys())}.

## Key Compliance Risks
- Ensure all required emissions data is being tracked and reported
- Verify chemical products comply with applicable regulations
- Address any facility compliance issues identified by regulatory authorities

## Strategic Recommendations
1. Implement a structured compliance management system
2. Conduct regular internal audits against regulatory requirements
3. Stay informed about upcoming regulatory changes
4. Consider third-party verification of compliance status

## Regulatory Outlook
Environmental regulations continue to become more stringent globally, with increasing focus on emissions reporting, chemical safety, and supply chain responsibility. Companies that proactively manage compliance will be better positioned for future requirements.
"""
