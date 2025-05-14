"""
Reporting utilities for the Supply Chain Optimizer.
Provides functions to generate reports, charts, and visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import os
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import io
import tempfile
from PIL import Image

from app.config import get_logger

logger = get_logger(__name__)

# In app/dashboard/components.py, update the create_emissions_summary_chart function:
def create_emissions_summary_chart(total_emissions: Dict[str, Any]) -> Optional[str]:
    """
    Create an enhanced chart summarizing emissions by category
    
    Args:
        total_emissions: Dictionary with emissions data
        
    Returns:
        str: Base64 encoded image of the chart or None if failed
    """
    try:
        if not total_emissions or 'emissions_by_category' not in total_emissions:
            logger.warning("No emissions data available for chart creation")
            return None
        
        emissions_by_category = total_emissions['emissions_by_category']
        
        # Sort categories by emissions (descending)
        sorted_categories = sorted(
            emissions_by_category.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Limit to top 8 categories for readability if there are many
        if len(sorted_categories) > 8:
            top_categories = sorted_categories[:7]
            # Group remaining categories as "Other"
            other_emissions = sum(item[1] for item in sorted_categories[7:])
            if other_emissions > 0:
                top_categories.append(("Other", other_emissions))
            sorted_categories = top_categories
        
        categories = [item[0] for item in sorted_categories]
        values = [item[1] for item in sorted_categories]
        
        # Convert to tonnes for better readability
        values_tonnes = [val/1000 for val in values]
        
        # Create the chart with enhanced styling
        plt.figure(figsize=(10, 6))
        
        # Use a more professional style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Generate color gradient based on values (higher values = darker colors)
        # Using viridis colormap which is perceptually uniform and colorblind-friendly
        colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(categories)))
        
        # Create the bar chart
        bars = plt.bar(categories, values_tonnes, color=colors, width=0.7)
        
        # Add data labels with improved formatting
        for bar in bars:
            height = bar.get_height()
            # Only show label if the value is significant enough to display
            if height > 0.1:  # Only label bars with at least 0.1 tonnes
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + max(values_tonnes) * 0.02,  # Position label slightly above bar
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color='#333333'
                )
        
        # Set up the title and labels with better typography
        plt.title('Carbon Emissions by Category (tonnes CO₂e)', 
                 fontsize=14, 
                 fontweight='bold',
                 pad=15,
                 color='#333333')
        
        plt.ylabel('Emissions (tonnes CO₂e)', 
                  fontsize=12, 
                  fontweight='medium',
                  color='#333333')
        
        # Improve x-axis readability
        plt.xticks(rotation=45, 
                   ha='right', 
                   fontsize=10,
                   color='#333333')
        
        # Format y-axis with comma as thousands separator
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:,.1f}"))
        
        # Add subtle grid lines for readability
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Set background colors
        plt.gca().set_facecolor('#f9f9f9')
        plt.gcf().set_facecolor('#ffffff')
        
        # Add a box around the plot
        plt.box(True)
        
        # Add a subtle border to the figure
        plt.gcf().patch.set_linewidth(0.5)
        plt.gcf().patch.set_edgecolor('#dddddd')
        
        # Ensure layout fits properly
        plt.tight_layout()
        
        # Add total emissions as text annotation
        total_tonnes = sum(values_tonnes)
        plt.figtext(
            0.01, 0.01, 
            f"Total: {total_tonnes:.1f} tonnes CO₂e", 
            fontsize=9, 
            color='#555555'
        )
        
        # Save to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # Encode as base64 for embedding in HTML
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
        
    except Exception as e:
        logger.error(f"Error creating emissions summary chart: {str(e)}")
        return None


def create_optimization_impact_chart(
    optimization_targets: Dict[str, Any]
) -> Optional[str]:
    """
    Create a chart showing the impact of optimization recommendations
    
    Args:
        optimization_targets: Dictionary with optimization data
        
    Returns:
        str: Base64 encoded image of the chart or None if failed
    """
    try:
        if not optimization_targets or 'recommendations' not in optimization_targets:
            logger.warning("No optimization data available for chart creation")
            return None
        
        recommendations = optimization_targets['recommendations']
        
        # Extract categories and potential reductions
        categories = [r['category'] for r in recommendations]
        reductions = [r['potential_reduction_kg'] / 1000 for r in recommendations]  # Convert to tonnes
        
        # Create a dictionary to aggregate by category
        category_reductions = {}
        for cat, red in zip(categories, reductions):
            if cat in category_reductions:
                category_reductions[cat] += red
            else:
                category_reductions[cat] = red
        
        # Sort by reduction potential (descending)
        sorted_categories = sorted(
            category_reductions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        cats = [item[0] for item in sorted_categories]
        reds = [item[1] for item in sorted_categories]
        
        # Create the chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(cats, reds, color='#4CAF50')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        plt.title('Potential Emissions Reduction by Category (tonnes CO₂e)')
        plt.ylabel('Reduction Potential (tonnes CO₂e)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        
        # Encode as base64 for embedding in HTML
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
        
    except Exception as e:
        logger.error(f"Error creating optimization impact chart: {str(e)}")
        return None

def create_supplier_sustainability_chart(supplier_df: pd.DataFrame) -> Optional[str]:
    """
    Create a chart showing supplier sustainability scores
    
    Args:
        supplier_df: DataFrame with supplier data
        
    Returns:
        str: Base64 encoded image of the chart or None if failed
    """
    try:
        if supplier_df is None or len(supplier_df) == 0:
            logger.warning("No supplier data available for chart creation")
            return None
        
        if 'sustainability_score' not in supplier_df.columns:
            logger.warning("No sustainability score column in supplier data")
            return None
        
        # Get top and bottom suppliers by sustainability score
        top_suppliers = supplier_df.nlargest(5, 'sustainability_score')
        bottom_suppliers = supplier_df.nsmallest(5, 'sustainability_score')
        
        # Combine into a single dataframe
        plot_df = pd.concat([top_suppliers, bottom_suppliers])
        
        # Get supplier names or IDs
        if 'supplier_name' in plot_df.columns:
            labels = plot_df['supplier_name'].tolist()
        elif 'supplier_id' in plot_df.columns:
            labels = plot_df['supplier_id'].tolist()
        else:
            labels = [f"Supplier {i+1}" for i in range(len(plot_df))]
        
        scores = plot_df['sustainability_score'].tolist()
        
        # Create colors based on scores (green for high, red for low)
        colors = ['#4CAF50' if score >= 7 else
                  '#FFC107' if score >= 5 else
                  '#F44336' for score in scores]
        
        # Create the chart
        plt.figure(figsize=(12, 6))
        bars = plt.barh(labels, scores, color=colors)
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.1,
                bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}',
                ha='left',
                va='center',
                fontsize=9
            )
        
        plt.title('Supplier Sustainability Scores (Top 5 and Bottom 5)')
        plt.xlabel('Sustainability Score (0-10)')
        plt.xlim(0, 10)
        plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)  # Add reference line
        plt.tight_layout()
        
        # Save to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        
        # Encode as base64 for embedding in HTML
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
        
    except Exception as e:
        logger.error(f"Error creating supplier sustainability chart: {str(e)}")
        return None

def generate_pdf_report(
    company_name: str,
    total_emissions: Dict[str, Any],
    optimization_targets: Dict[str, Any],
    compliance_report: Optional[Dict[str, Any]] = None,
    supplier_df: Optional[pd.DataFrame] = None,
    ai_strategy: Optional[Dict[str, Any]] = None
) -> Optional[bytes]:
    """
    Generate a comprehensive PDF report
    
    Args:
        company_name: Name of the company
        total_emissions: Dictionary with emissions data
        optimization_targets: Dictionary with optimization data
        compliance_report: Dictionary with compliance data (optional)
        supplier_df: DataFrame with supplier data (optional)
        ai_strategy: Dictionary with AI strategy data (optional)
        
    Returns:
        bytes: PDF file content or None if generation failed
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            pdf_path = temp_file.name
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        heading_style = styles["Heading1"]
        subheading_style = styles["Heading2"]
        normal_style = styles["Normal"]
        
        # Create custom styles
        body_style = ParagraphStyle(
            "BodyText",
            parent=normal_style,
            spaceBefore=6,
            spaceAfter=6
        )
        
        # Build the content
        content = []
        
        # Title
        content.append(Paragraph(f"Supply Chain Sustainability Report", title_style))
        content.append(Paragraph(f"{company_name}", subheading_style))
        content.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", normal_style))
        content.append(Spacer(1, 0.25*inch))
        
        # Executive Summary
        content.append(Paragraph("Executive Summary", heading_style))
        
        total_tonnes = total_emissions.get('total_emissions_tonnes', 0)
        scope3_percent = total_emissions.get('scope_percentages', {}).get('scope3', 0)
        
        summary_text = f"""
        This report provides a comprehensive analysis of {company_name}'s supply chain sustainability performance.
        The company's total carbon footprint is {total_tonnes:.1f} tonnes CO₂e, with approximately
        {scope3_percent:.1f}% coming from Scope 3 (supply chain) emissions.
        
        Our analysis has identified potential emissions reduction opportunities of
        {optimization_targets.get('total_potential_reduction_kg', 0)/1000:.1f} tonnes CO₂e,
        representing a {optimization_targets.get('potential_reduction_percent', 0):.1f}% reduction from current levels.
        """
        
        content.append(Paragraph(summary_text, body_style))
        content.append(Spacer(1, 0.25*inch))
        
        # Emissions Analysis
        content.append(Paragraph("Emissions Analysis", heading_style))
        
        # Add emissions summary chart if available
        emissions_chart = create_emissions_summary_chart(total_emissions)
        if emissions_chart:
            img_data = base64.b64decode(emissions_chart)
            img_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
            with open(img_path, 'wb') as img_file:
                img_file.write(img_data)
            
            content.append(Image(img_path, width=6*inch, height=3.6*inch))
            content.append(Spacer(1, 0.1*inch))
        
        # Emissions breakdown table
        emissions_by_category = total_emissions.get('emissions_by_category', {})
        if emissions_by_category:
            content.append(Paragraph("Emissions by Category", subheading_style))
            
            # Convert to table data
            table_data = [['Category', 'Emissions (tonnes CO₂e)', 'Percentage']]
            
            total_emissions_kg = total_emissions.get('total_emissions_kg', 0)
            for category, emissions_kg in emissions_by_category.items():
                emissions_tonnes = emissions_kg / 1000
                percentage = (emissions_kg / total_emissions_kg * 100) if total_emissions_kg > 0 else 0
                table_data.append([category, f"{emissions_tonnes:.2f}", f"{percentage:.1f}%"])
            
            # Create the table
            emissions_table = Table(table_data, colWidths=[3*inch, 1.5*inch, 1*inch])
            emissions_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (2, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            content.append(emissions_table)
            content.append(Spacer(1, 0.25*inch))
        
        # Optimization Recommendations
        content.append(Paragraph("Optimization Recommendations", heading_style))
        
        # Add optimization impact chart if available
        optimization_chart = create_optimization_impact_chart(optimization_targets)
        if optimization_chart:
            img_data = base64.b64decode(optimization_chart)
            img_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
            with open(img_path, 'wb') as img_file:
                img_file.write(img_data)
            
            content.append(Image(img_path, width=6*inch, height=3.6*inch))
            content.append(Spacer(1, 0.1*inch))
        
        # Recommendations table
        recommendations = optimization_targets.get('recommendations', [])
        if recommendations:
            content.append(Paragraph("Key Recommendations", subheading_style))
            
            # Convert to table data
            table_data = [['Recommendation', 'Potential Reduction (tonnes CO₂e)', 'Difficulty']]
            
            for rec in recommendations:
                title = rec.get('title', '')
                reduction = rec.get('potential_reduction_kg', 0) / 1000  # Convert to tonnes
                difficulty = rec.get('difficulty', '')
                table_data.append([title, f"{reduction:.2f}", difficulty])
            
            # Create the table
            rec_table = Table(table_data, colWidths=[3*inch, 2*inch, 0.75*inch])
            rec_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            content.append(rec_table)
            content.append(Spacer(1, 0.25*inch))
        
        # Supplier Sustainability (if data available)
        if supplier_df is not None and len(supplier_df) > 0:
            content.append(Paragraph("Supplier Sustainability", heading_style))
            
            # Add supplier sustainability chart if available
            supplier_chart = create_supplier_sustainability_chart(supplier_df)
            if supplier_chart:
                img_data = base64.b64decode(supplier_chart)
                img_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                with open(img_path, 'wb') as img_file:
                    img_file.write(img_data)
                
                content.append(Image(img_path, width=6*inch, height=3.6*inch))
                content.append(Spacer(1, 0.1*inch))
            
            # Supplier summary
            if 'sustainability_score' in supplier_df.columns:
                avg_score = supplier_df['sustainability_score'].mean()
                high_risk = len(supplier_df[supplier_df['sustainability_score'] < 5])
                
                supplier_text = f"""
                The average sustainability score across all suppliers is {avg_score:.1f} out of 10.
                {high_risk} suppliers have been identified as high sustainability risk (score below 5).
                """
                
                content.append(Paragraph(supplier_text, body_style))
                content.append(Spacer(1, 0.25*inch))
        
        # Compliance Summary (if data available)
        if compliance_report:
            content.append(Paragraph("Regulatory Compliance", heading_style))
            
            applicable_regs = compliance_report.get('applicable_regulations', {})
            compliance_issues = compliance_report.get('compliance_issues', 0)
            
            compliance_text = f"""
            {len(applicable_regs)} regulations have been identified as applicable to {company_name}'s operations.
            The assessment found {compliance_issues} potential compliance issues requiring attention.
            """
            
            content.append(Paragraph(compliance_text, body_style))
            
            # Add upcoming regulations
            upcoming_regs = compliance_report.get('upcoming_regulations', [])
            if upcoming_regs:
                content.append(Paragraph("Upcoming Regulations", subheading_style))
                
                upcoming_text = "The following regulations will come into effect in the near future:"
                content.append(Paragraph(upcoming_text, body_style))
                
                for reg in upcoming_regs:
                    reg_text = f"""
                    <b>{reg.get('name', '')}</b> ({reg.get('compliance_date', '')}): {reg.get('description', '')}
                    """
                    content.append(Paragraph(reg_text, body_style))
            
            content.append(Spacer(1, 0.25*inch))
        
        # AI Strategy Recommendations (if available)
        if ai_strategy and 'strategy_text' in ai_strategy:
            content.append(Paragraph("Strategic Recommendations", heading_style))
            
            strategy_text = ai_strategy['strategy_text']
            
            # Split by headers and convert to paragraphs
            for line in strategy_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('# '):
                    # Main header
                    content.append(Paragraph(line[2:], heading_style))
                elif line.startswith('## '):
                    # Subheader
                    content.append(Paragraph(line[3:], subheading_style))
                else:
                    # Regular text
                    content.append(Paragraph(line, body_style))
            
            content.append(Spacer(1, 0.25*inch))
        
        # Next Steps
        content.append(Paragraph("Next Steps", heading_style))
        
        next_steps_text = f"""
        Based on this analysis, we recommend the following next steps:
        
        1. Implement the top optimization recommendations to achieve quick emissions reductions.
        
        2. Develop a supplier engagement program to improve supply chain sustainability.
        
        3. Address any compliance gaps identified in the assessment.
        
        4. Establish ongoing monitoring and reporting of sustainability metrics.
        
        5. Consider setting science-based targets for emissions reduction aligned with global climate goals.
        """
        
        content.append(Paragraph(next_steps_text, body_style))
        
        # Build the PDF
        doc.build(content)
        
        # Read the generated PDF
        with open(pdf_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
        
        # Clean up temporary files
        try:
            os.unlink(pdf_path)
        except:
            pass
        
        return pdf_content
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return None

def generate_html_report(
    company_name: str,
    total_emissions: Dict[str, Any],
    optimization_targets: Dict[str, Any],
    compliance_report: Optional[Dict[str, Any]] = None,
    supplier_df: Optional[pd.DataFrame] = None,
    ai_strategy: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Generate a comprehensive HTML report
    
    Args:
        company_name: Name of the company
        total_emissions: Dictionary with emissions data
        optimization_targets: Dictionary with optimization data
        compliance_report: Dictionary with compliance data (optional)
        supplier_df: DataFrame with supplier data (optional)
        ai_strategy: Dictionary with AI strategy data (optional)
        
    Returns:
        str: HTML report content or None if generation failed
    """
    try:
        # Generate charts
        emissions_chart = create_emissions_summary_chart(total_emissions)
        optimization_chart = create_optimization_impact_chart(optimization_targets)
        supplier_chart = create_supplier_sustainability_chart(supplier_df) if supplier_df is not None else None
        
        # Extract key metrics
        total_tonnes = total_emissions.get('total_emissions_tonnes', 0)
        scope3_percent = total_emissions.get('scope_percentages', {}).get('scope3', 0)
        reduction_potential = optimization_targets.get('total_potential_reduction_kg', 0)/1000
        reduction_percent = optimization_targets.get('potential_reduction_percent', 0)
        
        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Supply Chain Sustainability Report - {company_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2D7FB8;
                }}
                .header {{
                    text-align: center;
                    padding: 20px 0;
                    border-bottom: 1px solid #ddd;
                    margin-bottom: 30px;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .chart-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .metrics-container {{
                    display: flex;
                    justify-content: space-between;
                    flex-wrap: wrap;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px;
                    flex: 1;
                    min-width: 200px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2D7FB8;
                }}
                .recommendation {{
                    background-color: #f9f9f9;
                    border-left: 4px solid #4CAF50;
                    padding: 15px;
                    margin: 15px 0;
                }}
                .recommendation h3 {{
                    margin-top: 0;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px 0;
                    margin-top: 50px;
                    border-top: 1px solid #ddd;
                    font-size: 0.8em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Supply Chain Sustainability Report</h1>
                <h2>{company_name}</h2>
                <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>
                    This report provides a comprehensive analysis of {company_name}'s supply chain sustainability performance.
                    The company's total carbon footprint is {total_tonnes:.1f} tonnes CO₂e, with approximately
                    {scope3_percent:.1f}% coming from Scope 3 (supply chain) emissions.
                </p>
                <p>
                    Our analysis has identified potential emissions reduction opportunities of
                    {reduction_potential:.1f} tonnes CO₂e,
                    representing a {reduction_percent:.1f}% reduction from current levels.
                </p>
                
                <div class="metrics-container">
                    <div class="metric-card">
                        <h3>Total Emissions</h3>
                        <div class="metric-value">{total_tonnes:.1f} tonnes CO₂e</div>
                    </div>
                    <div class="metric-card">
                        <h3>Scope 3 Percentage</h3>
                        <div class="metric-value">{scope3_percent:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>Reduction Potential</h3>
                        <div class="metric-value">{reduction_potential:.1f} tonnes CO₂e</div>
                    </div>
                    <div class="metric-card">
                        <h3>Potential Reduction</h3>
                        <div class="metric-value">{reduction_percent:.1f}%</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Emissions Analysis</h2>
        """
        
        # Add emissions chart if available
        if emissions_chart:
            html += f"""
                <div class="chart-container">
                    <img src="data:image/png;base64,{emissions_chart}" alt="Emissions by Category" style="max-width:100%;">
                </div>
            """
        
        # Add emissions breakdown table
        emissions_by_category = total_emissions.get('emissions_by_category', {})
        if emissions_by_category:
            html += """
                <h3>Emissions by Category</h3>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Emissions (tonnes CO₂e)</th>
                        <th>Percentage</th>
                    </tr>
            """
            
            total_emissions_kg = total_emissions.get('total_emissions_kg', 0)
            for category, emissions_kg in emissions_by_category.items():
                emissions_tonnes = emissions_kg / 1000
                percentage = (emissions_kg / total_emissions_kg * 100) if total_emissions_kg > 0 else 0
                html += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{emissions_tonnes:.2f}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Optimization Recommendations</h2>
        """
        
        # Add optimization chart if available
        if optimization_chart:
            html += f"""
                <div class="chart-container">
                    <img src="data:image/png;base64,{optimization_chart}" alt="Potential Emissions Reduction" style="max-width:100%;">
                </div>
            """
        
        # Add recommendations
        recommendations = optimization_targets.get('recommendations', [])
        if recommendations:
            html += "<h3>Key Recommendations</h3>"
            
            for rec in recommendations:
                title = rec.get('title', '')
                description = rec.get('description', '')
                reduction = rec.get('potential_reduction_kg', 0) / 1000  # Convert to tonnes
                difficulty = rec.get('difficulty', '')
                timeframe = rec.get('timeframe', '')
                
                html += f"""
                <div class="recommendation">
                    <h3>{title}</h3>
                    <p>{description}</p>
                    <p><strong>Potential Reduction:</strong> {reduction:.2f} tonnes CO₂e</p>
                    <p><strong>Difficulty:</strong> {difficulty} | <strong>Timeframe:</strong> {timeframe}</p>
                </div>
                """
        
        html += """
            </div>
        """
        
        # Add supplier sustainability section if data available
        if supplier_df is not None and len(supplier_df) > 0:
            html += """
            <div class="section">
                <h2>Supplier Sustainability</h2>
            """
            
            if supplier_chart:
                html += f"""
                <div class="chart-container">
                    <img src="data:image/png;base64,{supplier_chart}" alt="Supplier Sustainability Scores" style="max-width:100%;">
                </div>
                """
            
            if 'sustainability_score' in supplier_df.columns:
                avg_score = supplier_df['sustainability_score'].mean()
                high_risk = len(supplier_df[supplier_df['sustainability_score'] < 5])
                
                html += f"""
                <p>
                    The average sustainability score across all suppliers is {avg_score:.1f} out of 10.
                    {high_risk} suppliers have been identified as high sustainability risk (score below 5).
                </p>
                """
            
            html += """
            </div>
            """
        
        # Add compliance section if data available
        if compliance_report:
            html += """
            <div class="section">
                <h2>Regulatory Compliance</h2>
            """
            
            applicable_regs = compliance_report.get('applicable_regulations', {})
            compliance_issues = compliance_report.get('compliance_issues', 0)
            
            html += f"""
            <p>
                {len(applicable_regs)} regulations have been identified as applicable to {company_name}'s operations.
                The assessment found {compliance_issues} potential compliance issues requiring attention.
            </p>
            """
            
            # Add applicable regulations table
            if applicable_regs:
                html += """
                <h3>Applicable Regulations</h3>
                <table>
                    <tr>
                        <th>Regulation</th>
                        <th>Description</th>
                        <th>Risk Level</th>
                    </tr>
                """
                
                for reg_name, reg_info in applicable_regs.items():
                    description = reg_info.get('description', '')
                    risk_level = reg_info.get('risk_level', '')
                    
                    html += f"""
                    <tr>
                        <td>{reg_name}</td>
                        <td>{description}</td>
                        <td>{risk_level}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            
            # Add upcoming regulations
            upcoming_regs = compliance_report.get('upcoming_regulations', [])
            if upcoming_regs:
                html += """
                <h3>Upcoming Regulations</h3>
                <p>The following regulations will come into effect in the near future:</p>
                <table>
                    <tr>
                        <th>Regulation</th>
                        <th>Description</th>
                        <th>Compliance Date</th>
                        <th>Days Remaining</th>
                    </tr>
                """
                
                for reg in upcoming_regs:
                    name = reg.get('name', '')
                    description = reg.get('description', '')
                    compliance_date = reg.get('compliance_date', '')
                    days_remaining = reg.get('days_until_compliance', '')
                    
                    html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{description}</td>
                        <td>{compliance_date}</td>
                        <td>{days_remaining}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            
            html += """
            </div>
            """
        
        # Add AI strategy if available
        if ai_strategy and 'strategy_text' in ai_strategy:
            html += """
            <div class="section">
                <h2>Strategic Recommendations</h2>
            """
            
            strategy_text = ai_strategy['strategy_text']
            
            # Convert markdown-like format to HTML
            strategy_html = ""
            for line in strategy_text.split('\n'):
                line = line.strip()
                if not line:
                    strategy_html += "<p>&nbsp;</p>"
                elif line.startswith('# '):
                    strategy_html += f"<h2>{line[2:]}</h2>"
                elif line.startswith('## '):
                    strategy_html += f"<h3>{line[3:]}</h3>"
                elif line.startswith('- '):
                    strategy_html += f"<li>{line[2:]}</li>"
                elif line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. '):
                    # If we detect a numbered list but aren't in one yet, start a new one
                    if not strategy_html.endswith('</ol>') and not strategy_html.endswith('<ol>'):
                        strategy_html += "<ol>"
                    strategy_html += f"<li>{line[3:]}</li>"
                else:
                    strategy_html += f"<p>{line}</p>"
            
            html += strategy_html
            
            html += """
            </div>
            """
        
        # Add next steps section
        html += """
        <div class="section">
            <h2>Next Steps</h2>
            <p>Based on this analysis, we recommend the following next steps:</p>
            <ol>
                <li>Implement the top optimization recommendations to achieve quick emissions reductions.</li>
                <li>Develop a supplier engagement program to improve supply chain sustainability.</li>
                <li>Address any compliance gaps identified in the assessment.</li>
                <li>Establish ongoing monitoring and reporting of sustainability metrics.</li>
                <li>Consider setting science-based targets for emissions reduction aligned with global climate goals.</li>
            </ol>
        </div>
        
        <div class="footer">
            <p>Generated by AI-Powered Sustainable Supply Chain Optimizer</p>
            <p>© 2025 All Rights Reserved</p>
        </div>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        return None
