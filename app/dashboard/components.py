"""
Custom Streamlit components for the Supply Chain Optimizer dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import io
import re
import os

from app.utils.emissions_calc import generate_optimization_targets
from app.config import get_logger

logger = get_logger(__name__)

def create_sidebar():
    """Create the sidebar navigation"""

    logo_path = "app/assets/images/logo.png" # Define path

    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=150)
    else:
        st.sidebar.warning(f"Logo image not found at {logo_path}. Please add the logo file.")
    # --- End Image Placeholder ---

    st.sidebar.title("AI Supply Chain Optimizer")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Data Upload", "Emissions Analysis", "Optimization", "Supplier Assessment", "Compliance", "Reports", "AI Assistant"]
    )
    
    # Add filters if data is available
    if 'procurement_data' in st.session_state and st.session_state['procurement_data'] is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filters")
        
        if 'product_category' in st.session_state['procurement_data'].columns:
            categories = ['All'] + sorted(st.session_state['procurement_data']['product_category'].unique().tolist())
            selected_category = st.sidebar.selectbox("Product Category", categories)
            
            if selected_category != 'All':
                st.session_state['category_filter'] = selected_category
            else:
                st.session_state['category_filter'] = None
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **About**
        
        This application helps SMEs analyze and optimize their supply chain sustainability using AI.
        
        Data is processed locally and API calls are made securely.
        """
    )
    
    return page

def render_kpi_metrics(metrics: Dict[str, Any]):
    """
    Render KPI metrics in a row of cards with enhanced visual design
    """
    cols = st.columns(len(metrics))
    
    for i, (title, value) in enumerate(metrics.items()):
        with cols[i]:
            # Add custom HTML/CSS for better-styled metric cards
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                text-align: center;
                height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <h3 style="color: #6c757d; font-size: 0.9rem; margin-bottom: 10px;">{title}</h3>
                <p style="color: #2c3e50; font-size: 1.8rem; font-weight: bold; margin: 0;">{value}</p>
            </div>
            """, unsafe_allow_html=True)


def render_emissions_summary(total_emissions: Dict[str, Any]):
    """
    Render emissions summary charts and metrics
    
    Args:
        total_emissions: Dictionary with emissions data
    """
    st.subheader("Emissions Overview")
    
    # Display key metrics
    metrics = {
        "Total Emissions": f"{total_emissions.get('total_emissions_tonnes', 0):.1f} tonnes CO₂e",
        "Scope 3 %": f"{total_emissions.get('scope_percentages', {}).get('scope3', 0):.1f}%",
        "Categories": len(total_emissions.get('emissions_by_category', {}))
    }
    
    render_kpi_metrics(metrics)
    
    # Create emissions by category chart
    emissions_by_category = total_emissions.get('emissions_by_category', {})
    
    if emissions_by_category:
        # Convert to dataframe for plotting
        df = pd.DataFrame({
            'Category': list(emissions_by_category.keys()),
            'Emissions (kg CO₂e)': list(emissions_by_category.values())
        })
        
        # Sort by emissions (descending)
        df = df.sort_values('Emissions (kg CO₂e)', ascending=False)
        
        # Convert to tonnes for better readability
        df['Emissions (tonnes CO₂e)'] = df['Emissions (kg CO₂e)'] / 1000
        
        # Create plotly bar chart
        fig = px.bar(
            df,
            x='Category',
            y='Emissions (tonnes CO₂e)',
            title='Carbon Emissions by Category',
            color='Emissions (tonnes CO₂e)',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Emissions (tonnes CO₂e)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)', # Transparent paper
            font=dict(family="Arial, sans-serif", size=12, color="#e0e0e0") # Light font color for dark mode
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display emissions by scope as pie chart
        emissions_by_scope = total_emissions.get('emissions_by_scope', {})
        
        if emissions_by_scope:
            # Convert to dataframe for plotting
            scope_df = pd.DataFrame({
                'Scope': [f"Scope {scope.replace('scope', '')}" for scope in emissions_by_scope.keys()],
                'Emissions (kg CO₂e)': list(emissions_by_scope.values())
            })
            
            # Convert to tonnes
            scope_df['Emissions (tonnes CO₂e)'] = scope_df['Emissions (kg CO₂e)'] / 1000
            
            # Create plotly pie chart
            fig = px.pie(
                scope_df,
                values='Emissions (tonnes CO₂e)',
                names='Scope',
                title='Emissions by Scope',
                color='Scope',
                color_discrete_map={
                    'Scope 1': '#2E8B57',
                    'Scope 2': '#4682B4',
                    'Scope 3': '#B8860B'
                }
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)', # Transparent paper
                font=dict(color='#e0e0e0') # Light font color for dark mode
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No emissions data available by category.")

def render_optimization_summary(optimization_targets: Dict[str, Any]):
    """
    Render optimization recommendations and impact
    
    Args:
        optimization_targets: Dictionary with optimization data
    """
    st.subheader("Optimization Opportunities")
    
    # Display key metrics
    reduction_kg = optimization_targets.get('total_potential_reduction_kg', 0)
    reduction_tonnes = reduction_kg / 1000
    reduction_percent = optimization_targets.get('potential_reduction_percent', 0)
    
    metrics = {
        "Potential Reduction": f"{reduction_tonnes:.1f} tonnes CO₂e",
        "Reduction %": f"{reduction_percent:.1f}%",
        "Recommendations": len(optimization_targets.get('recommendations', []))
    }
    
    render_kpi_metrics(metrics)
    
    # Display recommendations
    recommendations = optimization_targets.get('recommendations', [])
    
    if recommendations:
        # Convert to dataframe
        df = pd.DataFrame(recommendations)
        
        # Select relevant columns and rename
        if 'potential_reduction_kg' in df.columns:
            df['potential_reduction_tonnes'] = df['potential_reduction_kg'] / 1000
            df = df[['category', 'title', 'description', 'potential_reduction_tonnes', 'difficulty', 'timeframe']]
            df.columns = ['Category', 'Recommendation', 'Description', 'Reduction Potential (tonnes)', 'Difficulty', 'Timeframe']
        
        # Sort by reduction potential
        df = df.sort_values('Reduction Potential (tonnes)', ascending=False)
        
        # Create bar chart of reduction potential by recommendation
        fig = px.bar(
            df,
            x='Recommendation',
            y='Reduction Potential (tonnes)',
            title='Potential Emissions Reduction by Recommendation',
            color='Difficulty',
            color_discrete_map={
                'Low': '#4CAF50',
                'Medium': '#FFC107',
                'High': '#F44336'
            }
        )
        
        fig.update_layout(
            xaxis_title="Recommendation",
            yaxis_title="Reduction Potential (tonnes CO₂e)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display recommendations as expandable sections
        for i, rec in enumerate(recommendations):
            with st.expander(f"{i+1}. {rec.get('title', 'Recommendation')} - {rec.get('potential_reduction_kg')/1000:.1f} tonnes CO₂e"):
                st.write(f"**Category:** {rec.get('category', 'N/A')}")
                st.write(f"**Description:** {rec.get('description', 'N/A')}")
                st.write(f"**Difficulty:** {rec.get('difficulty', 'N/A')}")
                st.write(f"**Timeframe:** {rec.get('timeframe', 'N/A')}")
    else:
        st.info("No optimization recommendations available.")

def render_supplier_assessment(supplier_df: pd.DataFrame, risk_assessment: Optional[Dict[str, Any]] = None):
    """
    Render supplier sustainability assessment
    
    Args:
        supplier_df: DataFrame with supplier data
        risk_assessment: Dictionary with risk assessment results (optional)
    """
    st.subheader("Supplier Sustainability Assessment")
    
    if supplier_df is None or len(supplier_df) == 0:
        st.info("No supplier data available.")
        return
    
    # Display key metrics
    metrics = {
        "Suppliers": len(supplier_df),
        "Avg Sustainability Score": f"{supplier_df['sustainability_score'].mean():.1f}/10" if 'sustainability_score' in supplier_df.columns else "N/A",
        "High Risk Suppliers": len(supplier_df[supplier_df['sustainability_risk'] >= 7]) if 'sustainability_risk' in supplier_df.columns else "N/A"
    }
    
    render_kpi_metrics(metrics)
    
    # Create supplier sustainability chart
    if 'sustainability_score' in supplier_df.columns:
        # Create scatter plot of suppliers
        fig = px.scatter(
            supplier_df,
            x='sustainability_score',
            y='sustainability_risk' if 'sustainability_risk' in supplier_df.columns else 'risk_score',
            color='category' if 'category' in supplier_df.columns else None,
            hover_name='supplier_name' if 'supplier_name' in supplier_df.columns else None,
            size='annual_revenue_usd' if 'annual_revenue_usd' in supplier_df.columns else None,
            title='Supplier Sustainability Assessment',
            labels={
                'sustainability_score': 'Sustainability Score (higher is better)',
                'sustainability_risk': 'Risk Score (lower is better)',
                'risk_score': 'Risk Score (lower is better)'
            }
        )
        
        # Add a vertical line at score 5 for reference
        fig.add_vline(x=5, line_dash="dash", line_color="gray")
        
        # Add a horizontal line at risk 5 for reference
        if 'sustainability_risk' in supplier_df.columns or 'risk_score' in supplier_df.columns:
            fig.add_hline(y=5, line_dash="dash", line_color="gray")
        
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display supplier distribution by risk category
        if 'risk_category' in supplier_df.columns:
            risk_counts = supplier_df['risk_category'].value_counts().reset_index()
            risk_counts.columns = ['Risk Category', 'Count']
            
            fig = px.pie(
                risk_counts,
                values='Count',
                names='Risk Category',
                title='Suppliers by Risk Category',
                color='Risk Category',
                color_discrete_map={
                    'Low': '#4CAF50',
                    'Medium': '#FFC107',
                    'High': '#F44336'
                }
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)', # Transparent paper
                font=dict(color='#e0e0e0') # Light font color for dark mode
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Display risk hotspots if available
    if risk_assessment and 'high_risk_suppliers' in risk_assessment:
        st.subheader("Risk Hotspots")
        
        # Display high risk suppliers
        high_risk_suppliers = risk_assessment.get('high_risk_suppliers', [])
        if high_risk_suppliers:
            st.write("**High Risk Suppliers:**")
            
            # Convert to dataframe for display
            high_risk_df = pd.DataFrame(high_risk_suppliers)
            
            # Reorder and rename columns for better display
            display_columns = []
            rename_map = {}
            
            if 'supplier_name' in high_risk_df.columns:
                display_columns.append('supplier_name')
                rename_map['supplier_name'] = 'Supplier Name'
            
            if 'supplier_id' in high_risk_df.columns:
                display_columns.append('supplier_id')
                rename_map['supplier_id'] = 'Supplier ID'
            
            if 'category' in high_risk_df.columns:
                display_columns.append('category')
                rename_map['category'] = 'Category'
            
            if 'country' in high_risk_df.columns:
                display_columns.append('country')
                rename_map['country'] = 'Country'
            
            if 'overall_risk_score' in high_risk_df.columns:
                display_columns.append('overall_risk_score')
                rename_map['overall_risk_score'] = 'Risk Score'
            
            display_df = high_risk_df[display_columns].rename(columns=rename_map)
            
            st.dataframe(display_df)
        
        # Display high risk categories
        high_risk_categories = risk_assessment.get('high_risk_categories', {})
        if high_risk_categories:
            st.write("**High Risk Categories:**")
            
            # Convert to dataframe
            category_df = pd.DataFrame({
                'Category': list(high_risk_categories.keys()),
                'Risk Score': list(high_risk_categories.values())
            })
            
            # Sort by risk score
            category_df = category_df.sort_values('Risk Score', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                category_df,
                x='Category',
                y='Risk Score',
                title='Risk Score by Category',
                color='Risk Score',
                color_continuous_scale='Reds'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display high risk countries
        high_risk_countries = risk_assessment.get('high_risk_countries', {})
        if high_risk_countries:
            st.write("**High Risk Countries:**")
            
            # Convert to dataframe
            country_df = pd.DataFrame({
                'Country': list(high_risk_countries.keys()),
                'Risk Score': list(high_risk_countries.values())
            })
            
            # Sort by risk score
            country_df = country_df.sort_values('Risk Score', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                country_df,
                x='Country',
                y='Risk Score',
                title='Risk Score by Country',
                color='Risk Score',
                color_continuous_scale='Reds'
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_compliance_summary(compliance_report: Dict[str, Any]):
    """
    Render compliance assessment summary
    
    Args:
        compliance_report: Dictionary with compliance assessment data
    """
    st.subheader("Regulatory Compliance Assessment")
    
    if not compliance_report:
        st.info("No compliance data available.")
        return
    
    # Display key metrics
    metrics = {
        "Applicable Regulations": len(compliance_report.get('applicable_regulations', {})),
        "Compliance Issues": compliance_report.get('compliance_issues', 0),
        "Upcoming Regulations": len(compliance_report.get('upcoming_regulations', []))
    }
    
    render_kpi_metrics(metrics)
    
    # Display applicable regulations
    applicable_regulations = compliance_report.get('applicable_regulations', {})
    if applicable_regulations:
        st.write("**Applicable Regulations:**")
        
        # Convert to dataframe
        reg_rows = []
        for reg_name, reg_info in applicable_regulations.items():
            reg_rows.append({
                'Regulation': reg_name,
                'Description': reg_info.get('description', 'N/A'),
                'Risk Level': reg_info.get('risk_level', 'N/A'),
                'Compliance Date': reg_info.get('compliance_date', 'Current')
            })
        
        reg_df = pd.DataFrame(reg_rows)
        
        # Sort by risk level
        risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
        reg_df['Risk Order'] = reg_df['Risk Level'].map(lambda x: risk_order.get(x, 3))
        reg_df = reg_df.sort_values('Risk Order').drop(columns=['Risk Order'])
        
        st.dataframe(reg_df)
    
    # Display compliance status
    emissions_compliance = compliance_report.get('emissions_compliance', {})
    chemical_compliance = compliance_report.get('chemical_compliance', {})
    
    all_compliance = {**emissions_compliance, **chemical_compliance}
    
    if all_compliance:
        st.write("**Compliance Status:**")
        
        # Convert to dataframe
        status_rows = []
        for reg_name, status_info in all_compliance.items():
            status_rows.append({
                'Regulation': reg_name,
                'Status': status_info.get('status', 'Unknown'),
                'Gaps': len(status_info.get('gaps', [])),
                'Recommendations': len(status_info.get('recommendations', []))
            })
        
        status_df = pd.DataFrame(status_rows)
        
        # Create color-coded status indicators
        status_colors = {
            'Compliant': '#4CAF50',
            'Partially compliant': '#FFC107',
            'Non-compliant': '#F44336',
            'Unknown': '#9E9E9E',
            'Likely compliant': '#8BC34A',
            'Needs assessment': '#FF9800',
            'Preparing': '#2196F3'
        }
        
        # Display status with colored badges
        cols = st.columns(len(status_rows))
        for i, row in enumerate(status_rows):
            status = row['Status']
            color = status_colors.get(status, '#9E9E9E')
            
            with cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        background-color: {color};
                        padding: 10px;
                        border-radius: 5px;
                        color: white;
                        text-align: center;
                        margin-bottom: 10px;
                    ">
                        <h4 style="margin:0;">{row['Regulation']}</h4>
                        <p style="margin:0;">{status}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # Display upcoming regulations
    upcoming_regulations = compliance_report.get('upcoming_regulations', [])
    if upcoming_regulations:
        st.write("**Upcoming Regulations:**")
        
        # Convert to dataframe
        upcoming_df = pd.DataFrame(upcoming_regulations)
        
        # Select and rename columns
        if 'name' in upcoming_df.columns and 'compliance_date' in upcoming_df.columns:
            display_columns = ['name', 'compliance_date', 'days_until_compliance', 'risk_level']
            rename_map = {
                'name': 'Regulation',
                'compliance_date': 'Compliance Date',
                'days_until_compliance': 'Days Remaining',
                'risk_level': 'Risk Level'
            }
            
            upcoming_display = upcoming_df[display_columns].rename(columns=rename_map)
            
            # Sort by days remaining
            upcoming_display = upcoming_display.sort_values('Days Remaining')
            
            st.dataframe(upcoming_display)
            
            # Create timeline visualization
            # Replace the existing timeline code with:
            fig = px.bar(
                upcoming_df,
                x='days_until_compliance',
                y='name',
                orientation='h',
                color='risk_level',
                labels={
                    'days_until_compliance': 'Days Until Compliance',
                    'name': 'Regulation',
                    'risk_level': 'Risk Level'
                },
                color_discrete_map={
                    'High': '#F44336',
                    'Medium': '#FFC107',
                    'Low': '#4CAF50'
                }
            )

            # Customize layout
            fig.update_layout(
                title='Regulatory Timeline',
                xaxis_title='Days Until Compliance',
                yaxis_title='Regulation',
                height=400
            )
            
            # Reverse y-axis to show soonest regulations at the top
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Display compliance insights
    compliance_insights = compliance_report.get('compliance_insights', None)
    if compliance_insights:
        st.subheader("Compliance Insights")
        st.markdown(compliance_insights)

def render_ai_chat_interface(nlp_agent):
    """
    Render AI chatbot interface
    
    Args:
        nlp_agent: NLP agent instance
    """
    st.subheader("AI Supply Chain Sustainability Assistant")
    
    # Initialize chat history if not already in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle new input
    if prompt := st.chat_input("Ask about your supply chain sustainability..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get data for context
        total_emissions = st.session_state.get('total_emissions', None)
        optimization_data = st.session_state.get('optimization_targets', None)
        supplier_data = None
        compliance_data = st.session_state.get('compliance_report', None)
        
        if 'supplier_df' in st.session_state and st.session_state['supplier_df'] is not None:
            supplier_data = {
                'supplier_count': len(st.session_state['supplier_df']),
                'risk_assessment': st.session_state.get('risk_assessment', None)
            }
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = nlp_agent.process_query(
                query=prompt,
                total_emissions=total_emissions,
                optimization_data=optimization_data,
                supplier_data=supplier_data,
                compliance_data=compliance_data
            )
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
            
            # Display any extracted data visualizations
            extracted_data = response.get('data', {})
            
            if 'emissions_value' in extracted_data:
                st.metric("Emissions", f"{extracted_data['emissions_value']/1000:.2f} tonnes CO₂e")
            
            if 'reduction_percentage' in extracted_data:
                st.metric("Potential Reduction", f"{extracted_data['reduction_percentage']:.1f}%")
            
            if 'recommendations' in extracted_data and extracted_data['recommendations']:
                with st.expander("Recommended Actions"):
                    for i, rec in enumerate(extracted_data['recommendations']):
                        st.write(f"{i+1}. {rec}")

def render_data_upload_interface():
    """Render the data upload interface"""
    
    st.subheader("Upload Supply Chain Data")
    
    st.write("""
    Upload your supply chain data to analyze emissions, identify optimization opportunities, 
    and assess sustainability risks. We support the following data types:
    """)
    
    # Create tabs for different data types
    upload_tab, sample_tab = st.tabs(["Upload Your Data", "Use Sample Data"])
    
    with upload_tab:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Procurement Data**")
            st.write("Contains information about purchased goods and services.")
            procurement_file = st.file_uploader("Upload Procurement CSV", type="csv", key="procurement_upload")
            
            st.write("**Supplier Data**")
            st.write("Contains information about your suppliers.")
            supplier_file = st.file_uploader("Upload Supplier CSV", type="csv", key="supplier_upload")
        
        with col2:
            st.write("**Logistics Data**")
            st.write("Contains information about transportation and shipping.")
            logistics_file = st.file_uploader("Upload Logistics CSV", type="csv", key="logistics_upload")
            
            st.write("**Company Information**")
            st.text_input("Company Name", key="company_name", value="Sample Company Inc.")
            
            company_region = st.selectbox(
                "Region",
                ["US", "EU", "UK", "China", "India", "Japan", "Brazil", "Other"],
                index=0,
                key="company_region"
            )
            
            industry_sector = st.selectbox(
                "Industry",
                ["Manufacturing", "Retail", "Food & Beverage", "Textiles", "Electronics", 
                 "Construction", "Automotive", "Healthcare", "Other"],
                index=0,
                key="industry_sector"
            )
    
    with sample_tab:
        st.write("Use sample data to explore the application's features.")
        
        if st.button("Load Sample Data"):
            st.session_state['use_sample_data'] = True
            st.success("Sample data loaded successfully!")
    
    # Process button
    if st.button("Process Data"):
        return True
    
    return False

def render_loading_spinner():
    """Render loading spinner with progress bar"""
    
    progress_container = st.empty()
    progress_bar = progress_container.progress(0)
    
    for i in range(0, 101, 10):
        progress_bar.progress(i)
        if i == 0:
            st.write("Loading data...")
        elif i == 20:
            st.write("Calculating emissions...")
        elif i == 40:
            st.write("Analyzing supply chain...")
        elif i == 60:
            st.write("Generating optimization strategies...")
        elif i == 80:
            st.write("Checking regulatory compliance...")
        
        import time
        time.sleep(0.1)
    
    # Clear the progress bar
    progress_container.empty()

def download_button(object_to_download, download_filename, button_text):
    """
    Generate a download button for any object that can be serialized to bytes
    
    Args:
        object_to_download: The object to be downloaded
        download_filename: Filename for the download
        button_text: Text to display on the button
        
    Returns:
        A Streamlit component for downloading the object
    """
    try:
        # If object is PDF content (bytes)
        if isinstance(object_to_download, bytes):
            b64 = base64.b64encode(object_to_download).decode()
            button_uuid = str(hash(button_text))
            button_id = re.sub(r"\d+", "", button_uuid)
            
            custom_css = f""" 
                <style>
                    #{button_id} {{
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        background-color: rgb(255, 255, 255);
                        color: rgb(38, 39, 48);
                        padding: .25rem .75rem;
                        position: relative;
                        text-decoration: none;
                        border-radius: 4px;
                        border-width: 1px;
                        border-style: solid;
                        border-color: rgb(230, 234, 241);
                        border-image: initial;
                    }} 
                    #{button_id}:hover {{
                        border-color: rgb(246, 51, 102);
                        color: rgb(246, 51, 102);
                    }}
                    #{button_id}:active {{
                        box-shadow: none;
                        background-color: rgb(246, 51, 102);
                        color: white;
                        }}
                </style> """
            
            dl_link = f"""
                {custom_css}
                <a download="{download_filename}" id="{button_id}" href="data:application/octet-stream;base64,{b64}">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-download me-2" viewBox="0 0 16 16">
                        <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                        <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
                    </svg>
                    {button_text}
                </a>
            """
            
            return st.markdown(dl_link, unsafe_allow_html=True)
        
        # If object is string (like HTML)
        elif isinstance(object_to_download, str):
            if download_filename.endswith('.html'):
                b64 = base64.b64encode(object_to_download.encode()).decode()
                button_uuid = str(hash(button_text))
                button_id = re.sub(r"\d+", "", button_uuid)
                
                custom_css = f""" 
                    <style>
                        #{button_id} {{
                            display: inline-flex;
                            align-items: center;
                            justify-content: center;
                            background-color: rgb(255, 255, 255);
                            color: rgb(38, 39, 48);
                            padding: .25rem .75rem;
                            position: relative;
                            text-decoration: none;
                            border-radius: 4px;
                            border-width: 1px;
                            border-style: solid;
                            border-color: rgb(230, 234, 241);
                            border-image: initial;
                        }} 
                        #{button_id}:hover {{
                            border-color: rgb(246, 51, 102);
                            color: rgb(246, 51, 102);
                        }}
                        #{button_id}:active {{
                            box-shadow: none;
                            background-color: rgb(246, 51, 102);
                            color: white;
                            }}
                    </style> """
                
                dl_link = f"""
                    {custom_css}
                    <a download="{download_filename}" id="{button_id}" href="data:text/html;base64,{b64}">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-download me-2" viewBox="0 0 16 16">
                            <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                            <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
                        </svg>
                        {button_text}
                    </a>
                """
                
                return st.markdown(dl_link, unsafe_allow_html=True)
            else:
                st.error("Unsupported string format for download")
                return None
        
        # Other object types not supported
        else:
            st.error("Unsupported object type for download")
            return None
        
    except Exception as e:
        st.error(f"Error creating download button: {str(e)}")
        return None
