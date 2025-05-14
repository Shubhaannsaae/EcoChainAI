"""
Main dashboard for the Supply Chain Optimizer.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

from app.utils.data_loader import (
    load_csv_data, detect_data_type, load_sample_data,
    generate_sample_procurement_data, generate_sample_logistics_data, generate_sample_supplier_data
)
from app.utils.emissions_calc import (
    calculate_logistics_emissions, calculate_procurement_emissions, 
    calculate_supplier_emissions, calculate_total_emissions,
    generate_optimization_targets
)
from app.utils.optimization import (
    generate_optimal_routes, generate_sustainable_procurement_plan,
    generate_supplier_sustainability_plan, generate_ai_sustainability_strategy
)
from app.utils.compliance import generate_compliance_report
from app.utils.reporting import generate_pdf_report, generate_html_report
from app.models.risk_assessment import calculate_supplier_risk_scores, identify_risk_hotspots, generate_risk_mitigation_strategies
from app.models.nlp_agent import NLPAgent
from app.dashboard.components import (
    create_sidebar, render_emissions_summary, render_optimization_summary,
    render_supplier_assessment, render_compliance_summary, render_kpi_metrics,
    render_ai_chat_interface, render_data_upload_interface, render_loading_spinner,
    download_button
)
from app.config import get_logger

logger = get_logger(__name__)

def setup_dashboard():
    """
    Set up and render the main dashboard
    """
    # Initialize session state if needed
    if 'initialized' not in st.session_state:
        initialize_session_state()
    
    # Create sidebar and get selected page
    page = create_sidebar()
    
    # Display the selected page
    if page == "Dashboard":
        show_dashboard_page()
    elif page == "Data Upload":
        show_data_upload_page()
    elif page == "Emissions Analysis":
        show_emissions_analysis_page()
    elif page == "Optimization":
        show_optimization_page()
    elif page == "Supplier Assessment":
        show_supplier_assessment_page()
    elif page == "Compliance":
        show_compliance_page()
    elif page == "Reports":
        show_reports_page()
    elif page == "AI Assistant":
        show_ai_assistant_page()

def initialize_session_state():
    """Initialize session state variables"""
    
    st.session_state.initialized = True
    st.session_state.procurement_data = None
    st.session_state.logistics_data = None
    st.session_state.supplier_df = None
    st.session_state.category_filter = None
    st.session_state.use_sample_data = False
    
    # Analysis results
    st.session_state.procurement_emissions = None
    st.session_state.logistics_emissions = None
    st.session_state.supplier_emissions = None
    st.session_state.total_emissions = None
    st.session_state.optimization_targets = None
    st.session_state.risk_assessment = None
    st.session_state.compliance_report = None
    st.session_state.ai_strategy = None
    
    # Chat history for AI assistant
    st.session_state.chat_history = []
    
    # Company info
    st.session_state.company_name = "Sample Company Inc."
    st.session_state.company_region = "US"
    st.session_state.industry_sector = "Manufacturing"


def show_dashboard_page():
    """Display an enhanced main dashboard overview"""
    
    # Apply custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        color: #1E6797;
        margin-bottom: 1.5rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #2E7D32;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .info-text {
        color: #424242;
        font-size: 1rem;
    }
    .highlight {
        color: #2E7D32;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with animation
    st.markdown('<h1 class="main-header">Supply Chain Sustainability Dashboard</h1>', unsafe_allow_html=True)
    
    # Check if data is available
    data_available = (
        st.session_state.procurement_data is not None or
        st.session_state.logistics_data is not None or
        st.session_state.supplier_df is not None
    )
    
    if not data_available:
        # Show enhanced welcome screen
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image("https://cdn-icons-png.flaticon.com/512/6519/6519723.png", width=250)
        
        with col2:
            st.markdown("""
            <h2 style="color: #1E6797;">Welcome to EcoChainAI!</h2>
            <p class="info-text">Unlock sustainability insights for your business with our AI-powered platform.</p>
            <ul style="color: #424242;">
                <li><span class="highlight">Analyze</span> your carbon footprint</li>
                <li><span class="highlight">Optimize</span> your supply chain</li>
                <li><span class="highlight">Comply</span> with regulations</li>
                <li><span class="highlight">Improve</span> your sustainability</li>
            </ul>
            """, unsafe_allow_html=True)
            
            st.button("🚀 Load Sample Data", key="load_sample", on_click=load_sample_dataset)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add "Getting Started" section
        st.markdown('<h3 class="subheader">Getting Started</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card" style="text-align: center;">
                <h4 style="color: #1976D2;">Step 1: Upload Data</h4>
                <p>Upload your supply chain data or use our sample dataset</p>
                <p style="font-size: 2rem;">📊</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card" style="text-align: center;">
                <h4 style="color: #1976D2;">Step 2: Analyze</h4>
                <p>View emissions and identify optimization opportunities</p>
                <p style="font-size: 2rem;">🔍</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="card" style="text-align: center;">
                <h4 style="color: #1976D2;">Step 3: Optimize</h4>
                <p>Implement AI-generated recommendations</p>
                <p style="font-size: 2rem;">✅</p>
            </div>
            """, unsafe_allow_html=True)
        
        return

    # Display enhanced overview metrics
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data_types = []
        if st.session_state.procurement_data is not None:
            data_types.append("Procurement")
        if st.session_state.logistics_data is not None:
            data_types.append("Logistics")
        if st.session_state.supplier_df is not None:
            data_types.append("Supplier")
        
        st.metric("Data Sources", len(data_types))
    
    with col2:
        if st.session_state.total_emissions is not None:
            total_tonnes = st.session_state.total_emissions.get('total_emissions_tonnes', 0)
            st.metric("Total Emissions", f"{total_tonnes:.1f} tonnes CO₂e")
        else:
            st.metric("Total Emissions", "Not calculated")
    
    with col3:
        if st.session_state.optimization_targets is not None:
            reduction = st.session_state.optimization_targets.get('potential_reduction_percent', 0)
            st.metric("Reduction Potential", f"{reduction:.1f}%", delta=f"-{reduction:.1f}%")
        else:
            st.metric("Reduction Potential", "Not calculated")
    
    with col4:
        if st.session_state.supplier_df is not None:
            if 'sustainability_score' in st.session_state.supplier_df.columns:
                avg_score = st.session_state.supplier_df['sustainability_score'].mean()
                st.metric("Supplier Sustainability", f"{avg_score:.1f}/10")
            else:
                st.metric("Suppliers", len(st.session_state.supplier_df))
        else:
            st.metric("Suppliers", "No data")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display main sections with enhanced styling
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Emissions summary
        if st.session_state.total_emissions is not None:
            st.markdown('<h3 class="subheader">Emissions Overview</h3>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_emissions_summary(st.session_state.total_emissions)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Optimization summary
        if st.session_state.optimization_targets is not None:
            st.markdown('<h3 class="subheader">Optimization Opportunities</h3>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_optimization_summary(st.session_state.optimization_targets)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # AI Strategy Insights with enhanced styling
    st.markdown('<h3 class="subheader">AI-Generated Sustainability Strategy</h3>', unsafe_allow_html=True)
    
    if st.session_state.ai_strategy is not None:
        with st.expander("View Strategy", expanded=True):
            st.markdown('<div class="card" style="background-color: #f8fdf9;">', unsafe_allow_html=True)
            st.markdown(st.session_state.ai_strategy.get('strategy_text', ''))
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Show button to generate strategy
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            <p class="info-text">Generate an AI-powered sustainability strategy based on your data to identify key opportunities and create an implementation roadmap.</p>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🤖 Generate Strategy", key="gen_strategy"):
                with st.spinner("Generating sustainability strategy..."):
                    strategy = generate_ai_sustainability_strategy(
                        logistics_data=st.session_state.get('logistics_emissions'),
                        procurement_data=st.session_state.get('procurement_emissions'),
                        supplier_data=st.session_state.get('supplier_emissions'),
                        total_emissions=st.session_state.get('total_emissions'),
                        optimization_targets=st.session_state.get('optimization_targets')
                    )
                    
                    st.session_state.ai_strategy = strategy
                    st.rerun()
                    
        st.markdown('</div>', unsafe_allow_html=True)

def show_data_upload_page():
    """Display the data upload interface"""
    
    st.title("Data Upload")
    
    # Show file upload interface
    if render_data_upload_interface():
        with st.spinner("Processing data..."):
            process_uploaded_data()
        
        st.success("Data processed successfully!")
        st.markdown("Navigate to the **Dashboard** to see your results.")
    
    # Show current data summary if available
    if st.session_state.procurement_data is not None or st.session_state.logistics_data is not None or st.session_state.supplier_df is not None:
        st.markdown("---")
        st.subheader("Current Data Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.procurement_data is not None:
                st.write(f"**Procurement Data:** {len(st.session_state.procurement_data)} records")
            else:
                st.write("**Procurement Data:** None")
        
        with col2:
            if st.session_state.logistics_data is not None:
                st.write(f"**Logistics Data:** {len(st.session_state.logistics_data)} records")
            else:
                st.write("**Logistics Data:** None")
        
        with col3:
            if st.session_state.supplier_df is not None:
                st.write(f"**Supplier Data:** {len(st.session_state.supplier_df)} records")
            else:
                st.write("**Supplier Data:** None")
        
        if st.button("Clear All Data"):
            for key in ['procurement_data', 'logistics_data', 'supplier_df',
                       'procurement_emissions', 'logistics_emissions', 'supplier_emissions',
                       'total_emissions', 'optimization_targets', 'risk_assessment',
                       'compliance_report', 'ai_strategy']:
                if key in st.session_state:
                    st.session_state[key] = None
            
            st.session_state.chat_history = []
            st.success("All data cleared!")
            st.rerun()

def show_emissions_analysis_page():
    """Display the emissions analysis page"""
    
    st.title("Emissions Analysis")
    
    # Check if data is available
    if st.session_state.total_emissions is None:
        if st.session_state.procurement_data is not None or st.session_state.logistics_data is not None:
            if st.button("Calculate Emissions"):
                with st.spinner("Calculating emissions..."):
                    run_emissions_calculations()
                st.success("Emissions calculated!")
                st.rerun()
        else:
            st.info("Please upload supply chain data to analyze emissions.")
            return
    
    # Display emissions summary
    render_emissions_summary(st.session_state.total_emissions)
    
    # Display detailed emissions breakdown
    st.markdown("---")
    st.subheader("Detailed Emissions Analysis")
    
    emissions_tab1, emissions_tab2, emissions_tab3 = st.tabs(["Logistics", "Procurement", "Data Tables"])
    
    with emissions_tab1:
        if st.session_state.logistics_emissions is not None:
            logistics_summary = st.session_state.logistics_emissions[1]
            
            # Key metrics
            metrics = {
                "Total Logistics Emissions": f"{logistics_summary.get('total_emissions_kg', 0)/1000:.1f} tonnes CO₂e",
                "Total Tonne-KM": f"{logistics_summary.get('total_tonne_km', 0):.1f}",
                "Avg Emission Factor": f"{logistics_summary.get('avg_emission_factor', 0)*1000:.1f} g/tonne-km"
            }
            
            render_kpi_metrics(metrics)
            
            # Emissions by transport mode
            emissions_by_mode = logistics_summary.get('emissions_by_mode', {})
            
            if emissions_by_mode:
                # Convert to dataframe for plotting
                mode_df = pd.DataFrame({
                    'Transport Mode': list(emissions_by_mode.keys()),
                    'Emissions (kg CO₂e)': list(emissions_by_mode.values())
                })
                
                # Sort by emissions (descending)
                mode_df = mode_df.sort_values('Emissions (kg CO₂e)', ascending=False)
                
                # Convert to tonnes for better readability
                mode_df['Emissions (tonnes CO₂e)'] = mode_df['Emissions (kg CO₂e)'] / 1000
                
                # Create bar chart
                st.bar_chart(mode_df.set_index('Transport Mode')['Emissions (tonnes CO₂e)'])
        else:
            st.info("No logistics emissions data available.")
    
    with emissions_tab2:
        if st.session_state.procurement_emissions is not None:
            procurement_summary = st.session_state.procurement_emissions[1]
            
            # Key metrics
            metrics = {
                "Total Procurement Emissions": f"{procurement_summary.get('total_emissions_kg', 0)/1000:.1f} tonnes CO₂e",
                "Emissions per Unit": f"{procurement_summary.get('emissions_per_unit', 0):.1f} kg CO₂e",
                "Categories": len(procurement_summary.get('emissions_by_category', {}))
            }
            
            render_kpi_metrics(metrics)
            
            # Emissions by category
            emissions_by_category = procurement_summary.get('emissions_by_category', {})
            
            if emissions_by_category:
                # Convert to dataframe for plotting
                category_df = pd.DataFrame({
                    'Product Category': list(emissions_by_category.keys()),
                    'Emissions (kg CO₂e)': list(emissions_by_category.values())
                })
                
                # Sort by emissions (descending)
                category_df = category_df.sort_values('Emissions (kg CO₂e)', ascending=False)
                
                # Convert to tonnes for better readability
                category_df['Emissions (tonnes CO₂e)'] = category_df['Emissions (kg CO₂e)'] / 1000
                
                # Create bar chart
                st.bar_chart(category_df.set_index('Product Category')['Emissions (tonnes CO₂e)'])
        else:
            st.info("No procurement emissions data available.")
    
    with emissions_tab3:
        # Show raw data tables
        data_tab1, data_tab2 = st.tabs(["Logistics Data", "Procurement Data"])
        
        with data_tab1:
            if st.session_state.logistics_data is not None and 'emissions_kg_co2e' in st.session_state.logistics_data.columns:
                st.dataframe(st.session_state.logistics_data)
            else:
                st.info("No logistics data with emissions calculations available.")
        
        with data_tab2:
            if st.session_state.procurement_data is not None and 'emissions_kg_co2e' in st.session_state.procurement_data.columns:
                st.dataframe(st.session_state.procurement_data)
            else:
                st.info("No procurement data with emissions calculations available.")
    
    # Recalculate button
    st.markdown("---")
    if st.button("Recalculate Emissions"):
        with st.spinner("Recalculating emissions..."):
            run_emissions_calculations()
        st.success("Emissions recalculated!")
        st.rerun()

def show_optimization_page():
    """Display the optimization page"""
    
    st.title("Emissions Optimization")
    
    # Check if emissions data is available
    if st.session_state.total_emissions is None:
        st.info("Please analyze your emissions first before exploring optimization opportunities.")
        
        if st.button("Go to Emissions Analysis"):
            run_emissions_calculations()
            st.session_state.current_page = "Emissions Analysis"
            st.rerun()
        
        return
    
    # Check if optimization data is available
    if st.session_state.optimization_targets is None:
        if st.button("Generate Optimization Recommendations"):
            with st.spinner("Analyzing optimization opportunities..."):
                generate_optimization_recommendations()
            st.success("Optimization recommendations generated!")
            st.rerun()
        return
    
    # Display optimization summary
    render_optimization_summary(st.session_state.optimization_targets)
    
    # Detailed optimization tabs
    st.markdown("---")
    st.subheader("Detailed Optimization Analysis")
    
    opt_tab1, opt_tab2, opt_tab3 = st.tabs(["Logistics Optimization", "Procurement Optimization", "AI Strategy"])
    
    with opt_tab1:
        if st.session_state.logistics_data is not None:
            # Generate logistics optimization if needed
            if 'logistics_optimization' not in st.session_state or st.session_state.logistics_optimization is None:
                with st.spinner("Generating logistics optimization..."):
                    optimized_df, summary = generate_optimal_routes(st.session_state.logistics_data)
                    st.session_state.logistics_optimization = (optimized_df, summary)
            
            opt_df, opt_summary = st.session_state.logistics_optimization
            
            # Display key metrics
            metrics = {
                "Original Emissions": f"{opt_summary.get('original_emissions_kg', 0)/1000:.1f} tonnes CO₂e",
                "Optimized Emissions": f"{opt_summary.get('optimized_emissions_kg', 0)/1000:.1f} tonnes CO₂e",
                "Reduction": f"{opt_summary.get('percent_reduction', 0):.1f}%",
                "Shipments Optimized": f"{opt_summary.get('shipments_optimized', 0)}/{opt_summary.get('total_shipments', 0)}"
            }
            
            render_kpi_metrics(metrics)
            
            # Display reduction by method
            reduction_by_method = opt_summary.get('reduction_by_method', {})
            
            if reduction_by_method:
                # Remove 'No change' category if present
                if 'No change' in reduction_by_method:
                    del reduction_by_method['No change']
                
                # Convert to dataframe for plotting
                method_df = pd.DataFrame({
                    'Optimization Method': list(reduction_by_method.keys()),
                    'Emissions Reduction (kg CO₂e)': list(reduction_by_method.values())
                })
                
                # Sort by reduction amount (descending)
                method_df = method_df.sort_values('Emissions Reduction (kg CO₂e)', ascending=False)
                
                # Convert to tonnes for better readability
                method_df['Emissions Reduction (tonnes CO₂e)'] = method_df['Emissions Reduction (kg CO₂e)'] / 1000
                
                # Create bar chart
                st.bar_chart(method_df.set_index('Optimization Method')['Emissions Reduction (tonnes CO₂e)'])
            
            # Display optimized shipments table
            if 'optimized_transport_mode' in opt_df.columns:
                # Filter to only show optimized shipments
                changed_shipments = opt_df[opt_df['optimization_method'] != 'No change']
                
                if not changed_shipments.empty:
                    st.subheader("Optimized Shipments")
                    
                    # Select columns to display
                    display_cols = ['transport_mode', 'optimized_transport_mode', 'distance_km', 
                                   'emissions_kg_co2e', 'optimized_emissions_kg_co2e', 
                                   'emissions_reduction_kg_co2e', 'optimization_method']
                    
                    # Check if all columns exist
                    display_cols = [col for col in display_cols if col in changed_shipments.columns]
                    
                    # Rename columns for better display
                    rename_map = {
                        'transport_mode': 'Current Mode',
                        'optimized_transport_mode': 'Optimized Mode',
                        'distance_km': 'Distance (km)',
                        'emissions_kg_co2e': 'Current Emissions (kg CO₂e)',
                        'optimized_emissions_kg_co2e': 'Optimized Emissions (kg CO₂e)',
                        'emissions_reduction_kg_co2e': 'Reduction (kg CO₂e)',
                        'optimization_method': 'Method'
                    }
                    
                    st.dataframe(changed_shipments[display_cols].rename(columns=rename_map))
        else:
            st.info("No logistics data available for optimization.")
    
    with opt_tab2:
        if st.session_state.procurement_data is not None:
            # Generate procurement optimization if needed
            if 'procurement_optimization' not in st.session_state or st.session_state.procurement_optimization is None:
                with st.spinner("Generating procurement optimization..."):
                    optimized_df, summary = generate_sustainable_procurement_plan(st.session_state.procurement_data)
                    st.session_state.procurement_optimization = (optimized_df, summary)
            
            opt_df, opt_summary = st.session_state.procurement_optimization
            
            # Display key metrics
            metrics = {
                "Original Emissions": f"{opt_summary.get('original_emissions_kg', 0)/1000:.1f} tonnes CO₂e",
                "Optimized Emissions": f"{opt_summary.get('optimized_emissions_kg', 0)/1000:.1f} tonnes CO₂e",
                "Reduction": f"{opt_summary.get('percent_reduction', 0):.1f}%",
                "Cost Impact": f"{opt_summary.get('cost_impact_percent', 0):.1f}%" if opt_summary.get('cost_impact_percent') is not None else "N/A"
            }
            
            render_kpi_metrics(metrics)
            
            # Display reduction by category
            reduction_by_category = opt_summary.get('reduction_by_category', {})
            
            if reduction_by_category:
                # Convert to dataframe for plotting
                category_df = pd.DataFrame({
                    'Product Category': list(reduction_by_category.keys()),
                    'Emissions Reduction (kg CO₂e)': list(reduction_by_category.values())
                })
                
                # Sort by reduction amount (descending)
                category_df = category_df.sort_values('Emissions Reduction (kg CO₂e)', ascending=False)
                
                # Convert to tonnes for better readability
                category_df['Emissions Reduction (tonnes CO₂e)'] = category_df['Emissions Reduction (kg CO₂e)'] / 1000
                
                # Create bar chart
                st.bar_chart(category_df.set_index('Product Category')['Emissions Reduction (tonnes CO₂e)'])
            
            # Display sustainable alternatives
            if 'sustainable_alternative' in opt_df.columns:
                st.subheader("Sustainable Alternatives")
                
                # Group by product category and alternative
                alternatives = opt_df.groupby(['product_category', 'sustainable_alternative']).agg({
                    'emissions_reduction_kg_co2e': 'sum',
                    'emissions_reduction_percent': 'mean',
                    'cost_impact_percent': 'mean',
                    'implementation_difficulty': 'first'
                }).reset_index()
                
                # Convert kg to tonnes
                alternatives['emissions_reduction_tonnes'] = alternatives['emissions_reduction_kg_co2e'] / 1000
                
                # Format for display
                alternatives_display = alternatives[alternatives['sustainable_alternative'] != 'No alternative identified']
                alternatives_display = alternatives_display.rename(columns={
                    'product_category': 'Product Category',
                    'sustainable_alternative': 'Sustainable Alternative',
                    'emissions_reduction_tonnes': 'Reduction Potential (tonnes)',
                    'emissions_reduction_percent': 'Reduction %',
                    'cost_impact_percent': 'Cost Impact %',
                    'implementation_difficulty': 'Difficulty'
                })
                
                st.dataframe(alternatives_display)
        else:
            st.info("No procurement data available for optimization.")
    
    with opt_tab3:
        # Display AI-generated strategy
        if st.session_state.ai_strategy is None:
            if st.button("Generate AI Sustainability Strategy"):
                with st.spinner("Generating comprehensive sustainability strategy..."):
                    strategy = generate_ai_sustainability_strategy(
                        logistics_data=st.session_state.get('logistics_emissions'),
                        procurement_data=st.session_state.get('procurement_emissions'),
                        supplier_data=st.session_state.get('supplier_emissions'),
                        total_emissions=st.session_state.get('total_emissions'),
                        optimization_targets=st.session_state.get('optimization_targets')
                    )
                    
                    st.session_state.ai_strategy = strategy
                st.success("Strategy generated!")
                st.rerun()
        else:
            st.markdown(st.session_state.ai_strategy.get('strategy_text', ''))
            
            if st.button("Regenerate Strategy"):
                with st.spinner("Regenerating sustainability strategy..."):
                    strategy = generate_ai_sustainability_strategy(
                        logistics_data=st.session_state.get('logistics_emissions'),
                        procurement_data=st.session_state.get('procurement_emissions'),
                        supplier_data=st.session_state.get('supplier_emissions'),
                        total_emissions=st.session_state.get('total_emissions'),
                        optimization_targets=st.session_state.get('optimization_targets')
                    )
                    
                    st.session_state.ai_strategy = strategy
                st.success("Strategy regenerated!")
                st.rerun()

def show_supplier_assessment_page():
    """Display the supplier assessment page"""
    
    st.title("Supplier Sustainability Assessment")
    
    # Check if supplier data is available
    if st.session_state.supplier_df is None:
        st.info("Please upload supplier data to perform a sustainability assessment.")
        return
    
    # Generate supplier risk scores if needed
    if 'risk_scores' not in st.session_state or st.session_state.risk_scores is None:
        with st.spinner("Calculating supplier risk scores..."):
            risk_df = calculate_supplier_risk_scores(st.session_state.supplier_df)
            st.session_state.risk_scores = risk_df
            
            # Generate risk assessment
            risk_assessment = identify_risk_hotspots(risk_df)
            st.session_state.risk_assessment = risk_assessment
            
            # Generate risk mitigation strategies
            mitigation_strategies = generate_risk_mitigation_strategies(risk_assessment)
            st.session_state.mitigation_strategies = mitigation_strategies
    
    # Display supplier assessment
    render_supplier_assessment(st.session_state.risk_scores, st.session_state.risk_assessment)
    
    # Display risk mitigation strategies
    st.markdown("---")
    st.subheader("Risk Mitigation Strategies")
    
    if 'mitigation_strategies' in st.session_state and st.session_state.mitigation_strategies:
        for i, strategy in enumerate(st.session_state.mitigation_strategies):
            with st.expander(f"{strategy.get('risk_area', 'Strategy')} ({strategy.get('risk_level', 'Medium')} Risk)", expanded=i==0):
                st.write(f"**Strategy:** {strategy.get('strategy', '')}")
                st.write(f"**Implementation Timeline:** {strategy.get('implementation_timeline', '')}")
                st.write(f"**Expected Impact:** {strategy.get('expected_impact', '')}")
                st.write(f"**Responsible Party:** {strategy.get('responsible_party', '')}")
                
                st.write("**Actions:**")
                for action in strategy.get('actions', []):
                    st.write(f"- {action}")
    else:
        st.info("No risk mitigation strategies available.")
    
    # Display supplier data table
    st.markdown("---")
    st.subheader("Supplier Data")
    
    if 'risk_scores' in st.session_state and st.session_state.risk_scores is not None:
        # Display supplier data with risk scores
        risk_df = st.session_state.risk_scores
        
        # Select columns to display
        display_cols = []
        
        if 'supplier_name' in risk_df.columns:
            display_cols.append('supplier_name')
        elif 'supplier_id' in risk_df.columns:
            display_cols.append('supplier_id')
        
        if 'category' in risk_df.columns:
            display_cols.append('category')
        
        if 'country' in risk_df.columns:
            display_cols.append('country')
        
        # Add risk score columns
        risk_score_cols = [
            'environmental_risk_score', 
            'business_risk_score', 
            'compliance_risk_score', 
            'overall_risk_score', 
            'risk_category'
        ]
        
        display_cols.extend([col for col in risk_score_cols if col in risk_df.columns])
        
        # Rename columns for better display
        rename_map = {
            'supplier_name': 'Supplier Name',
            'supplier_id': 'Supplier ID',
            'category': 'Category',
            'country': 'Country',
            'environmental_risk_score': 'Environmental Risk',
            'business_risk_score': 'Business Risk',
            'compliance_risk_score': 'Compliance Risk',
            'overall_risk_score': 'Overall Risk',
            'risk_category': 'Risk Level'
        }
        
        st.dataframe(risk_df[display_cols].rename(columns=rename_map))
    else:
        st.dataframe(st.session_state.supplier_df)
    
    # Recalculate button
    if st.button("Recalculate Risk Assessment"):
        with st.spinner("Recalculating supplier risk assessment..."):
            # Clear existing risk assessment
            st.session_state.risk_scores = None
            st.session_state.risk_assessment = None
            st.session_state.mitigation_strategies = None
            
            # Recalculate
            risk_df = calculate_supplier_risk_scores(st.session_state.supplier_df)
            st.session_state.risk_scores = risk_df
            
            # Generate risk assessment
            risk_assessment = identify_risk_hotspots(risk_df)
            st.session_state.risk_assessment = risk_assessment
            
            # Generate risk mitigation strategies
            mitigation_strategies = generate_risk_mitigation_strategies(risk_assessment)
            st.session_state.mitigation_strategies = mitigation_strategies
        
        st.success("Risk assessment recalculated!")
        st.rerun()

def show_compliance_page():
    """Display the regulatory compliance page"""
    
    st.title("Regulatory Compliance")
    
    # Check if emissions data is available
    if st.session_state.total_emissions is None:
        st.info("Please analyze your emissions first to assess regulatory compliance.")
        return
    
    # Generate compliance report if needed
    if 'compliance_report' not in st.session_state or st.session_state.compliance_report is None:
        with st.spinner("Generating compliance assessment..."):
            report = generate_compliance_report(
                company_region=st.session_state.company_region,
                industry_sector=st.session_state.industry_sector,
                total_emissions=st.session_state.total_emissions,
                procurement_df=st.session_state.procurement_data,
                facility_zip_codes=["90210", "10001"],  # Example ZIP codes
                revenue_usd=100000000,  # Example revenue
                employee_count=500  # Example employee count
            )
            
            st.session_state.compliance_report = report
    
    # Display compliance assessment
    render_compliance_summary(st.session_state.compliance_report)
    
    # Company settings
    st.markdown("---")
    st.subheader("Company Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        company_name = st.text_input("Company Name", value=st.session_state.company_name)
        if company_name != st.session_state.company_name:
            st.session_state.company_name = company_name
    
    with col2:
        company_region = st.selectbox(
            "Region",
            ["US", "EU", "UK", "China", "India", "Japan", "Brazil", "Other"],
            index=["US", "EU", "UK", "China", "India", "Japan", "Brazil", "Other"].index(st.session_state.company_region)
        )
        if company_region != st.session_state.company_region:
            st.session_state.company_region = company_region
            st.session_state.compliance_report = None  # Clear existing report
    
    with col3:
        industry_sector = st.selectbox(
            "Industry",
            ["Manufacturing", "Retail", "Food & Beverage", "Textiles", "Electronics", 
             "Construction", "Automotive", "Healthcare", "Other"],
            index=["Manufacturing", "Retail", "Food & Beverage", "Textiles", "Electronics", 
                  "Construction", "Automotive", "Healthcare", "Other"].index(st.session_state.industry_sector)
        )
        if industry_sector != st.session_state.industry_sector:
            st.session_state.industry_sector = industry_sector
            st.session_state.compliance_report = None  # Clear existing report
    
    # Facility locations
    st.subheader("Facility Locations")
    
    facility_zip_codes = st.session_state.get('facility_zip_codes', ["90210", "10001"])
    
    zip_col1, zip_col2 = st.columns(2)
    
    with zip_col1:
        facility_zip_1 = st.text_input("Facility 1 ZIP Code", value=facility_zip_codes[0])
    
    with zip_col2:
        facility_zip_2 = st.text_input("Facility 2 ZIP Code", value=facility_zip_codes[1] if len(facility_zip_codes) > 1 else "")
    
    updated_zips = [zip_code for zip_code in [facility_zip_1, facility_zip_2] if zip_code]
    
    if updated_zips != facility_zip_codes:
        st.session_state.facility_zip_codes = updated_zips
        st.session_state.compliance_report = None  # Clear existing report
    
    # Company size
    st.subheader("Company Size")
    
    size_col1, size_col2 = st.columns(2)
    
    with size_col1:
        revenue = st.number_input(
            "Annual Revenue (USD)",
            min_value=0,
            value=st.session_state.get('revenue_usd', 100000000),
            step=1000000,
            format="%d"
        )
        if revenue != st.session_state.get('revenue_usd', 0):
            st.session_state.revenue_usd = revenue
            st.session_state.compliance_report = None  # Clear existing report
    
    with size_col2:
        employees = st.number_input(
            "Number of Employees",
            min_value=0,
            value=st.session_state.get('employee_count', 500),
            step=10,
            format="%d"
        )
        if employees != st.session_state.get('employee_count', 0):
            st.session_state.employee_count = employees
            st.session_state.compliance_report = None  # Clear existing report
    
    # Regenerate button
    if st.button("Regenerate Compliance Assessment"):
        with st.spinner("Regenerating compliance assessment..."):
            report = generate_compliance_report(
                company_region=st.session_state.company_region,
                industry_sector=st.session_state.industry_sector,
                total_emissions=st.session_state.total_emissions,
                procurement_df=st.session_state.procurement_data,
                facility_zip_codes=st.session_state.get('facility_zip_codes', ["90210", "10001"]),
                revenue_usd=st.session_state.get('revenue_usd', 100000000),
                employee_count=st.session_state.get('employee_count', 500)
            )
            
            st.session_state.compliance_report = report
        
        st.success("Compliance assessment regenerated!")
        st.rerun()

def show_reports_page():
    """Display the reports generation page"""
    
    st.title("Sustainability Reports")
    
    # Check if data is available
    if st.session_state.total_emissions is None:
        st.info("Please analyze your emissions first to generate reports.")
        return
    
    st.subheader("Generate Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**PDF Report**")
        st.write("Generate a comprehensive PDF report with all analysis results.")
        
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                pdf_content = generate_pdf_report(
                    company_name=st.session_state.company_name,
                    total_emissions=st.session_state.total_emissions,
                    optimization_targets=st.session_state.optimization_targets,
                    compliance_report=st.session_state.compliance_report,
                    supplier_df=st.session_state.risk_scores if 'risk_scores' in st.session_state else st.session_state.supplier_df,
                    ai_strategy=st.session_state.ai_strategy
                )
            
            if pdf_content:
                st.session_state.pdf_report = pdf_content
                st.success("PDF report generated!")
            else:
                st.error("Failed to generate PDF report.")
        
        if 'pdf_report' in st.session_state and st.session_state.pdf_report:
            download_button(
                st.session_state.pdf_report,
                "sustainability_report.pdf",
                "Download PDF Report"
            )
    
    with col2:
        st.write("**HTML Report**")
        st.write("Generate an interactive HTML report that can be shared online.")
        
        if st.button("Generate HTML Report"):
            with st.spinner("Generating HTML report..."):
                html_content = generate_html_report(
                    company_name=st.session_state.company_name,
                    total_emissions=st.session_state.total_emissions,
                    optimization_targets=st.session_state.optimization_targets,
                    compliance_report=st.session_state.compliance_report,
                    supplier_df=st.session_state.risk_scores if 'risk_scores' in st.session_state else st.session_state.supplier_df,
                    ai_strategy=st.session_state.ai_strategy
                )
            
            if html_content:
                st.session_state.html_report = html_content
                st.success("HTML report generated!")
            else:
                st.error("Failed to generate HTML report.")
        
        if 'html_report' in st.session_state and st.session_state.html_report:
            download_button(
                st.session_state.html_report,
                "sustainability_report.html",
                "Download HTML Report"
            )
    
    # Report preview
    st.markdown("---")
    st.subheader("Report Preview")
    
    if 'html_report' in st.session_state and st.session_state.html_report:
        with st.expander("HTML Report Preview", expanded=True):
            st.components.v1.html(st.session_state.html_report, height=600, scrolling=True)
    elif 'pdf_report' in st.session_state and st.session_state.pdf_report:
        st.info("PDF report generated. Use the download button above to access it.")
    else:
        st.info("Generate a report to see a preview.")

def show_ai_assistant_page():
    """Display the AI assistant chat interface"""
    
    st.title("AI Sustainability Assistant")
    
    # Initialize NLP agent if not already done
    if 'nlp_agent' not in st.session_state:
        st.session_state.nlp_agent = NLPAgent(session_data=st.session_state)
    
    # Display chat interface
    render_ai_chat_interface(st.session_state.nlp_agent)
    
    # Display tips
    with st.expander("Usage Tips"):
        st.markdown("""
        ### Example Questions to Ask:
        
        - What is my current carbon footprint?
        - How can I reduce my logistics emissions?
        - Which suppliers have the highest sustainability risk?
        - What regulations am I subject to?
        - How much could I save by optimizing my supply chain?
        - What's the best way to start reducing my emissions?
        - Generate a 12-month sustainability improvement plan.
        - What's the average emissions per shipment?
        """)

def process_uploaded_data():
    """Process uploaded data files"""
    
    # Check if using sample data
    if st.session_state.get('use_sample_data', False):
        load_sample_dataset()
        return
    
    # Process procurement data
    procurement_file = st.session_state.get('procurement_upload', None)
    if procurement_file is not None:
        procurement_data = load_csv_data(procurement_file)
        st.session_state.procurement_data = procurement_data
    
    # Process logistics data
    logistics_file = st.session_state.get('logistics_upload', None)
    if logistics_file is not None:
        logistics_data = load_csv_data(logistics_file)
        st.session_state.logistics_data = logistics_data
        # Process supplier data
    supplier_file = st.session_state.get('supplier_upload', None)
    if supplier_file is not None:
        supplier_data = load_csv_data(supplier_file)
        st.session_state.supplier_df = supplier_data
    
    # Reset analysis results
    st.session_state.procurement_emissions = None
    st.session_state.logistics_emissions = None
    st.session_state.supplier_emissions = None
    st.session_state.total_emissions = None
    st.session_state.optimization_targets = None
    st.session_state.risk_assessment = None
    st.session_state.compliance_report = None
    st.session_state.ai_strategy = None
    
    # Run analysis if data is available
    run_emissions_calculations()

def load_sample_dataset():
    """Load sample dataset for demonstration"""
    
    # Generate sample data
    procurement_data = generate_sample_procurement_data(100)
    logistics_data = generate_sample_logistics_data(100)
    supplier_data = generate_sample_supplier_data(50)
    
    # Store in session state
    st.session_state.procurement_data = procurement_data
    st.session_state.logistics_data = logistics_data
    st.session_state.supplier_df = supplier_data
    
    # Reset analysis results
    st.session_state.procurement_emissions = None
    st.session_state.logistics_emissions = None
    st.session_state.supplier_emissions = None
    st.session_state.total_emissions = None
    st.session_state.optimization_targets = None
    st.session_state.risk_assessment = None
    st.session_state.compliance_report = None
    st.session_state.ai_strategy = None
    
    # Run analysis
    run_emissions_calculations()

def run_emissions_calculations():
    """Run all emissions calculations and analysis"""
    
    # Calculate procurement emissions
    if st.session_state.procurement_data is not None:
        procurement_result = calculate_procurement_emissions(st.session_state.procurement_data)
        st.session_state.procurement_emissions = procurement_result
        
        # Update the dataframe with emissions data
        st.session_state.procurement_data = procurement_result[0]
    
    # Calculate logistics emissions
    if st.session_state.logistics_data is not None:
        logistics_result = calculate_logistics_emissions(st.session_state.logistics_data)
        st.session_state.logistics_emissions = logistics_result
        
        # Update the dataframe with emissions data
        st.session_state.logistics_data = logistics_result[0]
    
    # Calculate supplier emissions
    if st.session_state.supplier_df is not None:
        supplier_result = calculate_supplier_emissions(st.session_state.supplier_df)
        st.session_state.supplier_emissions = supplier_result
        
        # Update the dataframe with emissions data
        st.session_state.supplier_df = supplier_result[0]
    
    # Calculate total emissions
    total_emissions = calculate_total_emissions(
        logistics_summary=st.session_state.logistics_emissions[1] if st.session_state.logistics_emissions else None,
        procurement_summary=st.session_state.procurement_emissions[1] if st.session_state.procurement_emissions else None,
        supplier_summary=st.session_state.supplier_emissions[1] if st.session_state.supplier_emissions else None
    )
    
    st.session_state.total_emissions = total_emissions
    
    # Generate optimization targets
    if total_emissions:
        optimization_targets = generate_optimization_targets(
            total_summary=total_emissions,
            logistics_df=st.session_state.logistics_data,
            procurement_df=st.session_state.procurement_data,
            supplier_df=st.session_state.supplier_df
        )
        
        st.session_state.optimization_targets = optimization_targets

def generate_optimization_recommendations():
    """Generate all optimization recommendations"""
    
    # Generate optimization targets if not already available
    if st.session_state.optimization_targets is None and st.session_state.total_emissions is not None:
        optimization_targets = generate_optimization_targets(
            total_summary=st.session_state.total_emissions,
            logistics_df=st.session_state.logistics_data,
            procurement_df=st.session_state.procurement_data,
            supplier_df=st.session_state.supplier_df
        )
        
        st.session_state.optimization_targets = optimization_targets
    
    # Generate logistics optimization
    if st.session_state.logistics_data is not None:
        optimized_df, summary = generate_optimal_routes(st.session_state.logistics_data)
        st.session_state.logistics_optimization = (optimized_df, summary)
    
    # Generate procurement optimization
    if st.session_state.procurement_data is not None:
        optimized_df, summary = generate_sustainable_procurement_plan(st.session_state.procurement_data)
        st.session_state.procurement_optimization = (optimized_df, summary)
    
    # Generate supplier optimization
    if st.session_state.supplier_df is not None:
        # Calculate risk scores if not already done
        if 'risk_scores' not in st.session_state or st.session_state.risk_scores is None:
            risk_df = calculate_supplier_risk_scores(st.session_state.supplier_df)
            st.session_state.risk_scores = risk_df
            
            # Generate risk assessment
            risk_assessment = identify_risk_hotspots(risk_df)
            st.session_state.risk_assessment = risk_assessment
            
            # Generate risk mitigation strategies
            mitigation_strategies = generate_risk_mitigation_strategies(risk_assessment)
            st.session_state.mitigation_strategies = mitigation_strategies
        
        # Generate supplier optimization
        optimized_df, summary = generate_supplier_sustainability_plan(st.session_state.risk_scores)
        st.session_state.supplier_optimization = (optimized_df, summary)
    
    # Generate AI sustainability strategy
    strategy = generate_ai_sustainability_strategy(
        logistics_data=st.session_state.get('logistics_emissions'),
        procurement_data=st.session_state.get('procurement_emissions'),
        supplier_data=st.session_state.get('supplier_emissions'),
        total_emissions=st.session_state.get('total_emissions'),
        optimization_targets=st.session_state.get('optimization_targets')
    )
    
    st.session_state.ai_strategy = strategy

if __name__ == "__main__":
    setup_dashboard()
