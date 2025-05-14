"""
Natural Language Processing agent for conversational interface with the Supply Chain Optimizer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import json
import re
from datetime import datetime

from app.utils.api_clients import query_claude, query_openai
from app.config import get_logger

logger = get_logger(__name__)

class NLPAgent:
    """
    NLP Agent for handling natural language queries and generating responses using AI.
    """
    
    def __init__(self, session_data: Dict[str, Any] = None):
        """
        Initialize the NLP Agent
        
        Args:
            session_data: Dictionary containing session data
        """
        self.session_data = session_data or {}
        self.conversation_history = []
        self.logger = get_logger(__name__)
    
    def process_query(
        self,
        query: str,
        total_emissions: Optional[Dict[str, Any]] = None,
        optimization_data: Optional[Dict[str, Any]] = None,
        supplier_data: Optional[Dict[str, Any]] = None,
        compliance_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language query and generate a response
        
        Args:
            query: User's query
            total_emissions: Emissions data (optional)
            optimization_data: Optimization data (optional)
            supplier_data: Supplier data (optional)
            compliance_data: Compliance data (optional)
            
        Returns:
            dict: Response with answer and any additional data
        """
        # Ensure query is not empty
        if not query or not query.strip():
            return {
                "answer": "I don't see a question. How can I help you with your supply chain sustainability?",
                "query_type": "empty",
                "data": {}
            }
        
        # Normalize query
        query = query.strip()
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Determine query type
        query_type = self._classify_query(query)
        
        # Get relevant data based on query type
        context_data = self._get_context_data(
            query_type=query_type,
            total_emissions=total_emissions,
            optimization_data=optimization_data,
            supplier_data=supplier_data,
            compliance_data=compliance_data
        )
        
        # Generate response
        response = self._generate_ai_response(
            query=query,
            query_type=query_type,
            context_data=context_data
        )
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response["answer"]})
        
        # Return response with data
        return response
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the type of query to determine how to handle it
        
        Args:
            query: User's query
            
        Returns:
            str: Query type
        """
        query_lower = query.lower()
        
        # Define patterns for different query types
        patterns = {
            "emissions": r"\b(carbon|emissions|footprint|co2|greenhouse gas|ghg)\b",
            "optimization": r"\b(optimize|optimization|reduce|saving|improve|strategy|recommendation)\b",
            "supplier": r"\b(supplier|vendor|sourcing|procurement)\b",
            "compliance": r"\b(compliance|regulation|policy|standard|law|legal|requirement)\b",
            "cost": r"\b(cost|price|expense|saving|budget|financial)\b",
            "report": r"\b(report|document|certificate|evidence|documentation)\b"
        }
        
        # Check each pattern
        matches = {}
        for query_type, pattern in patterns.items():
            matches[query_type] = len(re.findall(pattern, query_lower))
        
        # Get the query type with the most matches
        primary_type = max(matches.items(), key=lambda x: x[1])[0]
        
        # If no significant matches, classify as general
        if matches[primary_type] == 0:
            return "general"
        
        return primary_type
    
    def _get_context_data(
        self,
        query_type: str,
        total_emissions: Optional[Dict[str, Any]] = None,
        optimization_data: Optional[Dict[str, Any]] = None,
        supplier_data: Optional[Dict[str, Any]] = None,
        compliance_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get context data relevant to the query type
        
        Args:
            query_type: Type of query
            total_emissions: Emissions data
            optimization_data: Optimization data
            supplier_data: Supplier data
            compliance_data: Compliance data
            
        Returns:
            dict: Context data
        """
        context_data = {
            "query_type": query_type,
            "available_data": []
        }
        
        # Add relevant data based on query type
        if query_type in ["emissions", "general", "cost"] and total_emissions:
            context_data["total_emissions"] = total_emissions
            context_data["available_data"].append("emissions")
        
        if query_type in ["optimization", "general", "cost"] and optimization_data:
            context_data["optimization_data"] = optimization_data
            context_data["available_data"].append("optimization")
        
        if query_type in ["supplier", "general"] and supplier_data:
            context_data["supplier_data"] = supplier_data
            context_data["available_data"].append("supplier")
        
        if query_type in ["compliance", "general", "report"] and compliance_data:
            context_data["compliance_data"] = compliance_data
            context_data["available_data"].append("compliance")
        
        return context_data
    
    def _generate_ai_response(
        self,
        query: str,
        query_type: str,
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an AI response to the query using available context data
        
        Args:
            query: User's query
            query_type: Type of query
            context_data: Context data
            
        Returns:
            dict: Response with answer and any additional data
        """
        # Prepare prompt for the AI
        prompt = self._prepare_prompt(query, query_type, context_data)
        
        try:
            # Try to use Claude for response generation
            response_text = query_claude(prompt)
            ai_service = "Claude"
        except Exception as e:
            self.logger.warning(f"Error using Claude API: {str(e)}. Attempting with OpenAI...")
            
            try:
                # Fallback to OpenAI
                response_text = query_openai(prompt)
                ai_service = "OpenAI"
            except Exception as e2:
                self.logger.error(f"Error generating AI response: {str(e2)}")
                
                # Return error message as the response
                return {
                    "answer": "I apologize, but I'm having trouble generating a response. Please try again or rephrase your question.",
                    "query_type": query_type,
                    "error": str(e2),
                    "data": {}
                }
        
        # Extract any structured data from the response
        extracted_data = self._extract_data_from_response(response_text)
        
        # Construct final response
        response = {
            "answer": response_text,
            "query_type": query_type,
            "generated_by": ai_service,
            "timestamp": datetime.now().isoformat(),
            "data": extracted_data
        }
        
        return response
    
    def _prepare_prompt(
        self,
        query: str,
        query_type: str,
        context_data: Dict[str, Any]
    ) -> str:
        """
        Prepare a prompt for the AI model
        
        Args:
            query: User's query
            query_type: Type of query
            context_data: Context data
            
        Returns:
            str: Prompt for the AI model
        """
        # Convert context data to JSON for the prompt
        context_json = json.dumps(context_data, indent=2)
        
        # Format conversation history
        conversation_context = ""
        if self.conversation_history:
            conversation_context = "\nOur conversation so far:\n"
            for i, message in enumerate(self.conversation_history[-4:]):  # Include only the last 4 messages
                role = "User" if message["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {message['content']}\n"
        
        # Create base prompt with system instructions
        prompt = f"""
You are an AI assistant specializing in sustainable supply chain management. You help users understand and optimize their supply chain's environmental impact, compliance status, and sustainability performance.

I'll provide you with data about the user's supply chain, and your task is to answer their question accurately and helpfully based on this data.

Here's the supply chain data:
{conversation_context}
The user's question is: {query}

Please provide a clear, concise, and actionable response. If the data doesn't contain information needed to answer the question, acknowledge that and suggest what information would be needed.

If responding to a question about emissions, include specific numbers. If asked about optimization, provide specific recommendations. If asked about compliance, highlight key regulations and requirements.

Make sure your response is professional, accurate, and focuses on the data provided. Avoid making claims that aren't supported by the data.
"""
        
        return prompt
    
    def _extract_data_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract structured data from the AI response if available
        
        Args:
            response_text: AI's response text
            
        Returns:
            dict: Extracted structured data
        """
        extracted_data = {}
        
        # Look for numeric values related to emissions
        emissions_match = re.search(r'(\d+\.?\d*)\s*(tonnes|tons|kg)\s*CO2e', response_text, re.IGNORECASE)
        if emissions_match:
            value = float(emissions_match.group(1))
            unit = emissions_match.group(2).lower()
            
            # Convert to kg for consistency
            if 'ton' in unit:
                value = value * 1000
            
            extracted_data['emissions_value'] = value
            extracted_data['emissions_unit'] = 'kg CO2e'
        
        # Look for percentage reductions
        reduction_match = re.search(r'(\d+\.?\d*)%\s*(reduction|decrease)', response_text, re.IGNORECASE)
        if reduction_match:
            extracted_data['reduction_percentage'] = float(reduction_match.group(1))
        
        # Look for recommendations
        recommendations = []
        
        # Pattern to match numbered or bulleted recommendations
        recommendation_patterns = [
            r'\d+\.\s*([A-Z][^.!?]*[.!?])',  # Numbered points
            r'[-•]\s*([A-Z][^.!?]*[.!?])'    # Bulleted points
        ]
        
        for pattern in recommendation_patterns:
            matches = re.findall(pattern, response_text)
            recommendations.extend(matches)
        
        if recommendations:
            extracted_data['recommendations'] = recommendations
        
        return extracted_data
