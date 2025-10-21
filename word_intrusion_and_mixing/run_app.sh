#!/bin/bash
"""
Script to run the Word Intrusion Streamlit Application

This script launches the Streamlit web interface for the word intrusion package.
"""

echo "🚀 Starting Word Intrusion Streamlit Application..."
echo ""
echo "📝 This application provides:"
echo "   • File processing to unified format"
echo "   • Word intrusion task generation"
echo "   • Interactive preview and configuration"
echo ""
echo "🌐 The app will open in your default browser"
echo "   If it doesn't open automatically, go to: http://localhost:8501"
echo ""


# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -q streamlit pandas numpy

# Run the Streamlit app
echo "🚀 Launching application..."
streamlit run streamlit_app.py
