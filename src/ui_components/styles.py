"""
MEMBER 3: Dashboard Styling & Custom CSS Module
"""

import streamlit as st


def apply_custom_styles():
    """
    Injects custom CSS for modern, responsive card layout and clean aesthetics.
    """
    st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        
        .hero-container {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
            border-radius: 14px;
            padding: 22px 28px;
            color: #ffffff;
            margin-bottom: 20px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);
        }
        .hero-title {
            font-size: 1.85rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.3px;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            color: #cbd5e1;
            margin-top: 6px;
            line-height: 1.5;
        }

        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
            text-align: center;
        }
        @media (prefers-color-scheme: dark) {
            .metric-card {
                background-color: #1e293b;
                border-color: #334155;
            }
        }
    </style>
    """, unsafe_allow_html=True)
