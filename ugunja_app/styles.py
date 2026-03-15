"""Custom styling for the Streamlit app."""

import streamlit as st


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* --- HIDE STREAMLIT BRANDING --- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* --- CUSTOM BRAND COLORS (UGUNJA ORANGE) --- */
        :root {
            --primary-color: #FFA500; /* Ugunja Orange */
            --background-color: #0E1117; /* Streamlit Dark BG */
            --secondary-background-color: #1a1f2b; /* Card BG */
            --text-color: #FAFAFA;
            --dark-grey: #333333;
        }

        /* --- APPLY BRAND COLOR TO WIDGETS --- */
        
        /* Slider Track */
        div[data-baseweb="slider"] > div:nth-child(2) > div {
            background: var(--primary-color) !important;
        }
        /* Slider Thumb */
        div[data-baseweb="slider"] > div:nth-child(3) {
            background: var(--primary-color) !important;
        }
        /* Radio buttons */
        div[data-baseweb="radio"] > div > label > div[role="radio"] {
            background: var(--primary-color) !important;
            border-color: var(--primary-color) !important;
        }
        
        /* --- STYLE METRICS TO BE HIGH-IMPACT --- */
        div[data-testid="stMetricValue"] {
            font-size: 2.75rem; /* Make the number bigger */
            font-weight: 700;
            color: var(--primary-color); /* Make the metric value ORANGE */
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 1.1rem;
            font-weight: 400;
            color: var(--text-color);
        }
        
        div[data-testid="stMetricDelta"] {
            font-size: 1rem !important;
        }

        /* --- STYLE SIDEBAR --- */
        [data-testid="stSidebar"] {
            background-color: #141820; /* Slightly different dark for sidebar */
        }
        [data-testid="stSidebar"] .st-emotion-cache-16txtl3 { /* Sidebar radio labels */
            font-size: 1.1rem;
            font-weight: 500;
        }
        
        /* --- STYLE TABS --- */
        button[data-baseweb="tab"] {
            font-size: 1rem;
            font-weight: 500;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--primary-color);
            border-bottom-color: var(--primary-color);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
