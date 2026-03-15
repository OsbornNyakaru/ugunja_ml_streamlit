"""Ugunja Intelligence Engine Streamlit app."""

import streamlit as st

from ugunja_app import config
from ugunja_app.data import load_data
from ugunja_app.sidebar import render_sidebar
from ugunja_app.styles import apply_custom_css
from ugunja_app.pages import (
    demand_forecasting,
    predictive_maintenance,
    route_optimization,
)


st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE,
)

apply_custom_css()

demand_df, maintenance_df, route_df = load_data()

if demand_df is None:
    st.stop()

page = render_sidebar()

if page == config.PAGE_ROUTE_OPT:
    route_optimization.render(route_df)
elif page == config.PAGE_PRED_MAINT:
    predictive_maintenance.render(maintenance_df)
elif page == config.PAGE_DEMAND:
    demand_forecasting.render(demand_df)
