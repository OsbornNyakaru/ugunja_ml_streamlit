"""Sidebar UI components."""

import streamlit as st

from . import config


def render_sidebar() -> str:
    st.sidebar.image(config.LOGO_URL, width=200)
    st.sidebar.title(config.SIDEBAR_TITLE)

    page = st.sidebar.radio(
        "Select a Model's Insights",
        config.PAGES,
        label_visibility="collapsed",
    )

    st.sidebar.info(config.SIDEBAR_INFO)
    return page
