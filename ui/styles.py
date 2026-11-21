import os
import streamlit as st


def load_css(css_path: str = "static/styles.css") -> None:
    """Load external CSS stylesheet into Streamlit."""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)
