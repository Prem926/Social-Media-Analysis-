import subprocess

import streamlit as st
from streamlit_extras.colored_header import colored_header

# Set up the page configuration
st.set_page_config(
    page_title="Jugadoo",
    page_icon="🎭",
    layout="centered",
)

# Add a colored header for a modern look
colored_header(
    label="Welcome to Jugadoo!",
    color_name="blue-70",
)

# Add an image or logo (optional)
st.image(
    "./Jugadoo.png",  # Replace with your image URL or local file path
    caption="Where ideas come to life!",
    use_container_width =True,
)

# Create columns for buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Go to Page Text insights"):
        subprocess.Popen(["streamlit", "run", "analysis.py"])  # Launch Page 1
        st.stop()

with col2:
    if st.button("🎉 Go to Page Visual analysis"):
        subprocess.Popen(["streamlit", "run", "main.py"])  # Launch Page 2
        st.stop()

# Add a footer or motivational message
st.markdown(
    """
    ---
    *Powered by Jugadoo Labs* 💡
    """
)
