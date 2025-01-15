import json
import os

import requests
import streamlit as st
from dotenv import dotenv_values, load_dotenv

load_dotenv() 
# Constants
BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = os.getenv("LANGFLOW_ID")
FLOW_ID = os.getenv("FLOW_ID")
APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ENDPOINT = ""
st.set_page_config(
    page_title="Analysis",
    page_icon="🔬",
    layout="centered",
)
def run_flow(message, endpoint, tweaks, application_token):
    """
    Run the flow and return the JSON response.
    """
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{endpoint}"

    # Prepare the payload without duplicating 'input_value'
    payload = {
        "output_type": "chat",
        "input_type": "chat",
    }

    # If tweaks have 'input_value', do not set it directly in the payload
    if tweaks and "ChatInput-uKcNV" in tweaks and "input_value" in tweaks["ChatInput-uKcNV"]:
        payload["tweaks"] = tweaks
    else:
        payload["input_value"] = message
        if tweaks:
            payload["tweaks"] = tweaks

    headers = {
        "Authorization": f"Bearer {application_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

# Streamlit App
st.title("Langflow API Frontend")
st.write("Interact with Langflow API and customize flow settings.")
# Input Fields
message = st.text_area("Input Message", placeholder="Enter your message here...")

endpoint = st.text_input("Flow Endpoint", value=ENDPOINT or FLOW_ID)
application_token = st.text_input("Application Token", value=APPLICATION_TOKEN, type="password")

default_tweaks = {
    "ChatInput-uKcNV": {
        "files": "",
        "background_color": "",
        "chat_icon": "",
        "input_value": "",
        "sender": "User",
        "sender_name": "User",
        "session_id": "",
        "should_store_message": True,
        "text_color": "",
    },
}
tweaks = st.text_area(
    "Flow Tweaks (JSON Format)",
    value=json.dumps(default_tweaks, indent=2),
    height=200,
)

# Run Flow Button
if st.button("Run Flow"):
    try:
        tweaks_dict = json.loads(tweaks)

        # Remove `input_value` from tweaks if it exists
        if "ChatInput-uKcNV" in tweaks_dict and "input_value" in tweaks_dict["ChatInput-uKcNV"]:
            del tweaks_dict["ChatInput-uKcNV"]["input_value"]

        response = run_flow(
            message=message,
            endpoint=endpoint,
            tweaks=tweaks_dict,
            application_token=application_token,
        )

        st.subheader("Conversation Output")
        conversation_text = response.get("outputs")[0].get("outputs")[0].get("outputs").get("message").get("message").get("text")

        # Format the response as a styled conversation
        st.markdown("### AI's Response:")
        for line in conversation_text.split("\n"):
            if line.strip().startswith("* "):
                st.markdown(f"- {line.strip()[2:]}")
            elif line.strip().startswith("**"):
                st.markdown(f"**{line.strip()}**")
            else:
                st.write(line.strip())

    except Exception as e:
        st.error(f"An error occurred: {e}")
