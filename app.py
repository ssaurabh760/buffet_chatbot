import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from helper import (
    PositionalEncoding, 
    MultiHeadAttentionLayer,
    create_padding_mask,
    create_look_ahead_mask,
    scaled_dot_product_attention,
    add_pos_enc, 
    predict
)

# Load model with ALL custom objects
model = load_model('./models/model.h5', custom_objects={
    "PositionalEncoding": PositionalEncoding,
    "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
    "create_padding_mask": create_padding_mask,
    "create_look_ahead_mask": create_look_ahead_mask,
    "scaled_dot_product_attention": scaled_dot_product_attention,
    "add_pos_enc": add_pos_enc,  
})

if "ti" not in st.session_state:
    st.session_state["ti"] = ""

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = ""

def generate_response(text, model):
    prediction = predict(text, model)
    return prediction

# Streamlit UI
st.title("🌳 Livermore Chatbot")
st.write("Ask anything about investing and trading!")

st.markdown(
    """
    <style>
    .scrollable-box {
        height: 400px;
        overflow-y: auto;
        background-color: #f4f4f4;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-family: monospace;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<div class='scrollable-box'>{st.session_state['conversation_history']}</div>",
    unsafe_allow_html=True,
)

def submit():
    user_input = st.session_state["ti"]
    if user_input:
        response = generate_response(user_input, model)
        st.session_state["conversation_history"] += f"<b>You:</b> {user_input}<br><br><b>Livermore:</b> {response}<br><br>"
        st.session_state["ti"] = ""

st.text_input("You: ", key="ti", on_change=submit)

st.markdown("---")
st.markdown("**Livermore Chatbot** - Powered by TensorFlow Transformer")