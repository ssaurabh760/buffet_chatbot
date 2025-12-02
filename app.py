"""
Warren Buffett Investment Advisor Chatbot
Powered by TensorFlow Transformer

Streamlit Application for serving the chatbot
"""

import streamlit as st
import tensorflow as tf
from helper import (
    PositionalEncoding, 
    MultiHeadAttentionLayer,
    create_padding_mask,
    create_look_ahead_mask,
    scaled_dot_product_attention,
    add_pos_enc, 
    predict,
    get_tokenizer
)
import os

# Page configuration
st.set_page_config(
    page_title="🦉 Warren Buffett Investment Advisor",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
    <style>
    .scrollable-box {
        height: 500px;
        overflow-y: auto;
        background-color: #f0f2f6;
        padding: 15px;
        border: 2px solid #ddd;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .chat-user {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #2196F3;
    }
    .chat-assistant {
        background-color: #f3e5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #9c27b0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained transformer model."""
    try:
        # Try SavedModel format first (recommended)
        if os.path.exists('./models/buffett_model.keras'):
            st.info("Loading SavedModel format...")
            model = tf.keras.models.load_model('./models/buffett_model.keras')
            if model:
                st.info("True")
            return model
        
        # Fallback to H5 format with custom objects
        elif os.path.exists('./models/model.h5'):
            st.warning("Loading H5 format (legacy)...")
            model = tf.keras.models.load_model('./models/model.h5', custom_objects={
                "PositionalEncoding": PositionalEncoding,
                "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
                "create_padding_mask": create_padding_mask,
                "create_look_ahead_mask": create_look_ahead_mask,
                "scaled_dot_product_attention": scaled_dot_product_attention,
                "add_pos_enc": add_pos_enc,  
            })
            return model
        else:
            st.error("❌ Model files not found!")
            st.info("Place either './models/buffett_model' or './models/model.h5' in your project")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Make sure the model and tokenizer files are in the './models/' directory")
        st.stop()

# Initialize session state
if "model" not in st.session_state:
    with st.spinner("🦉 Loading Warren Buffett AI..."):
        st.session_state.model = load_model()
        st.session_state.tokenizer = get_tokenizer()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("## 🦉")
with col2:
    st.markdown("""
    # Warren Buffett Investment Advisor
    *Powered by Transformer AI trained on investment wisdom*
    """)

# Sidebar information
with st.sidebar:
    st.markdown("### 📊 About This Chatbot")
    st.markdown("""
    This is an AI chatbot trained on Warren Buffett's investment philosophy:
    
    **Features:**
    - ✅ Value investing principles
    - ✅ Stock analysis techniques
    - ✅ Business evaluation
    - ✅ Risk management
    - ✅ Asset allocation strategies
    
    **Model:**
    - Transformer architecture (TensorFlow)
    - Trained on 600 Q&A pairs
    - 256-dimensional embeddings
    
    **Limitations:**
    - Responses based on training data
    - Not real-time market data
    - Educational purposes only
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Responses")
    st.markdown("""
    1. Ask specific investment questions
    2. Use clear, complete sentences
    3. Ask about business analysis concepts
    4. Get explanations of investment philosophy
    """)
    
    st.markdown("---")
    if st.button("🔄 Clear History"):
        st.session_state.conversation_history = []
        st.success("Conversation cleared!")
        st.rerun()

# Main content area
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 💬 Conversation")
with col2:
    if st.button("📋 Copy All"):
        st.info("Copy the conversation below")

# Display conversation history
if st.session_state.conversation_history:
    conversation_text = ""
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            conversation_text += f'<div class="chat-user"><b>You:</b> {msg["content"]}</div>'
        else:
            conversation_text += f'<div class="chat-assistant"><b>Warren Buffett:</b> {msg["content"]}</div>'
    
    st.markdown(conversation_text, unsafe_allow_html=True)
else:
    st.markdown(
        '<div style="text-align: center; color: #666; padding: 40px;">'
        '<p>💬 No conversation yet. Ask Warren Buffett something!</p>'
        '</div>',
        unsafe_allow_html=True
    )

# Input section
st.markdown("---")
st.markdown("### ✍️ Your Question")

col1, col2 = st.columns([5, 1])

with col1:
    user_question = st.text_input(
        "Ask anything about investing:",
        placeholder="e.g., What is your investing philosophy?",
        key="user_question_input",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("📤 Send", use_container_width=True)

# Process user input
if send_button and user_question.strip():
    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_question
    })
    
    # Generate response
    with st.spinner("🤔 Warren is thinking..."):
        try:
            response = predict(user_question, st.session_state.model)
            
            # Clean up response
            if isinstance(response, bytes):
                response = response.decode('utf-8')
            response = str(response).strip()
            
            if not response or response.lower() == "unknown":
                response = "I appreciate your question, but I don't have enough information to provide a detailed answer. Could you rephrase your question or ask about another investment topic?"
            
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            st.info("Try asking a different question")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Model:** TensorFlow Transformer")
with col2:
    st.markdown("**Training Data:** 600 Q&A pairs")
with col3:
    st.markdown("**Status:** ✅ Online")