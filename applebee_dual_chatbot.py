"""
applebee_dual_chatbot.py

AppleBee Project with:
- Tab 1: 📊 Warren Buffett Dashboard (existing)
- Tab 2: 🤖 Groq API Chatbot (cloud-based)
- Tab 3: 🏗️ Local Transformer Chatbot (local inference)

Run with: streamlit run applebee_dual_chatbot.py
"""

import streamlit as st
import tensorflow as tf
from groq import Groq
from local_transformer_helper import TransformerChatbot, load_or_create_chatbot
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AppleBee - Dual Chatbot",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🌳 AppleBee")
    st.markdown("---")
    st.markdown("""
    **Warren Buffett Investment Dashboard**
    
    with Dual AI Chatbots:
    - 🤖 Cloud-based (Groq)
    - 🏗️ Local Transformer
    """)
    st.markdown("---")
    
    # Model selection
    st.markdown("### ⚙️ Settings")
    
    # Groq settings
    groq_enabled = st.checkbox("Enable Groq API Chatbot", value=True)
    
    # Transformer settings
    transformer_enabled = st.checkbox("Enable Local Transformer Chatbot", value=False)
    
    st.markdown("---")
    
    # About
    with st.expander("ℹ️ About This Project"):
        st.markdown("""
        **AppleBee** combines:
        
        1. **Dashboard**: Warren Buffett-style stock analysis
        2. **Chatbots**: Two different AI approaches
           - Groq API: High quality, cloud-based
           - Transformer: Local inference, educational
        
        **Made by**: Saurabh
        **Status**: Development
        """)

# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Groq API", "🏗️ Local Transformer"])

# =============================================================================
# TAB 1: DASHBOARD
# =============================================================================

with tab1:
    st.title("📊 Warren Buffett Stock Analysis Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Portfolio Value", "$1.2M", "+2.5%")
    
    with col2:
        st.metric("YTD Return", "15.8%", "+1.2%")
    
    with col3:
        st.metric("Buffett Style Score", "8.7/10", "-0.1")
    
    st.markdown("---")
    
    # Sample stocks
    st.subheader("📈 Analyzed Stocks")
    
    sample_data = {
        "Stock": ["BRK-B", "MSFT", "AAPL", "JPM", "KO"],
        "Price": ["$384.50", "$425.30", "$178.90", "$156.20", "$67.45"],
        "Buffett Score": [9.2, 8.5, 7.8, 8.9, 8.1],
        "P/E Ratio": [22.5, 35.2, 28.4, 10.2, 24.1],
        "Dividend": ["2.18%", "0.71%", "0.42%", "2.65%", "3.06%"],
    }
    
    st.dataframe(sample_data, use_container_width=True)
    
    st.markdown("---")
    
    st.info("""
    💡 **Note**: This is a placeholder dashboard. 
    
    In the actual project, this would show:
    - Real stock data from Yahoo Finance
    - Financial metrics (P/E, PEG, Debt/Equity, etc.)
    - Buffett-style investment analysis
    - Technical and fundamental analysis charts
    """)

# =============================================================================
# TAB 2: GROQ API CHATBOT
# =============================================================================

with tab2:
    st.title("🤖 Investment Advisor - Groq API")
    
    st.markdown("""
    **AI Chatbot powered by Groq's Llama 3.1 Model** (Cloud-based)
    
    ✨ Features:
    - Real-time responses via API
    - Integrated with stock data
    - High-quality financial advice
    - Understands investment concepts
    """)
    
    # Check if Groq is enabled
    if not groq_enabled:
        st.warning("⚠️ Groq API Chatbot is disabled in settings")
        st.stop()
    
    # Initialize session state
    if "groq_chat_history" not in st.session_state:
        st.session_state.groq_chat_history = []
    
    # Display chat history
    for message in st.session_state.groq_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    user_input = st.chat_input("Ask about investment strategies, stocks, or Warren Buffett...")
    
    if user_input:
        # Add user message
        st.session_state.groq_chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get Groq API key
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            st.error("❌ GROQ_API_KEY not found in .env file")
            st.info("Please add your Groq API key to .env file")
        else:
            # Show thinking animation
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                with message_placeholder.container():
                    with st.spinner("🤔 Thinking..."):
                        try:
                            # Create Groq client
                            client = Groq(api_key=api_key)
                            
                            # Build messages with system prompt
                            messages = [
                                {
                                    "role": "system",
                                    "content": """You are an expert financial education advisor specializing in 
                                    Warren Buffett's value investing principles. 
                                    
                                    You should:
                                    - Explain investment concepts clearly
                                    - Reference Warren Buffett's philosophy
                                    - Provide practical advice
                                    - Mention relevant financial metrics
                                    - Always remind users that this is educational and not financial advice
                                    
                                    Keep responses concise and professional."""
                                }
                            ]
                            
                            # Add conversation history
                            for msg in st.session_state.groq_chat_history[:-1]:  # Exclude current message
                                messages.append(msg)
                            
                            # Add current user message
                            messages.append({
                                "role": "user",
                                "content": user_input
                            })
                            
                            # Get response from Groq
                            response = client.chat.completions.create(
                                model="llama-3.1-70b-versatile",
                                messages=messages,
                                max_tokens=1024,
                                temperature=0.7,
                            )
                            
                            assistant_message = response.choices[0].message.content
                            
                            # Display response
                            message_placeholder.markdown(assistant_message)
                            
                            # Add to history
                            st.session_state.groq_chat_history.append({
                                "role": "assistant",
                                "content": assistant_message
                            })
                            
                        except Exception as e:
                            message_placeholder.error(f"❌ Error: {str(e)}")
                            st.info("Make sure your GROQ_API_KEY is valid")

# =============================================================================
# TAB 3: LOCAL TRANSFORMER CHATBOT
# =============================================================================

with tab3:
    st.title("🏗️ Local Transformer Chatbot")
    
    st.markdown("""
    **AI Chatbot powered by Local TensorFlow Transformer**
    
    ✨ Features:
    - Runs locally (no API calls)
    - Trained on movie dialogues
    - Offline-capable
    - Educational value
    """)
    
    # Check if transformer is enabled
    if not transformer_enabled:
        st.info("💡 Local Transformer Chatbot is currently disabled in settings")
        st.markdown("""
        To enable this chatbot:
        1. Scroll to sidebar and check "Enable Local Transformer Chatbot"
        2. Ensure model files are in the correct location
        """)
        st.stop()
    
    # Model paths
    model_path = "./models/model.h5"
    tokenizer_path = "./models/tokenizer_vocab.subwords"
    
    # Check if model exists
    model_exists = os.path.exists(model_path) and os.path.exists(tokenizer_path)
    
    if not model_exists:
        st.warning("⚠️ Model files not found!")
        
        st.markdown("""
        ### 📥 Setup Instructions
        
        To use the local transformer chatbot:
        
        **Option A: Download Pre-trained Model**
        1. Visit: https://github.com/bryanlimy/tf2-transformer-chatbot
        2. Download `model.h5` and `tokenizer_vocab`
        3. Create directory: `./models/`
        4. Place files there:
           - `./models/transformer_model.h5`
           - `./models/tokenizer`
        
        **Option B: Train Your Own Model**
        1. Use professor's notebook: `tf2_tpu_transformer_chatbot.ipynb`
        2. Run on Google Colab (free GPU/TPU)
        3. Download trained model
        4. Place in `./models/` directory
        
        **Resources:**
        - 📓 Professor's Notebook
        - 📚 TensorFlow Documentation
        - 🔗 GitHub Repository
        """)
        st.stop()
    
    # Initialize session state
    if "local_chat_history" not in st.session_state:
        st.session_state.local_chat_history = []
    
    # Load model once (cached)
    @st.cache_resource
    def load_transformer():
        """Load transformer model (cached to avoid reloading)"""
        try:
            return load_or_create_chatbot(model_path, tokenizer_path)
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None
    
    chatbot = load_transformer()
    
    if chatbot is None:
        st.stop()
    
    # Display chat history
    for message in st.session_state.local_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    user_input = st.chat_input("Chat with the local transformer...")
    
    if user_input:
        # Add user message
        st.session_state.local_chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with message_placeholder.container():
                with st.spinner("🤔 Generating response..."):
                    try:
                        response = chatbot.predict(user_input)
                        message_placeholder.markdown(response)
                        
                        st.session_state.local_chat_history.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                    except Exception as e:
                        message_placeholder.error(f"❌ Error: {str(e)}")
    
    # Model info
    with st.expander("📊 Model Information"):
        st.markdown(f"""
        **Model Details:**
        - **Architecture**: Transformer (Encoder-Decoder)
        - **Training Data**: Cornell Movie-Dialogs Corpus (220k conversations)
        - **Max Sequence Length**: 40 tokens
        - **Vocabulary Size**: {chatbot.tokenizer.vocab_size + 2:,}
        - **Trainable Parameters**: {chatbot.model.count_params():,}
        
        **How It Works:**
        1. Encodes your input using an encoder
        2. Decodes response using a decoder
        3. Generates tokens one at a time (auto-regressive)
        4. Uses attention mechanisms to focus on relevant parts
        
        **Limitations:**
        - Trained on movie dialogue, not financial content
        - Best for general conversation
        - May produce repetitive responses sometimes
        """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**AppleBee v1.0**")
    st.markdown("Warren Buffett Dashboard")

with footer_col2:
    st.markdown("**Dual Chatbots**")
    st.markdown("🤖 Groq + 🏗️ Transformer")

with footer_col3:
    st.markdown("**Built with**")
    st.markdown("TensorFlow + Streamlit")