"""
AppleBee - Warren Buffett Stock Analysis Dashboard
A Streamlit application for analyzing stocks using Warren Buffett's investment principles
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os
import re
import json

# Try to import yfinance
try:
    import yfinance as yf
    # Fix for Yahoo Finance authentication issue
    yf.set_tz_cache_location("/tmp/yfinance_cache")
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Try to import TensorFlow for chatbot
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Try to import Groq for API chatbot
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Also support requests-based Groq API calls as fallback
import requests

# ============================================================================
# CHATBOT MODULE (Integrated for simplicity)
# ============================================================================

# Model directory - where trained model files should be placed
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

def preprocess_sentence_chatbot(sentence):
    """Clean and preprocess a sentence for chatbot"""
    if not isinstance(sentence, str):
        sentence = str(sentence)
    
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "what is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    sentence = re.sub(r"[^a-zA-Z0-9?.!,%]+", " ", sentence)
    sentence = sentence.strip()
    return sentence


class SimpleTokenizer:
    """A simple word-level tokenizer for the chatbot."""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.oov_token = '<OOV>'
        self.pad_token = '<PAD>'
    
    def encode(self, sentence):
        return [self.word2idx.get(word, 1) for word in sentence.split()]
    
    def decode(self, tokens):
        words = [self.idx2word.get(idx, self.oov_token) for idx in tokens 
                 if idx != 0 and idx in self.idx2word]
        return ' '.join(words)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        tokenizer.vocab_size = data['vocab_size']
        return tokenizer


# Custom layers for model loading (only define if TensorFlow is available)
if TF_AVAILABLE:
    import numpy as np_tf
    
    # Transformer components matching the training script
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(position, d_model):
        angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def scaled_dot_product_attention(q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    class MultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads):
            super(MultiHeadAttention, self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model
            self.depth = d_model // num_heads
            self.wq = tf.keras.layers.Dense(d_model)
            self.wk = tf.keras.layers.Dense(d_model)
            self.wv = tf.keras.layers.Dense(d_model)
            self.dense = tf.keras.layers.Dense(d_model)
        
        def split_heads(self, x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])
        
        def call(self, v, k, q, mask):
            batch_size = tf.shape(q)[0]
            q = self.wq(q)
            k = self.wk(k)
            v = self.wv(v)
            q = self.split_heads(q, batch_size)
            k = self.split_heads(k, batch_size)
            v = self.split_heads(v, batch_size)
            scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
            output = self.dense(concat_attention)
            return output

    def point_wise_feed_forward_network(d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

    class EncoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super(EncoderLayer, self).__init__()
            self.mha = MultiHeadAttention(d_model, num_heads)
            self.ffn = point_wise_feed_forward_network(d_model, dff)
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)
        
        def call(self, x, training=False, mask=None):
            attn_output = self.mha(x, x, x, mask)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(x + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)
            return out2

    class DecoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super(DecoderLayer, self).__init__()
            self.mha1 = MultiHeadAttention(d_model, num_heads)
            self.mha2 = MultiHeadAttention(d_model, num_heads)
            self.ffn = point_wise_feed_forward_network(d_model, dff)
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)
            self.dropout3 = tf.keras.layers.Dropout(rate)
        
        def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
            attn1 = self.mha1(x, x, x, look_ahead_mask)
            attn1 = self.dropout1(attn1, training=training)
            out1 = self.layernorm1(attn1 + x)
            attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)
            attn2 = self.dropout2(attn2, training=training)
            out2 = self.layernorm2(attn2 + out1)
            ffn_output = self.ffn(out2)
            ffn_output = self.dropout3(ffn_output, training=training)
            out3 = self.layernorm3(ffn_output + out2)
            return out3

    class Encoder(tf.keras.layers.Layer):
        def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
            super(Encoder, self).__init__()
            self.d_model = d_model
            self.num_layers = num_layers
            self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
            self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
            self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
            self.dropout = tf.keras.layers.Dropout(rate)
        
        def call(self, x, training=False, mask=None):
            seq_len = tf.shape(x)[1]
            x = self.embedding(x)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]
            x = self.dropout(x, training=training)
            for i in range(self.num_layers):
                x = self.enc_layers[i](x, training=training, mask=mask)
            return x

    class Decoder(tf.keras.layers.Layer):
        def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
            super(Decoder, self).__init__()
            self.d_model = d_model
            self.num_layers = num_layers
            self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
            self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
            self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
            self.dropout = tf.keras.layers.Dropout(rate)
        
        def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
            seq_len = tf.shape(x)[1]
            x = self.embedding(x)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]
            x = self.dropout(x, training=training)
            for i in range(self.num_layers):
                x = self.dec_layers[i](x, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
            return x

    class Transformer(tf.keras.Model):
        def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
            super(Transformer, self).__init__()
            self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
            self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
            self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
        def call(self, inputs, training=False):
            inp, tar = inputs
            enc_padding_mask = create_padding_mask(inp)
            dec_padding_mask = create_padding_mask(inp)
            look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
            dec_target_padding_mask = create_padding_mask(tar)
            combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
            
            enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)
            dec_output = self.decoder(tar, enc_output, training=training, look_ahead_mask=combined_mask, padding_mask=dec_padding_mask)
            final_output = self.final_layer(dec_output)
            return final_output


class BuffettChatbot:
    """Warren Buffett Investment Advisor Chatbot"""
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.config = None
        self.loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        if not TF_AVAILABLE:
            return
        
        try:
            config_path = os.path.join(self.model_dir, "config.json")
            if not os.path.exists(config_path):
                return
            
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
            if not os.path.exists(tokenizer_path):
                return
            
            self.tokenizer = SimpleTokenizer.load(tokenizer_path)
            
            # Check for weights file
            weights_path = os.path.join(self.model_dir, "transformer_weights.weights.h5")
            if not os.path.exists(weights_path):
                return
            
            # Rebuild model architecture from config
            vocab_size = self.config["vocab_size"]
            num_layers = self.config["num_layers"]
            d_model = self.config["d_model"]
            num_heads = self.config["num_heads"]
            dff = self.config["units"]
            dropout = self.config["dropout"]
            max_length = self.config["max_length"]
            
            self.model = Transformer(
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                input_vocab_size=vocab_size,
                target_vocab_size=vocab_size,
                pe_input=vocab_size,
                pe_target=vocab_size,
                rate=dropout
            )
            
            # Build model by calling it once
            sample_input = (tf.zeros((1, max_length), dtype=tf.int32), tf.zeros((1, max_length-1), dtype=tf.int32))
            _ = self.model(sample_input)
            
            # Load weights
            self.model.load_weights(weights_path)
            self.loaded = True
            
        except Exception as e:
            print(f"Error loading chatbot: {e}")
            self.loaded = False
    
    def is_loaded(self):
        return self.loaded
    
    def _evaluate(self, sentence):
        sentence = preprocess_sentence_chatbot(sentence)
        START_TOKEN = self.config["start_token"]
        END_TOKEN = self.config["end_token"]
        MAX_LENGTH = self.config["max_length"]
        
        # Tokenize and pad input
        sentence_tok = [START_TOKEN] + self.tokenizer.encode(sentence) + [END_TOKEN]
        sentence_tok = tf.keras.preprocessing.sequence.pad_sequences([sentence_tok], maxlen=MAX_LENGTH, padding="post")
        encoder_input = tf.cast(sentence_tok, tf.int32)
        
        # Start with START token
        decoder_input = [START_TOKEN]
        output = tf.expand_dims(decoder_input, 0)
        output = tf.cast(output, tf.int32)
        
        for i in range(MAX_LENGTH):
            predictions = self.model((encoder_input, output), training=False)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)
            predicted_id_val = int(predicted_id.numpy()[0][0])
            
            if predicted_id_val == END_TOKEN:
                break
            
            output = tf.concat([output, predicted_id], axis=-1)
        
        return tf.squeeze(output, axis=0)
    
    def chat(self, message):
        if not self.loaded:
            return None
        try:
            prediction = self._evaluate(message)
            response = self.tokenizer.decode(
                [i for i in prediction.numpy() if i < self.tokenizer.vocab_size]
            )
            return response if response else "I'm not sure how to respond to that."
        except Exception as e:
            return f"Error: {str(e)}"


@st.cache_resource
def load_chatbot():
    """Load the chatbot model (cached)"""
    return BuffettChatbot(MODEL_DIR)


# ============================================================================
# GROQ API CHATBOT
# ============================================================================

BUFFETT_SYSTEM_PROMPT = """You are Warren Buffett, the legendary investor and CEO of Berkshire Hathaway. 
You are having a conversation about investing, business, and life wisdom.

Respond in first person as Warren Buffett would, drawing from his well-known investment philosophy:
- Value investing principles
- Focus on intrinsic value and margin of safety
- Long-term holding perspective ("Our favorite holding period is forever")
- Circle of competence
- Quality businesses with durable competitive advantages (moats)
- Importance of management integrity
- Avoiding speculation and market timing
- Being fearful when others are greedy and greedy when others are fearful

Keep responses conversational, wise, and occasionally use folksy humor as Buffett is known for.
Be helpful and educational while staying in character."""

def call_groq_api(message: str, api_key: str, conversation_history: list = None) -> str:
    """Call the Groq API to get a response from the Buffett chatbot"""
    
    if not api_key:
        return "❌ No API key provided. Please enter your Groq API key in the sidebar."
    
    # Aggressively clean the API key - remove ALL whitespace and hidden characters
    import re as re_clean
    api_key = api_key.strip()
    api_key = re_clean.sub(r'\s+', '', api_key)  # Remove all whitespace
    api_key = ''.join(c for c in api_key if c.isprintable() and not c.isspace())  # Only printable non-space chars
    
    # Validate API key format
    if not api_key.startswith("gsk_"):
        return f"❌ Invalid API key format. Groq API keys should start with 'gsk_'. Your key starts with '{api_key[:4]}...' - please check your key at console.groq.com/keys"
    
    if len(api_key) < 20:
        return f"❌ API key seems too short ({len(api_key)} characters). Please check your key."
    
    # Build messages list
    messages = [{"role": "system", "content": BUFFETT_SYSTEM_PROMPT}]
    
    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history[-10:]:  # Keep last 10 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Always use requests for reliability (groq library can have issues)
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Try multiple models
        models_to_try = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768"
        ]
        
        last_error = None
        for model in models_to_try:
            try:
                data = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 401:
                    # Authentication failed - show details
                    error_detail = response.text[:300] if response.text else "No details"
                    last_error = f"401 Auth Error: {error_detail}"
                    continue
                else:
                    last_error = f"Status {response.status_code}: {response.text[:200]}"
                    continue
                    
            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                continue
            except Exception as e:
                last_error = str(e)
                continue
        
        # If we get here, all models failed
        return f"❌ API call failed.\n\n**Last error:** {last_error}\n\n**Debug info:**\n- Key length after cleaning: {len(api_key)}\n- Key: {api_key[:8]}...{api_key[-4:]}\n- Using: requests library"
            
    except Exception as e:
        error_msg = str(e)
        return f"❌ Error: {error_msg}\n\n**Debug info:**\n- Key length: {len(api_key)} chars\n- Key prefix: {api_key[:8]}..."


def is_model_available():
    """Check if the trained model files exist"""
    required_files = ["config.json", "tokenizer.json", "transformer_weights.weights.h5"]
    
    for f in required_files:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            return False
    
    return True

# Page configuration
st.set_page_config(
    page_title="AppleBee - Warren Buffett Stock Analyzer",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-gold: #D4AF37;
        --dark-green: #1B4D3E;
        --cream: #F5F5DC;
        --dark-bg: #0E1117;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #D4AF37;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #D4AF37;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-top: -1.5rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #D4AF37;
        margin-bottom: 1rem;
    }
    
    .metric-pass {
        border-left-color: #00C853 !important;
    }
    
    .metric-fail {
        border-left-color: #FF5252 !important;
    }
    
    .metric-neutral {
        border-left-color: #FFC107 !important;
    }
    
    /* Status badges */
    .status-pass {
        background-color: #00C853;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .status-fail {
        background-color: #FF5252;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .status-neutral {
        background-color: #FFC107;
        color: black;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #D4AF37;
        border-bottom: 2px solid #333;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Info boxes */
    .buffett-quote {
        background: linear-gradient(135deg, #1B4D3E 0%, #0d2818 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #D4AF37;
        font-style: italic;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Warren Buffett's ratio thresholds and explanations
BUFFETT_RATIOS = {
    "income_statement": {
        "gross_margin": {
            "name": "Gross Margin",
            "threshold": 0.40,
            "rule": "≥ 40%",
            "comparison": ">=",
            "logic": "Signals the company isn't competing on price. High margins indicate a durable competitive advantage (moat)."
        },
        "sga_margin": {
            "name": "SG&A Expense Margin",
            "threshold": 0.30,
            "rule": "≤ 30%",
            "comparison": "<=",
            "logic": "Wide-moat companies don't need to spend a lot on overhead to operate. Low SG&A relative to gross profit indicates efficiency."
        },
        "rd_margin": {
            "name": "R&D Expense Margin",
            "threshold": 0.30,
            "rule": "≤ 30%",
            "comparison": "<=",
            "logic": "R&D expenses don't always create value for shareholders. Companies overly dependent on R&D may lack sustainable advantages."
        },
        "depreciation_margin": {
            "name": "Depreciation Margin",
            "threshold": 0.10,
            "rule": "≤ 10%",
            "comparison": "<=",
            "logic": "Buffett doesn't like businesses that need to invest heavily in depreciating assets to maintain their competitive advantage."
        },
        "interest_expense_margin": {
            "name": "Interest Expense Margin",
            "threshold": 0.15,
            "rule": "≤ 15%",
            "comparison": "<=",
            "logic": "Great businesses don't need debt to finance themselves. Low interest expense indicates financial strength."
        },
        "tax_rate": {
            "name": "Effective Tax Rate",
            "threshold": 0.21,
            "rule": "~21% (Current Corporate Rate)",
            "comparison": "info",
            "logic": "Great businesses are so profitable that they are forced to pay their full tax load. Very low rates may indicate accounting tricks."
        },
        "net_margin": {
            "name": "Net Margin",
            "threshold": 0.20,
            "rule": "≥ 20%",
            "comparison": ">=",
            "logic": "Great companies convert 20% or more of their revenue into net income. This indicates exceptional business quality."
        },
        "eps_growth": {
            "name": "EPS Growth",
            "threshold": 0,
            "rule": "Positive & Growing",
            "comparison": ">",
            "logic": "Great companies increase profits every year. Consistent EPS growth indicates a sustainable competitive advantage."
        }
    },
    "balance_sheet": {
        "debt_to_equity": {
            "name": "Debt to Equity Ratio",
            "threshold": 0.80,
            "rule": "≤ 80%",
            "comparison": "<=",
            "logic": "Conservative leverage indicates financial stability. Companies with low debt can weather economic storms better."
        },
        "cash_to_debt": {
            "name": "Cash to Debt Ratio",
            "threshold": 1.0,
            "rule": "≥ 1.0x",
            "comparison": ">=",
            "logic": "Companies with more cash than debt have strong financial positions and flexibility for opportunities."
        },
        "retained_earnings_growth": {
            "name": "Retained Earnings Growth",
            "threshold": 0,
            "rule": "Positive & Growing",
            "comparison": ">",
            "logic": "Growing retained earnings indicate the company is reinvesting profits effectively and building shareholder value."
        },
        "preferred_stock_ratio": {
            "name": "Preferred Stock to Equity",
            "threshold": 0.05,
            "rule": "≤ 5% or None",
            "comparison": "<=",
            "logic": "Buffett prefers simple capital structures. Preferred stock is a prior claim on earnings before common stockholders, diluting shareholder value."
        },
        "treasury_stock_ratio": {
            "name": "Treasury Stock (Buybacks)",
            "threshold": 0,
            "rule": "Presence indicates buybacks",
            "comparison": "info",
            "logic": "Treasury stock represents shares the company has repurchased. Buffett approves of buybacks when done at sensible prices—it increases per-share intrinsic value."
        }
    },
    "cash_flow": {
        "capex_margin": {
            "name": "CapEx to Net Income",
            "threshold": 0.50,
            "rule": "≤ 50%",
            "comparison": "<=",
            "logic": "Companies that don't need heavy capital expenditures to maintain their business generate more free cash flow for shareholders."
        }
    }
}


# Sample data for demonstration (based on actual Apple financials)
SAMPLE_DATA = {
    "AAPL": {
        "info": {
            "longName": "Apple Inc.",
            "symbol": "AAPL",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "currentPrice": 237.33,
            "marketCap": 3580000000000,
            "currency": "USD"
        },
        "income_stmt": pd.DataFrame({
            pd.Timestamp("2024-09-28"): {
                "Total Revenue": 391035000000,
                "Gross Profit": 180683000000,
                "Selling General And Administration": 26097000000,
                "Research And Development": 31370000000,
                "Reconciled Depreciation": 11445000000,
                "Operating Income": 123216000000,
                "Interest Expense": 0,
                "Pretax Income": 123485000000,
                "Tax Provision": 29749000000,
                "Net Income": 93736000000,
                "Basic EPS": 6.11
            },
            pd.Timestamp("2023-09-30"): {
                "Total Revenue": 383285000000,
                "Gross Profit": 169148000000,
                "Selling General And Administration": 24932000000,
                "Research And Development": 29915000000,
                "Reconciled Depreciation": 11519000000,
                "Operating Income": 114301000000,
                "Interest Expense": 3933000000,
                "Pretax Income": 113736000000,
                "Tax Provision": 16741000000,
                "Net Income": 96995000000,
                "Basic EPS": 6.16
            },
            pd.Timestamp("2022-09-24"): {
                "Total Revenue": 394328000000,
                "Gross Profit": 170782000000,
                "Selling General And Administration": 25094000000,
                "Research And Development": 26251000000,
                "Reconciled Depreciation": 11104000000,
                "Operating Income": 119437000000,
                "Interest Expense": 2931000000,
                "Pretax Income": 119103000000,
                "Tax Provision": 19300000000,
                "Net Income": 99803000000,
                "Basic EPS": 6.15
            },
            pd.Timestamp("2021-09-25"): {
                "Total Revenue": 365817000000,
                "Gross Profit": 152836000000,
                "Selling General And Administration": 21973000000,
                "Research And Development": 21914000000,
                "Reconciled Depreciation": 11284000000,
                "Operating Income": 108949000000,
                "Interest Expense": 2645000000,
                "Pretax Income": 109207000000,
                "Tax Provision": 14527000000,
                "Net Income": 94680000000,
                "Basic EPS": 5.67
            }
        }).T.T,  # Transpose to get correct orientation
        "balance_sheet": pd.DataFrame({
            pd.Timestamp("2024-09-28"): {
                "Total Debt": 96837000000,
                "Total Equity Gross Minority Interest": 56950000000,
                "Cash And Cash Equivalents": 29943000000,
                "Cash Cash Equivalents And Short Term Investments": 65171000000,
                "Retained Earnings": -19154000000,
                "Total Assets": 364980000000,
                "Preferred Stock": 0,
                "Treasury Stock": -174239000000
            },
            pd.Timestamp("2023-09-30"): {
                "Total Debt": 111088000000,
                "Total Equity Gross Minority Interest": 62146000000,
                "Cash And Cash Equivalents": 29965000000,
                "Cash Cash Equivalents And Short Term Investments": 61555000000,
                "Retained Earnings": -214000000,
                "Total Assets": 352583000000,
                "Preferred Stock": 0,
                "Treasury Stock": -158421000000
            },
            pd.Timestamp("2022-09-24"): {
                "Total Debt": 120069000000,
                "Total Equity Gross Minority Interest": 50672000000,
                "Cash And Cash Equivalents": 23646000000,
                "Cash Cash Equivalents And Short Term Investments": 48304000000,
                "Retained Earnings": -3068000000,
                "Total Assets": 352755000000,
                "Preferred Stock": 0,
                "Treasury Stock": -149680000000
            }
        }).T.T,
        "cash_flow": pd.DataFrame({
            pd.Timestamp("2024-09-28"): {
                "Operating Cash Flow": 118254000000,
                "Capital Expenditure": -9447000000,
                "Net Income": 93736000000
            },
            pd.Timestamp("2023-09-30"): {
                "Operating Cash Flow": 110543000000,
                "Capital Expenditure": -10959000000,
                "Net Income": 96995000000
            },
            pd.Timestamp("2022-09-24"): {
                "Operating Cash Flow": 122151000000,
                "Capital Expenditure": -10708000000,
                "Net Income": 99803000000
            }
        }).T.T
    },
    "MSFT": {
        "info": {
            "longName": "Microsoft Corporation",
            "symbol": "MSFT",
            "sector": "Technology",
            "industry": "Software—Infrastructure",
            "currentPrice": 423.46,
            "marketCap": 3150000000000,
            "currency": "USD"
        },
        "income_stmt": pd.DataFrame({
            pd.Timestamp("2024-06-30"): {
                "Total Revenue": 245122000000,
                "Gross Profit": 171006000000,
                "Selling General And Administration": 28054000000,
                "Research And Development": 29510000000,
                "Reconciled Depreciation": 22287000000,
                "Operating Income": 109433000000,
                "Interest Expense": 2935000000,
                "Pretax Income": 110654000000,
                "Tax Provision": 22033000000,
                "Net Income": 88136000000,
                "Basic EPS": 11.86
            },
            pd.Timestamp("2023-06-30"): {
                "Total Revenue": 211915000000,
                "Gross Profit": 146052000000,
                "Selling General And Administration": 24506000000,
                "Research And Development": 27195000000,
                "Reconciled Depreciation": 13861000000,
                "Operating Income": 88523000000,
                "Interest Expense": 1968000000,
                "Pretax Income": 89694000000,
                "Tax Provision": 16950000000,
                "Net Income": 72361000000,
                "Basic EPS": 9.72
            }
        }).T.T,
        "balance_sheet": pd.DataFrame({
            pd.Timestamp("2024-06-30"): {
                "Total Debt": 72394000000,
                "Total Equity Gross Minority Interest": 268477000000,
                "Cash And Cash Equivalents": 18315000000,
                "Cash Cash Equivalents And Short Term Investments": 75530000000,
                "Retained Earnings": 173144000000,
                "Total Assets": 512163000000,
                "Preferred Stock": 0,
                "Treasury Stock": -66486000000
            },
            pd.Timestamp("2023-06-30"): {
                "Total Debt": 59965000000,
                "Total Equity Gross Minority Interest": 206223000000,
                "Cash And Cash Equivalents": 34704000000,
                "Cash Cash Equivalents And Short Term Investments": 111262000000,
                "Retained Earnings": 118848000000,
                "Total Assets": 411976000000,
                "Preferred Stock": 0,
                "Treasury Stock": -61814000000
            }
        }).T.T,
        "cash_flow": pd.DataFrame({
            pd.Timestamp("2024-06-30"): {
                "Operating Cash Flow": 118548000000,
                "Capital Expenditure": -44477000000,
                "Net Income": 88136000000
            },
            pd.Timestamp("2023-06-30"): {
                "Operating Cash Flow": 87582000000,
                "Capital Expenditure": -28107000000,
                "Net Income": 72361000000
            }
        }).T.T
    },
    "BRK-B": {
        "info": {
            "longName": "Berkshire Hathaway Inc.",
            "symbol": "BRK-B",
            "sector": "Financial Services",
            "industry": "Insurance—Diversified",
            "currentPrice": 472.83,
            "marketCap": 1030000000000,
            "currency": "USD"
        },
        "income_stmt": pd.DataFrame({
            pd.Timestamp("2024-09-30"): {
                "Total Revenue": 371900000000,
                "Gross Profit": 93000000000,
                "Selling General And Administration": 11200000000,
                "Research And Development": 0,
                "Reconciled Depreciation": 12800000000,
                "Operating Income": 42800000000,
                "Interest Expense": 2100000000,
                "Pretax Income": 89600000000,
                "Tax Provision": 11700000000,
                "Net Income": 89497000000,
                "Basic EPS": 41.32
            },
            pd.Timestamp("2023-09-30"): {
                "Total Revenue": 364482000000,
                "Gross Profit": 88500000000,
                "Selling General And Administration": 10800000000,
                "Research And Development": 0,
                "Reconciled Depreciation": 12200000000,
                "Operating Income": 37353000000,
                "Interest Expense": 1800000000,
                "Pretax Income": 46527000000,
                "Tax Provision": 10500000000,
                "Net Income": 96223000000,
                "Basic EPS": 44.02
            }
        }).T.T,
        "balance_sheet": pd.DataFrame({
            pd.Timestamp("2024-09-30"): {
                "Total Debt": 125800000000,
                "Total Equity Gross Minority Interest": 625100000000,
                "Cash And Cash Equivalents": 325200000000,
                "Cash Cash Equivalents And Short Term Investments": 325200000000,
                "Retained Earnings": 612400000000,
                "Total Assets": 1146000000000,
                "Preferred Stock": 0,
                "Treasury Stock": -52195000000
            },
            pd.Timestamp("2023-09-30"): {
                "Total Debt": 127100000000,
                "Total Equity Gross Minority Interest": 561400000000,
                "Cash And Cash Equivalents": 157200000000,
                "Cash Cash Equivalents And Short Term Investments": 157200000000,
                "Retained Earnings": 548900000000,
                "Total Assets": 1005000000000,
                "Preferred Stock": 0,
                "Treasury Stock": -43812000000
            }
        }).T.T,
        "cash_flow": pd.DataFrame({
            pd.Timestamp("2024-09-30"): {
                "Operating Cash Flow": 49800000000,
                "Capital Expenditure": -19200000000,
                "Net Income": 89497000000
            },
            pd.Timestamp("2023-09-30"): {
                "Operating Cash Flow": 43200000000,
                "Capital Expenditure": -16800000000,
                "Net Income": 96223000000
            }
        }).T.T
    }
}


def generate_sample_history(days=730):
    """Generate sample stock price history"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate random walk for price
    returns = np.random.normal(0.0005, 0.02, days)
    price = 150 * np.cumprod(1 + returns)
    
    # Generate OHLC data
    data = {
        'Open': price * (1 + np.random.uniform(-0.01, 0.01, days)),
        'High': price * (1 + np.random.uniform(0, 0.02, days)),
        'Low': price * (1 - np.random.uniform(0, 0.02, days)),
        'Close': price,
        'Volume': np.random.randint(50000000, 150000000, days)
    }
    
    return pd.DataFrame(data, index=dates)


def get_stock_data(symbol: str) -> dict:
    """Fetch comprehensive stock data using yfinance or sample data"""
    
    # First, try to use sample data if available
    symbol_upper = symbol.upper()
    if symbol_upper in SAMPLE_DATA:
        sample = SAMPLE_DATA[symbol_upper]
        return {
            "info": sample["info"],
            "income_stmt": sample["income_stmt"],
            "balance_sheet": sample["balance_sheet"],
            "cash_flow": sample["cash_flow"],
            "history": generate_sample_history(),
            "success": True,
            "is_sample": True
        }
    
    # Try yfinance if available
    if YFINANCE_AVAILABLE:
        try:
            stock = yf.Ticker(symbol)
            
            # Get financial statements with error handling
            try:
                income_stmt = stock.income_stmt
            except Exception:
                income_stmt = None
            
            try:
                balance_sheet = stock.balance_sheet
            except Exception:
                balance_sheet = None
            
            try:
                cash_flow = stock.cashflow
            except Exception:
                cash_flow = None
            
            # Get basic info with error handling
            try:
                info = stock.info
                if not info or len(info) < 5:  # Empty or minimal info dict
                    raise ValueError("No valid info returned")
            except Exception:
                info = {"longName": symbol, "symbol": symbol}
            
            # Get historical data for price chart
            try:
                history = stock.history(period="2y")
            except Exception:
                history = generate_sample_history()
            
            # Check if we got valid data
            if income_stmt is not None and not income_stmt.empty:
                return {
                    "info": info,
                    "income_stmt": income_stmt,
                    "balance_sheet": balance_sheet,
                    "cash_flow": cash_flow,
                    "history": history,
                    "success": True,
                    "is_sample": False
                }
            else:
                # Yahoo Finance API might be having issues
                return {
                    "success": False,
                    "error": f"Could not fetch financial data for {symbol}. Yahoo Finance API may be temporarily unavailable. Try AAPL, MSFT, or BRK-B for sample data."
                }
                
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg or "Crumb" in error_msg:
                return {
                    "success": False, 
                    "error": f"Yahoo Finance authentication error. This is a known issue with their API. Try AAPL, MSFT, or BRK-B for sample data, or try again later."
                }
            return {
                "success": False, 
                "error": f"Error fetching {symbol}: {error_msg}. Try AAPL, MSFT, or BRK-B for sample data."
            }
    
    # Return error if no data available
    return {
        "success": False, 
        "error": f"No data available for {symbol}. Try AAPL, MSFT, or BRK-B for sample data."
    }


def safe_get(df: pd.DataFrame, keys: list, column_idx: int = 0):
    """Safely get a value from a DataFrame with multiple possible key names"""
    if df is None or df.empty:
        return None
    
    for key in keys:
        if key in df.index:
            try:
                value = df.loc[key].iloc[column_idx]
                if pd.notna(value):
                    return float(value)
            except:
                continue
    return None


def calculate_buffett_ratios(data: dict) -> dict:
    """Calculate all Warren Buffett ratios from financial data"""
    income_stmt = data.get("income_stmt")
    balance_sheet = data.get("balance_sheet")
    cash_flow = data.get("cash_flow")
    
    ratios = {
        "income_statement": {},
        "balance_sheet": {},
        "cash_flow": {}
    }
    
    # ===== INCOME STATEMENT RATIOS =====
    if income_stmt is not None and not income_stmt.empty:
        # Get key values
        revenue = safe_get(income_stmt, ["Total Revenue", "Revenue"])
        gross_profit = safe_get(income_stmt, ["Gross Profit"])
        sga = safe_get(income_stmt, ["Selling General And Administration", "SG&A", "Selling And Marketing Expense"])
        rd = safe_get(income_stmt, ["Research And Development", "R&D"])
        depreciation = safe_get(income_stmt, ["Reconciled Depreciation", "Depreciation", "Depreciation And Amortization"])
        operating_income = safe_get(income_stmt, ["Operating Income", "EBIT"])
        interest_expense = safe_get(income_stmt, ["Interest Expense", "Interest Expense Non Operating"])
        pretax_income = safe_get(income_stmt, ["Pretax Income", "Income Before Tax"])
        tax_provision = safe_get(income_stmt, ["Tax Provision", "Income Tax Expense"])
        net_income = safe_get(income_stmt, ["Net Income", "Net Income Common Stockholders"])
        
        # Basic EPS for current and previous year
        eps_current = safe_get(income_stmt, ["Basic EPS", "Diluted EPS"], 0)
        eps_previous = safe_get(income_stmt, ["Basic EPS", "Diluted EPS"], 1)
        
        # Calculate ratios
        # 1. Gross Margin
        if revenue and gross_profit:
            ratios["income_statement"]["gross_margin"] = gross_profit / revenue
        
        # 2. SG&A Margin
        if gross_profit and sga:
            ratios["income_statement"]["sga_margin"] = sga / gross_profit
        elif gross_profit:
            ratios["income_statement"]["sga_margin"] = 0  # No SG&A reported
        
        # 3. R&D Margin
        if gross_profit and rd:
            ratios["income_statement"]["rd_margin"] = rd / gross_profit
        elif gross_profit:
            ratios["income_statement"]["rd_margin"] = 0  # No R&D reported
        
        # 4. Depreciation Margin
        if gross_profit and depreciation:
            ratios["income_statement"]["depreciation_margin"] = depreciation / gross_profit
        
        # 5. Interest Expense Margin
        if operating_income and interest_expense:
            ratios["income_statement"]["interest_expense_margin"] = abs(interest_expense) / operating_income
        elif operating_income:
            ratios["income_statement"]["interest_expense_margin"] = 0  # No interest expense
        
        # 6. Effective Tax Rate
        if pretax_income and tax_provision and pretax_income > 0:
            ratios["income_statement"]["tax_rate"] = tax_provision / pretax_income
        
        # 7. Net Margin
        if revenue and net_income:
            ratios["income_statement"]["net_margin"] = net_income / revenue
        
        # 8. EPS Growth
        if eps_current and eps_previous and eps_previous != 0:
            ratios["income_statement"]["eps_growth"] = (eps_current - eps_previous) / abs(eps_previous)
    
    # ===== BALANCE SHEET RATIOS =====
    if balance_sheet is not None and not balance_sheet.empty:
        # Get key values
        total_debt = safe_get(balance_sheet, ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"])
        total_equity = safe_get(balance_sheet, ["Total Equity Gross Minority Interest", "Stockholders Equity", "Total Stockholder Equity"])
        cash = safe_get(balance_sheet, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"])
        retained_earnings_current = safe_get(balance_sheet, ["Retained Earnings"], 0)
        retained_earnings_previous = safe_get(balance_sheet, ["Retained Earnings"], 1)
        
        # Calculate ratios
        # 1. Debt to Equity
        if total_equity and total_debt and total_equity != 0:
            ratios["balance_sheet"]["debt_to_equity"] = total_debt / total_equity
        
        # 2. Cash to Debt
        if cash and total_debt and total_debt != 0:
            ratios["balance_sheet"]["cash_to_debt"] = cash / total_debt
        elif cash and (not total_debt or total_debt == 0):
            ratios["balance_sheet"]["cash_to_debt"] = float('inf')  # No debt is great!
        
        # 3. Retained Earnings Growth
        if retained_earnings_current and retained_earnings_previous and retained_earnings_previous != 0:
            ratios["balance_sheet"]["retained_earnings_growth"] = (retained_earnings_current - retained_earnings_previous) / abs(retained_earnings_previous)
        
        # 4. Preferred Stock Ratio
        preferred_stock = safe_get(balance_sheet, ["Preferred Stock", "Preferred Securities Outside Stock Equity", "Redeemable Preferred Stock"])
        if total_equity and total_equity != 0:
            if preferred_stock:
                ratios["balance_sheet"]["preferred_stock_ratio"] = abs(preferred_stock) / total_equity
            else:
                ratios["balance_sheet"]["preferred_stock_ratio"] = 0  # No preferred stock - ideal
        
        # 5. Treasury Stock (Share Buybacks)
        treasury_stock = safe_get(balance_sheet, ["Treasury Stock", "Treasury Shares Number"])
        if treasury_stock is not None:
            # Treasury stock is typically negative on balance sheet
            ratios["balance_sheet"]["treasury_stock_ratio"] = abs(treasury_stock) if treasury_stock else 0
        else:
            ratios["balance_sheet"]["treasury_stock_ratio"] = 0  # No buybacks recorded
    
    # ===== CASH FLOW RATIOS =====
    if cash_flow is not None and not cash_flow.empty:
        # Get key values
        capex = safe_get(cash_flow, ["Capital Expenditure", "Capital Expenditures"])
        operating_cash_flow = safe_get(cash_flow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        net_income_cf = safe_get(cash_flow, ["Net Income", "Net Income From Continuing Operations"])
        
        # Calculate CapEx Margin
        if capex and net_income_cf and net_income_cf != 0:
            ratios["cash_flow"]["capex_margin"] = abs(capex) / net_income_cf
    
    return ratios


def evaluate_ratio(value: float, threshold: float, comparison: str) -> str:
    """Evaluate if a ratio passes Buffett's criteria"""
    if value is None:
        return "neutral"
    
    if comparison == ">=":
        return "pass" if value >= threshold else "fail"
    elif comparison == "<=":
        return "pass" if value <= threshold else "fail"
    elif comparison == ">":
        return "pass" if value > threshold else "fail"
    elif comparison == "<":
        return "pass" if value < threshold else "fail"
    else:  # info
        return "neutral"


def format_percentage(value: float) -> str:
    """Format a decimal as percentage"""
    if value is None:
        return "N/A"
    if value == float('inf'):
        return "∞ (No Debt)"
    return f"{value * 100:.2f}%"


def format_ratio(value: float) -> str:
    """Format a ratio value"""
    if value is None:
        return "N/A"
    if value == float('inf'):
        return "∞ (No Debt)"
    return f"{value:.2f}x"


def format_currency(value: float) -> str:
    """Format a large number as currency (in billions)"""
    if value is None:
        return "N/A"
    if value == 0:
        return "$0 (No Buybacks)"
    if abs(value) >= 1e9:
        return f"${abs(value) / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${abs(value) / 1e6:.2f}M"
    else:
        return f"${abs(value):,.0f}"


def display_ratio_card(name: str, value: float, rule: str, logic: str, comparison: str, threshold: float, is_percentage: bool = True, format_type: str = None):
    """Display a single ratio as a styled card
    
    Args:
        format_type: 'percentage', 'ratio', or 'currency' (overrides is_percentage if provided)
    """
    status = evaluate_ratio(value, threshold, comparison)
    
    # Determine formatting
    if format_type == "currency":
        formatted_value = format_currency(value)
    elif format_type == "ratio" or not is_percentage:
        formatted_value = format_ratio(value)
    else:
        formatted_value = format_percentage(value)
    
    status_class = f"status-{status}"
    metric_class = f"metric-{status}"
    
    status_text = "✓ PASS" if status == "pass" else ("✗ FAIL" if status == "fail" else "ℹ INFO")
    
    st.markdown(f"""
    <div class="metric-card {metric_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h4 style="margin: 0; color: #D4AF37;">{name}</h4>
            <span class="{status_class}">{status_text}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.75rem;">
            <span style="font-size: 2rem; font-weight: bold; color: white;">{formatted_value}</span>
            <span style="color: #888;">Buffett's Rule: {rule}</span>
        </div>
        <p style="color: #aaa; font-size: 0.9rem; margin: 0;">{logic}</p>
    </div>
    """, unsafe_allow_html=True)


def create_gauge_chart(value: float, threshold: float, title: str, comparison: str):
    """Create a gauge chart for a ratio"""
    if value is None:
        return None
    
    # Determine color based on pass/fail
    status = evaluate_ratio(value, threshold, comparison)
    color = "#00C853" if status == "pass" else "#FF5252"
    
    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'size': 24}},
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold * 100], 'color': 'rgba(0, 200, 83, 0.3)' if comparison == "<=" else 'rgba(255, 82, 82, 0.3)'},
                {'range': [threshold * 100, 100], 'color': 'rgba(255, 82, 82, 0.3)' if comparison == "<=" else 'rgba(0, 200, 83, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#D4AF37", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig


def calculate_buffett_score(ratios: dict) -> tuple:
    """Calculate overall Buffett score"""
    total = 0
    passed = 0
    
    for category, metrics in BUFFETT_RATIOS.items():
        for ratio_key, ratio_info in metrics.items():
            if ratio_info["comparison"] != "info":
                total += 1
                if category in ratios and ratio_key in ratios[category]:
                    value = ratios[category][ratio_key]
                    if evaluate_ratio(value, ratio_info["threshold"], ratio_info["comparison"]) == "pass":
                        passed += 1
    
    return passed, total


def display_financial_statement(df: pd.DataFrame, title: str):
    """Display a financial statement DataFrame"""
    if df is None or df.empty:
        st.warning(f"No {title} data available")
        return
    
    # Format the DataFrame
    df_display = df.copy()
    
    # Convert column names to readable dates
    if isinstance(df_display.columns[0], (pd.Timestamp, datetime)):
        df_display.columns = [col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col) for col in df_display.columns]
    
    # Format numbers
    def format_value(x):
        if pd.notna(x) and isinstance(x, (int, float)):
            if abs(x) >= 1e9:
                return f"${x/1e9:.2f}B"
            elif abs(x) >= 1e6:
                return f"${x/1e6:.2f}M"
            else:
                return f"${x:,.0f}"
        return x
    
    df_display = df_display.map(format_value)
    
    st.dataframe(df_display, use_container_width=True, height=400)


# ===== MAIN APPLICATION =====
def main():
    # Header
    st.markdown('<h1 class="main-header">🐝 AppleBee</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Warren Buffett Stock Analysis Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Warren_Buffett_KU_Visit.jpg/220px-Warren_Buffett_KU_Visit.jpg", width=150)
        st.markdown("### 📊 Stock Selection")
        
        symbol = st.text_input(
            "Enter Stock Symbol",
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        analyze_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        **AppleBee** analyzes stocks using Warren Buffett's investment principles.
        
        The dashboard evaluates:
        - 📈 Income Statement Ratios
        - 📊 Balance Sheet Ratios  
        - 💰 Cash Flow Ratios
        
        All metrics are compared against Buffett's proven thresholds.
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Buffett's Wisdom")
        st.markdown("""
        > *"Rule No. 1: Never lose money. Rule No. 2: Never forget Rule No. 1."*
        
        > *"It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price."*
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Groq Chatbot", "🎩 Custom Chatbot", "📚 Learn"])
    
    # Initialize session state
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = None
    if 'quick_symbol' not in st.session_state:
        st.session_state.quick_symbol = None
    if 'groq_messages' not in st.session_state:
        st.session_state.groq_messages = []
    if 'custom_messages' not in st.session_state:
        st.session_state.custom_messages = []
    
    # Handle quick symbol selection
    if st.session_state.quick_symbol:
        symbol = st.session_state.quick_symbol
        st.session_state.quick_symbol = None  # Reset
    
    # Fetch data if button clicked or symbol changed
    if analyze_button or (symbol and symbol != st.session_state.current_symbol):
        with st.spinner(f"Fetching data for {symbol}..."):
            data = get_stock_data(symbol)
            if data["success"]:
                st.session_state.stock_data = data
                st.session_state.current_symbol = symbol
            else:
                st.error(f"⚠️ {data.get('error', 'Unknown error')}")
                st.info("💡 **Tip:** If Yahoo Finance is having issues, try: `pip install --upgrade yfinance`")
                st.session_state.stock_data = None
    
    # ===== TAB 1: DASHBOARD =====
    with tab1:
        if st.session_state.stock_data:
            data = st.session_state.stock_data
            info = data.get("info", {})
            
            # Company header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                company_name = info.get("longName", symbol)
                st.markdown(f"## {company_name} ({symbol})")
                sector = info.get("sector", "N/A")
                industry = info.get("industry", "N/A")
                st.markdown(f"**Sector:** {sector} | **Industry:** {industry}")
                
                # Show sample data indicator
                if data.get("is_sample", False):
                    st.info("📊 **Sample Data Mode**: Using pre-loaded financial data for demonstration. For live data, run the app locally with `streamlit run app.py`.")
            
            with col2:
                current_price = info.get("currentPrice", info.get("regularMarketPrice", "N/A"))
                if isinstance(current_price, (int, float)):
                    st.metric("Current Price", f"${current_price:.2f}")
                else:
                    st.metric("Current Price", "N/A")
            
            with col3:
                market_cap = info.get("marketCap", 0)
                if market_cap >= 1e12:
                    st.metric("Market Cap", f"${market_cap/1e12:.2f}T")
                elif market_cap >= 1e9:
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", f"${market_cap/1e6:.2f}M")
            
            st.markdown("---")
            
            # Calculate ratios
            ratios = calculate_buffett_ratios(data)
            passed, total = calculate_buffett_score(ratios)
            
            # Buffett Score Summary
            st.markdown('<h3 class="section-header">🎯 Warren Buffett Score</h3>', unsafe_allow_html=True)
            
            score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
            
            with score_col2:
                score_pct = (passed / total * 100) if total > 0 else 0
                score_color = "#00C853" if score_pct >= 70 else ("#FFC107" if score_pct >= 50 else "#FF5252")
                
                # Create score gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_pct,
                    number={'suffix': '%', 'font': {'size': 48, 'color': 'white'}},
                    title={'text': f"Buffett Criteria Passed: {passed}/{total}", 'font': {'size': 18, 'color': '#888'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#888'},
                        'bar': {'color': score_color},
                        'bgcolor': 'rgba(255,255,255,0.1)',
                        'borderwidth': 2,
                        'bordercolor': '#333',
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(255, 82, 82, 0.2)'},
                            {'range': [50, 70], 'color': 'rgba(255, 193, 7, 0.2)'},
                            {'range': [70, 100], 'color': 'rgba(0, 200, 83, 0.2)'}
                        ],
                    }
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=30, r=30, t=50, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            if score_pct >= 70:
                st.success(f"🌟 **Strong Buffett Candidate!** {company_name} passes {passed} out of {total} Warren Buffett criteria. This stock shows characteristics of a quality business with durable competitive advantages.")
            elif score_pct >= 50:
                st.warning(f"⚠️ **Mixed Results.** {company_name} passes {passed} out of {total} criteria. Some aspects align with Buffett's principles, but there are areas of concern.")
            else:
                st.error(f"❌ **Does Not Meet Criteria.** {company_name} only passes {passed} out of {total} criteria. This stock may not align well with Buffett's investment philosophy.")
            
            st.markdown("---")
            
            # ===== INCOME STATEMENT SECTION =====
            st.markdown('<h3 class="section-header">📈 Income Statement Analysis</h3>', unsafe_allow_html=True)
            
            income_ratios = ratios.get("income_statement", {})
            
            # Display income statement ratios in a grid
            col1, col2 = st.columns(2)
            
            with col1:
                for i, (key, info) in enumerate(BUFFETT_RATIOS["income_statement"].items()):
                    if i % 2 == 0:
                        value = income_ratios.get(key)
                        display_ratio_card(
                            info["name"],
                            value,
                            info["rule"],
                            info["logic"],
                            info["comparison"],
                            info["threshold"]
                        )
            
            with col2:
                for i, (key, info) in enumerate(BUFFETT_RATIOS["income_statement"].items()):
                    if i % 2 == 1:
                        value = income_ratios.get(key)
                        display_ratio_card(
                            info["name"],
                            value,
                            info["rule"],
                            info["logic"],
                            info["comparison"],
                            info["threshold"]
                        )
            
            # Show raw income statement
            with st.expander("📄 View Full Income Statement"):
                display_financial_statement(data.get("income_stmt"), "Income Statement")
            
            st.markdown("---")
            
            # ===== BALANCE SHEET SECTION =====
            st.markdown('<h3 class="section-header">📊 Balance Sheet Analysis</h3>', unsafe_allow_html=True)
            
            balance_ratios = ratios.get("balance_sheet", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                key = "debt_to_equity"
                info = BUFFETT_RATIOS["balance_sheet"][key]
                value = balance_ratios.get(key)
                display_ratio_card(
                    info["name"],
                    value,
                    info["rule"],
                    info["logic"],
                    info["comparison"],
                    info["threshold"],
                    is_percentage=False
                )
            
            with col2:
                key = "cash_to_debt"
                info = BUFFETT_RATIOS["balance_sheet"][key]
                value = balance_ratios.get(key)
                display_ratio_card(
                    info["name"],
                    value,
                    info["rule"],
                    info["logic"],
                    info["comparison"],
                    info["threshold"],
                    is_percentage=False
                )
            
            with col3:
                key = "retained_earnings_growth"
                info = BUFFETT_RATIOS["balance_sheet"][key]
                value = balance_ratios.get(key)
                display_ratio_card(
                    info["name"],
                    value,
                    info["rule"],
                    info["logic"],
                    info["comparison"],
                    info["threshold"]
                )
            
            # Second row for Preferred Stock and Treasury Stock
            col1, col2, col3 = st.columns(3)
            
            with col1:
                key = "preferred_stock_ratio"
                info = BUFFETT_RATIOS["balance_sheet"][key]
                value = balance_ratios.get(key)
                display_ratio_card(
                    info["name"],
                    value,
                    info["rule"],
                    info["logic"],
                    info["comparison"],
                    info["threshold"]
                )
            
            with col2:
                key = "treasury_stock_ratio"
                info = BUFFETT_RATIOS["balance_sheet"][key]
                value = balance_ratios.get(key)
                display_ratio_card(
                    info["name"],
                    value,
                    info["rule"],
                    info["logic"],
                    info["comparison"],
                    info["threshold"],
                    format_type="currency"
                )
            
            # Show raw balance sheet
            with st.expander("📄 View Full Balance Sheet"):
                display_financial_statement(data.get("balance_sheet"), "Balance Sheet")
            
            st.markdown("---")
            
            # ===== CASH FLOW SECTION =====
            st.markdown('<h3 class="section-header">💰 Cash Flow Analysis</h3>', unsafe_allow_html=True)
            
            cash_ratios = ratios.get("cash_flow", {})
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                key = "capex_margin"
                info = BUFFETT_RATIOS["cash_flow"][key]
                value = cash_ratios.get(key)
                display_ratio_card(
                    info["name"],
                    value,
                    info["rule"],
                    info["logic"],
                    info["comparison"],
                    info["threshold"]
                )
            
            # Show raw cash flow statement
            with st.expander("📄 View Full Cash Flow Statement"):
                display_financial_statement(data.get("cash_flow"), "Cash Flow Statement")
            
            st.markdown("---")
            
            # ===== STOCK PRICE CHART =====
            st.markdown('<h3 class="section-header">📈 Stock Price History</h3>', unsafe_allow_html=True)
            
            history = data.get("history")
            if history is not None and not history.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=history.index,
                    open=history['Open'],
                    high=history['High'],
                    low=history['Low'],
                    close=history['Close'],
                    name='Price'
                ))
                fig.update_layout(
                    title=f'{symbol} Stock Price (2 Years)',
                    yaxis_title='Price ($)',
                    xaxis_title='Date',
                    template='plotly_dark',
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Welcome screen
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h2>Welcome to AppleBee! 🐝</h2>
                <p style="font-size: 1.2rem; color: #888;">
                    Enter a stock symbol in the sidebar and click "Analyze Stock" to begin your Warren Buffett-style analysis.
                </p>
                <br>
                <p style="color: #D4AF37;">
                    <strong>Sample Data Available:</strong><br>
                    AAPL (Apple) | MSFT (Microsoft) | BRK-B (Berkshire Hathaway)
                </p>
                <br>
                <p style="font-size: 0.9rem; color: #666;">
                    💡 For live data on any stock, run this app locally with: <code>streamlit run app.py</code>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick start cards
            st.markdown("### 🚀 Quick Start")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #D4AF37;">📈 Apple (AAPL)</h4>
                    <p style="color: #888;">The world's most valuable company. See why Buffett loves it!</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Analyze AAPL", key="quick_aapl"):
                    st.session_state.quick_symbol = "AAPL"
                    st.rerun()
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #D4AF37;">💻 Microsoft (MSFT)</h4>
                    <p style="color: #888;">Tech giant with strong moats. Excellent financials!</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Analyze MSFT", key="quick_msft"):
                    st.session_state.quick_symbol = "MSFT"
                    st.rerun()
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #D4AF37;">🏛️ Berkshire (BRK-B)</h4>
                    <p style="color: #888;">Buffett's own company. The gold standard!</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Analyze BRK-B", key="quick_brkb"):
                    st.session_state.quick_symbol = "BRK-B"
                    st.rerun()
    
    # ===== TAB 2: GROQ API CHATBOT =====
    with tab2:
        st.markdown('<h3 class="section-header">🤖 Groq API - Warren Buffett Advisor</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="buffett-quote">
            <p>"Risk comes from not knowing what you're doing."</p>
            <p style="text-align: right; color: #D4AF37;">— Warren Buffett</p>
        </div>
        """, unsafe_allow_html=True)
        
        # API Key input - check secrets first, then allow manual input
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🔑 Groq API Settings")
            
            # Try to get API key from secrets first
            groq_api_key = None
            key_source = None
            
            if hasattr(st, 'secrets'):
                try:
                    if 'GROQ_API_KEY' in st.secrets:
                        raw_key = st.secrets['GROQ_API_KEY']
                        # Clean the key
                        groq_api_key = raw_key.strip()
                        groq_api_key = ''.join(c for c in groq_api_key if c.isprintable() and not c.isspace())
                        key_source = "secrets"
                except Exception as e:
                    st.warning(f"⚠️ Error reading secrets: {e}")
            
            if groq_api_key:
                st.success("✅ API Key loaded from secrets")
                # Show debug info
                with st.expander("🔍 Debug Info"):
                    raw_len = len(st.secrets.get('GROQ_API_KEY', ''))
                    clean_len = len(groq_api_key)
                    st.code(f"""Key source: {key_source}
Raw key length: {raw_len} chars
Clean key length: {clean_len} chars
Hidden chars removed: {raw_len - clean_len}
Key prefix: {groq_api_key[:8]}...
Key suffix: ...{groq_api_key[-4:]}
Starts with 'gsk_': {groq_api_key.startswith('gsk_')}""")
                    if raw_len != clean_len:
                        st.warning(f"⚠️ Found {raw_len - clean_len} hidden characters in key!")
                    if not groq_api_key.startswith('gsk_'):
                        st.error("⚠️ Key should start with 'gsk_'")
            else:
                groq_api_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    help="Get your free API key from https://console.groq.com/keys",
                    key="groq_api_key_input"
                )
                if groq_api_key:
                    raw_key = groq_api_key
                    groq_api_key = groq_api_key.strip()
                    groq_api_key = ''.join(c for c in groq_api_key if c.isprintable() and not c.isspace())
                    with st.expander("🔍 Debug Info"):
                        st.code(f"""Key source: manual input
Raw key length: {len(raw_key)} chars
Clean key length: {len(groq_api_key)} chars
Key prefix: {groq_api_key[:8] if len(groq_api_key) >= 8 else groq_api_key}...
Starts with 'gsk_': {groq_api_key.startswith('gsk_')}""")
                        if not groq_api_key.startswith('gsk_'):
                            st.error("⚠️ Key should start with 'gsk_'")
                else:
                    st.info("💡 Get a free API key at [console.groq.com](https://console.groq.com/keys)")
        
        # Model info
        with st.expander("ℹ️ About This Chatbot"):
            st.markdown("""
            **Model:** LLaMA 3.1 70B via Groq API
            
            **Features:**
            - Powered by Meta's LLaMA 3.1 70B model
            - Ultra-fast inference via Groq's LPU technology
            - Maintains conversation context
            - Responds in Warren Buffett's voice and philosophy
            
            **How to get an API key:**
            1. Go to [console.groq.com](https://console.groq.com)
            2. Sign up for a free account
            3. Navigate to API Keys section
            4. Create a new API key
            5. Paste it in the sidebar
            """)
        
        st.markdown("---")
        
        # Chat interface
        st.markdown("### 💬 Chat with Warren Buffett (Groq)")
        
        # Clear chat button
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🗑️ Clear", help="Clear chat history", key="clear_groq"):
                st.session_state.groq_messages = []
                st.rerun()
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.groq_messages:
                with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
                    st.markdown(message["content"])
        
        # Sample questions
        if not st.session_state.groq_messages:
            st.markdown("**Try asking:**")
            sample_questions_groq = [
                "What is your investment philosophy?",
                "How do you evaluate a company's moat?",
                "What mistakes should investors avoid?",
                "How do you think about market volatility?",
                "What advice would you give a new investor?",
            ]
            
            cols = st.columns(3)
            for i, question in enumerate(sample_questions_groq[:3]):
                with cols[i]:
                    if st.button(f"📝 {question[:25]}...", key=f"groq_sample_{i}", help=question):
                        st.session_state.groq_pending_question = question
                        st.rerun()
        
        # Handle pending question from sample buttons
        if "groq_pending_question" in st.session_state:
            prompt = st.session_state.groq_pending_question
            del st.session_state.groq_pending_question
            
            st.session_state.groq_messages.append({"role": "user", "content": prompt})
            
            if groq_api_key:
                response = call_groq_api(prompt, groq_api_key, st.session_state.groq_messages[:-1])
            else:
                response = "🔑 Please enter your Groq API key in the sidebar to use this chatbot."
            
            st.session_state.groq_messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        # Chat input
        if prompt := st.chat_input("Ask Warren Buffett anything...", key="groq_chat_input"):
            # Add user message
            st.session_state.groq_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant", avatar="🤖"):
                if groq_api_key:
                    with st.spinner("Warren is thinking..."):
                        response = call_groq_api(prompt, groq_api_key, st.session_state.groq_messages[:-1])
                else:
                    response = "🔑 Please enter your Groq API key in the sidebar to use this chatbot."
                
                st.markdown(response)
            
            st.session_state.groq_messages.append({"role": "assistant", "content": response})
    
    # ===== TAB 3: CUSTOM TRANSFORMER CHATBOT =====
    with tab3:
        st.markdown('<h3 class="section-header">🎩 Custom Transformer - Warren Buffett Advisor</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="buffett-quote">
            <p>"The most important investment you can make is in yourself."</p>
            <p style="text-align: right; color: #D4AF37;">— Warren Buffett</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if model is available
        model_available = is_model_available() and TF_AVAILABLE
        
        if model_available:
            # Load the chatbot
            chatbot = load_chatbot()
            
            if chatbot.is_loaded():
                st.success("✅ **Chatbot Ready!** The custom-trained Warren Buffett AI advisor is ready.")
                
                # Model info
                with st.expander("ℹ️ Model Information"):
                    st.markdown(f"""
                    **Model Type:** Custom Transformer (trained from scratch)
                    
                    **Architecture:**
                    - **Model Directory:** `{MODEL_DIR}`
                    - **Vocabulary Size:** {chatbot.config.get('vocab_size', 'N/A')}
                    - **Max Length:** {chatbot.config.get('max_length', 'N/A')}
                    - **Layers:** {chatbot.config.get('num_layers', 'N/A')}
                    - **Model Dimension:** {chatbot.config.get('d_model', 'N/A')}
                    
                    **Training:**
                    - Trained on 1,153 Warren Buffett Q&A pairs
                    - Custom TensorFlow/Keras implementation
                    - Runs locally without API calls
                    """)
            else:
                st.warning("⚠️ Model files found but failed to load. Check the console for errors.")
                model_available = False
        else:
            if not TF_AVAILABLE:
                st.warning("⚠️ **TensorFlow not installed.** Install TensorFlow to use the chatbot: `pip install tensorflow`")
            else:
                st.info("""
                🚀 **Train Your Chatbot!**
                
                To enable the custom AI chatbot:
                1. Open `train_chatbot_colab.py` in Google Colab
                2. Upload your Q&A CSV file with Warren Buffett investment knowledge
                3. Train the model (takes ~10-30 minutes with GPU)
                4. Download the model ZIP file
                5. Extract to the `model/` folder in this project
                
                Required files in `model/` folder:
                - `transformer_weights.weights.h5`
                - `tokenizer.json`
                - `config.json`
                """)
        
        st.markdown("---")
        
        # Chat interface
        st.markdown("### 💬 Chat with the Custom Buffett Bot")
        
        # Clear chat button
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🗑️ Clear", help="Clear chat history", key="clear_custom"):
                st.session_state.custom_messages = []
                st.rerun()
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.custom_messages:
                with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🎩"):
                    st.markdown(message["content"])
        
        # Sample questions
        if not st.session_state.custom_messages:
            st.markdown("**Try asking:**")
            sample_questions = [
                "What is gross margin?",
                "How do you select stocks?",
                "What is a good debt to equity ratio?",
                "Why does Buffett avoid high R&D companies?",
                "What makes a company a good investment?",
            ]
            
            cols = st.columns(3)
            for i, question in enumerate(sample_questions[:3]):
                with cols[i]:
                    if st.button(f"📝 {question[:25]}...", key=f"custom_sample_{i}", help=question):
                        st.session_state.custom_pending_question = question
                        st.rerun()
        
        # Handle pending question from sample buttons
        if "custom_pending_question" in st.session_state:
            prompt = st.session_state.custom_pending_question
            del st.session_state.custom_pending_question
            
            st.session_state.custom_messages.append({"role": "user", "content": prompt})
            
            if model_available and chatbot.is_loaded():
                response = chatbot.chat(prompt)
                if response is None:
                    response = "I'm having trouble generating a response. Please try again."
            else:
                response = "🚧 The chatbot model is not loaded. Please train the model first using the Colab notebook."
            
            st.session_state.custom_messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        # Chat input
        if prompt := st.chat_input("Ask about Warren Buffett's investment principles...", key="custom_chat_input"):
            # Add user message
            st.session_state.custom_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant", avatar="🎩"):
                if model_available and chatbot.is_loaded():
                    with st.spinner("Thinking..."):
                        response = chatbot.chat(prompt)
                        if response is None:
                            response = "I'm having trouble generating a response. Please try again."
                else:
                    response = "🚧 The chatbot model is not loaded. Please train the model first using the Colab notebook (`train_chatbot_colab.py`)."
                
                st.markdown(response)
            
            st.session_state.custom_messages.append({"role": "assistant", "content": response})
    
    # ===== TAB 4: LEARN =====
    with tab4:
        st.markdown('<h3 class="section-header">📚 Understanding Buffett\'s Investment Criteria</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        Warren Buffett, known as the "Oracle of Omaha," has developed a systematic approach to evaluating businesses.
        His investment philosophy focuses on finding companies with **durable competitive advantages** (moats) at **fair prices**.
        """)
        
        # Income Statement Section
        st.markdown("### 📈 Income Statement Ratios")
        
        for key, info in BUFFETT_RATIOS["income_statement"].items():
            with st.expander(f"**{info['name']}** — {info['rule']}"):
                st.markdown(f"""
                **What it measures:** {info['logic']}
                
                **Buffett's Threshold:** {info['rule']}
                
                **Why it matters:** This ratio helps identify companies that have genuine competitive advantages
                rather than those that are simply competing on price or spending excessively to maintain their market position.
                """)
        
        # Balance Sheet Section
        st.markdown("### 📊 Balance Sheet Ratios")
        
        for key, info in BUFFETT_RATIOS["balance_sheet"].items():
            with st.expander(f"**{info['name']}** — {info['rule']}"):
                st.markdown(f"""
                **What it measures:** {info['logic']}
                
                **Buffett's Threshold:** {info['rule']}
                
                **Why it matters:** A strong balance sheet indicates financial stability and provides
                the company with flexibility to weather economic downturns and capitalize on opportunities.
                """)
        
        # Cash Flow Section
        st.markdown("### 💰 Cash Flow Ratios")
        
        for key, info in BUFFETT_RATIOS["cash_flow"].items():
            with st.expander(f"**{info['name']}** — {info['rule']}"):
                st.markdown(f"""
                **What it measures:** {info['logic']}
                
                **Buffett's Threshold:** {info['rule']}
                
                **Why it matters:** Free cash flow is what's left after maintaining the business—
                companies that generate abundant free cash flow can reinvest in growth, pay dividends,
                or buy back shares.
                """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🎓 Key Takeaways
        
        1. **Focus on Quality:** Buffett looks for businesses with high margins, low debt, and consistent earnings growth.
        
        2. **Think Like a Business Owner:** Don't just look at stock prices—understand the underlying business.
        
        3. **Patience is Key:** Great businesses are rare. Wait for the right opportunity at the right price.
        
        4. **Margin of Safety:** Buy at prices that provide a buffer against errors in analysis.
        
        5. **Long-term Perspective:** Buffett's favorite holding period is "forever."
        """)


if __name__ == "__main__":
    main()