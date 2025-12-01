"""
local_transformer_helper.py

Helper module for using pre-trained/custom TensorFlow transformer models
in your AppleBee chatbot project.

Based on: tf2_tpu_transformer_chatbot.ipynb (your professor's code)
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import re
import os
from typing import Tuple

# =============================================================================
# SECTION 1: CUSTOM LAYERS
# (Identical to professor's implementation)
# =============================================================================

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Adds positional encoding to embeddings.
    
    Since transformers have no recurrence, we need to encode position info.
    """
    
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model,
        })
        return config

    def get_angles(self, position, i, d_model):
        """Calculate angles for positional encoding."""
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        """Generate positional encoding matrix."""
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model,
        )
        # Apply sin to even indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        """Add positional encoding to input embeddings."""
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]


def scaled_dot_product_attention(query, key, value, mask):
    """
    Calculate scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # Scale by sqrt(depth)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # Add mask (for padding tokens)
    if mask is not None:
        logits += mask * -1e9  # Large negative = softmax ≈ 0

    # Apply softmax to get attention weights
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # Multiply by values
    output = tf.matmul(attention_weights, value)

    return output


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    """
    Multi-head attention: split into multiple heads, apply attention in parallel.
    """
    
    def __init__(self, d_model, num_heads, **kwargs):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        # Linear layers for Q, K, V
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # Final output dense layer
        self.dense = tf.keras.layers.Dense(units=d_model)

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "d_model": self.d_model,
        })
        return config

    def split_heads(self, inputs, batch_size):
        """
        Reshape and transpose inputs to split into multiple heads.
        
        Input shape: (batch, seq_len, d_model)
        Output shape: (batch, num_heads, seq_len, depth)
        """
        inputs = tf.keras.layers.Lambda(
            lambda inputs: tf.reshape(
                inputs, shape=(batch_size, -1, self.num_heads, self.depth)
            )
        )(inputs)
        return tf.keras.layers.Lambda(
            lambda inputs: tf.transpose(inputs, perm=[0, 2, 1, 3])
        )(inputs)

    def call(self, inputs):
        query = inputs["query"]
        key = inputs["key"]
        value = inputs["value"]
        mask = inputs["mask"]
        
        batch_size = tf.shape(query)[0]

        # Apply linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split into multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Apply scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        
        # Transpose back: (batch, num_heads, seq_len, depth) -> (batch, seq_len, num_heads, depth)
        scaled_attention = tf.keras.layers.Lambda(
            lambda scaled_attention: tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        )(scaled_attention)

        # Concatenate heads: (batch, seq_len, d_model)
        concat_attention = tf.keras.layers.Lambda(
            lambda scaled_attention: tf.reshape(
                scaled_attention, (batch_size, -1, self.d_model)
            )
        )(scaled_attention)

        # Final linear layer
        outputs = self.dense(concat_attention)

        return outputs


def create_padding_mask(x):
    """
    Create mask for padding tokens (value 0).
    
    Returns: (batch_size, 1, 1, seq_len) - ready to broadcast
    """
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    """
    Create mask to prevent attention to future tokens.
    Also masks padding tokens.
    """
    seq_len = tf.shape(x)[1]
    # Upper triangular matrix (prevent future attention)
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    # Padding mask
    padding_mask = create_padding_mask(x)
    # Combine both
    return tf.maximum(look_ahead_mask, padding_mask)


# =============================================================================
# SECTION 2: TEXT PREPROCESSING
# =============================================================================

def preprocess_sentence(sentence: str) -> str:
    """
    Preprocess sentence: lowercase, remove punctuation, expand contractions.
    
    Args:
        sentence: Raw text input
        
    Returns:
        Cleaned sentence
    """
    sentence = sentence.lower().strip()
    
    # Add space before punctuation
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    
    # Expand contractions
    contractions = {
        r"i'm": "i am",
        r"he's": "he is",
        r"she's": "she is",
        r"it's": "it is",
        r"that's": "that is",
        r"what's": "that is",
        r"where's": "where is",
        r"how's": "how is",
        r"\'ll": " will",
        r"\'ve": " have",
        r"\'re": " are",
        r"\'d": " would",
        r"won't": "will not",
        r"can't": "cannot",
        r"n't": " not",
        r"n'": "ng",
        r"'bout": "about",
    }
    
    for pattern, replacement in contractions.items():
        sentence = re.sub(pattern, replacement, sentence)
    
    # Keep only alphanumeric and basic punctuation
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    
    return sentence


# =============================================================================
# SECTION 3: MAIN CHATBOT CLASS
# =============================================================================

class TransformerChatbot:
    """
    Wrapper class for using pre-trained transformer model.
    
    Usage:
        chatbot = TransformerChatbot('./models/model.h5', './models/tokenizer')
        response = chatbot.predict("Hello!")
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, max_length: int = 40):
        """
        Initialize chatbot with model and tokenizer.
        
        Args:
            model_path: Path to saved model.h5 file
            tokenizer_path: Path to tokenizer vocab file
            max_length: Maximum sequence length for generation
            
        Raises:
            FileNotFoundError: If model or tokenizer not found
        """
        self.max_length = max_length
        
        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        # Load tokenizer
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
            tokenizer_path
        )
        
        # Define special tokens
        self.START_TOKEN = [self.tokenizer.vocab_size]
        self.END_TOKEN = [self.tokenizer.vocab_size + 1]
        
        print(f"Loading model from {model_path}...")
        # Load model with custom layers
        self.model = load_model(
            model_path,
            custom_objects={
                "PositionalEncoding": PositionalEncoding,
                "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
            },
        )
        
        print("✅ Model loaded successfully!")
        self.print_info()
    
    def print_info(self):
        """Print model information."""
        print("\n" + "="*50)
        print("CHATBOT INFO")
        print("="*50)
        print(f"Vocabulary Size: {self.tokenizer.vocab_size + 2}")
        print(f"Max Sequence Length: {self.max_length}")
        print(f"Model Trainable Params: {self.model.count_params():,}")
        print("="*50 + "\n")
    
    def evaluate(self, sentence: str) -> tf.Tensor:
        """
        Generate response token-by-token (auto-regressive generation).
        
        Args:
            sentence: Input sentence from user
            
        Returns:
            Generated tokens (TensorFlow tensor)
        """
        # Preprocess
        sentence = preprocess_sentence(sentence)
        
        # Tokenize and add special tokens
        sentence = tf.expand_dims(
            self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0
        )
        
        # Initialize decoder input with START token
        output = tf.expand_dims(self.START_TOKEN, 0)
        
        # Generate tokens one at a time
        for i in range(self.max_length):
            # Get predictions from model
            predictions = self.model(
                inputs=[sentence, output], 
                training=False
            )
            
            # Select last token
            predictions = predictions[:, -1:, :]
            
            # Get most likely token
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
            # Stop if END token is generated
            if tf.equal(predicted_id, self.END_TOKEN[0]):
                break
            
            # Append to output
            output = tf.concat([output, predicted_id], axis=-1)
        
        return tf.squeeze(output, axis=0)
    
    def predict(self, sentence: str) -> str:
        """
        Generate response and convert tokens back to text.
        
        Args:
            sentence: Input sentence from user
            
        Returns:
            Generated response as text
        """
        # Generate tokens
        prediction = self.evaluate(sentence)
        
        # Convert back to text (filter out special tokens)
        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size]
        )
        
        return predicted_sentence
    
    def predict_with_confidence(self, sentence: str) -> Tuple[str, float]:
        """
        Generate response with confidence score.
        
        Args:
            sentence: Input sentence from user
            
        Returns:
            Tuple of (generated_text, confidence_score)
        """
        sentence_processed = preprocess_sentence(sentence)
        sentence_tokens = tf.expand_dims(
            self.START_TOKEN + self.tokenizer.encode(sentence_processed) + self.END_TOKEN, 
            axis=0
        )
        
        output = tf.expand_dims(self.START_TOKEN, 0)
        total_confidence = 1.0
        tokens_generated = 0
        
        for i in range(self.max_length):
            predictions = self.model(
                inputs=[sentence_tokens, output], 
                training=False
            )
            
            predictions = predictions[:, -1:, :]
            
            # Get probabilities
            probabilities = tf.nn.softmax(predictions[0, 0, :])
            
            # Get max probability
            max_prob = tf.reduce_max(probabilities).numpy()
            total_confidence *= max_prob
            
            # Get token ID
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
            if tf.equal(predicted_id, self.END_TOKEN[0]):
                break
            
            output = tf.concat([output, predicted_id], axis=-1)
            tokens_generated += 1
        
        # Normalize confidence
        if tokens_generated > 0:
            confidence = (total_confidence ** (1.0 / tokens_generated))
        else:
            confidence = 0.0
        
        predicted_sentence = self.tokenizer.decode(
            [i for i in output[0] if i < self.tokenizer.vocab_size 
             and i not in self.START_TOKEN + self.END_TOKEN]
        )
        
        return predicted_sentence, float(confidence)


# =============================================================================
# SECTION 4: UTILITY FUNCTIONS
# =============================================================================

def load_or_create_chatbot(
    model_path: str, 
    tokenizer_path: str,
    auto_download: bool = False
) -> TransformerChatbot:
    """
    Load chatbot, with option to auto-download if not found.
    
    Args:
        model_path: Path to model file
        tokenizer_path: Path to tokenizer file
        auto_download: If True, try to download from GitHub
        
    Returns:
        TransformerChatbot instance
        
    Raises:
        FileNotFoundError: If files not found and auto_download=False
    """
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        if auto_download:
            print("⚠️  Model files not found. Downloading...")
            # Implement download logic if needed
            raise NotImplementedError("Auto-download not implemented yet")
        else:
            raise FileNotFoundError(
                f"Model files not found at:\n"
                f"  Model: {model_path}\n"
                f"  Tokenizer: {tokenizer_path}\n"
                f"Download from: https://github.com/bryanlimy/tf2-transformer-chatbot"
            )
    
    return TransformerChatbot(model_path, tokenizer_path)


# =============================================================================
# SECTION 5: TEST FUNCTION
# =============================================================================

def test_chatbot(
    model_path: str = "./models/transformer_model.h5",
    tokenizer_path: str = "./models/tokenizer"
):
    """
    Test the chatbot with sample questions.
    
    Usage:
        test_chatbot()
    """
    try:
        chatbot = TransformerChatbot(model_path, tokenizer_path)
        
        # Test queries
        test_queries = [
            "Hello!",
            "How are you?",
            "Tell me a joke",
            "What's your name?",
        ]
        
        print("\n" + "="*50)
        print("CHATBOT RESPONSES")
        print("="*50)
        
        for query in test_queries:
            response = chatbot.predict(query)
            print(f"\nUser: {query}")
            print(f"Bot:  {response}")
        
        print("\n" + "="*50)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nTo use this, you need to:")
        print("1. Train the model using professor's notebook, OR")
        print("2. Download pre-trained model from:")
        print("   https://github.com/bryanlimy/tf2-transformer-chatbot")
        print("3. Place files in ./models/")


if __name__ == "__main__":
    # Quick test
    test_chatbot()