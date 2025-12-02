"""
Helper module for Warren Buffett Chatbot
Contains all custom layers, functions, and utilities for model inference
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import re
import os

print("✅ Helper module loaded")

# =============================================================================
# POSITIONAL ENCODING LAMBDA FUNCTION
# =============================================================================

D_MODEL = 256

def add_pos_enc(x):
    """Positional encoding for Lambda layers."""
    seq_length = tf.shape(x)[1]
    position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
    dim_indices = tf.range(D_MODEL, dtype=tf.float32)[tf.newaxis, :]
    angles = position / tf.pow(
        10000.0, (2 * (dim_indices // 2)) / tf.cast(D_MODEL, tf.float32)
    )
    angles = tf.where(
        tf.cast(dim_indices, tf.int32) % 2 == 0,
        tf.sin(angles),
        tf.cos(angles),
    )
    return x + angles[tf.newaxis, :, :]

# =============================================================================
# POSITIONAL ENCODING CLASS
# =============================================================================

class PositionalEncoding(tf.keras.layers.Layer):
    """Custom positional encoding layer."""
    
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
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model,
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

# =============================================================================
# ATTENTION FUNCTIONS
# =============================================================================

def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights."""
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    if mask is not None:
        logits += mask * -1e9

    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output

# =============================================================================
# MULTI-HEAD ATTENTION LAYER
# =============================================================================

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    """Custom multi-head attention layer."""
    
    def __init__(self, d_model, num_heads, **kwargs):
        assert d_model % num_heads == 0
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "d_model": self.d_model,
        })
        return config

    def split_heads(self, inputs, batch_size):
        inputs = tf.keras.layers.Lambda(
            lambda inputs: tf.reshape(
                inputs, shape=(batch_size, -1, self.num_heads, self.depth)
            )
        )(inputs)
        return tf.keras.layers.Lambda(
            lambda inputs: tf.transpose(inputs, perm=[0, 2, 1, 3])
        )(inputs)

    def call(self, inputs):
        query, key, value, mask = (
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["mask"],
        )
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.keras.layers.Lambda(
            lambda scaled_attention: tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        )(scaled_attention)

        concat_attention = tf.keras.layers.Lambda(
            lambda scaled_attention: tf.reshape(
                scaled_attention, (batch_size, -1, self.d_model)
            )
        )(scaled_attention)

        outputs = self.dense(concat_attention)
        return outputs

# =============================================================================
# MASK FUNCTIONS
# =============================================================================

def create_padding_mask(x):
    """Create padding mask for padding tokens."""
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    """Create look-ahead mask to prevent attending to future tokens."""
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

# =============================================================================
# TOKENIZER AND CONFIG (Lazy Load)
# =============================================================================

MAX_LENGTH = 80
_tokenizer = None
START_TOKEN = None
END_TOKEN = None
VOCAB_SIZE = None

def get_tokenizer():
    """
    Lazy load tokenizer - only when needed.
    Tries multiple paths for flexibility.
    """
    global _tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE
    
    if _tokenizer is None:
        # Try different possible tokenizer paths
        possible_paths = [
            './models/tokenizer_vocab',
            'models/tokenizer_vocab',
            '/models/tokenizer_vocab',
        ]
        
        tokenizer_loaded = False
        for path in possible_paths:
            try:
                if any(os.path.exists(f"{path}{ext}") for ext in ['', '.subwords']):
                    _tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(path)
                    tokenizer_loaded = True
                    break
            except:
                continue
        
        if not tokenizer_loaded:
            raise FileNotFoundError(
                f"Could not load tokenizer!\n"
                f"Tried paths: {possible_paths}\n"
                f"Make sure tokenizer_vocab.subwords exists in ./models/ directory"
            )
        
        # Initialize tokens
        START_TOKEN = [_tokenizer.vocab_size]
        END_TOKEN = [_tokenizer.vocab_size + 1]
        VOCAB_SIZE = _tokenizer.vocab_size + 2
        
        print(f"✅ Tokenizer loaded! Vocab size: {VOCAB_SIZE}")
    
    return _tokenizer

# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def preprocess_sentence(sentence):
    """Preprocess text for model input."""
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
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
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

# =============================================================================
# MODEL INFERENCE
# =============================================================================

def evaluate(sentence, model):
    """
    Generate model output for input sentence using auto-regressive decoding.
    """
    tokenizer = get_tokenizer()
    sentence = preprocess_sentence(sentence)

    # Tokenize input
    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0
    )

    output = tf.expand_dims(START_TOKEN, 0)

    # Auto-regressive generation
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # Stop if END token is predicted
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def predict(sentence, model):
    """
    Generate text prediction from model.
    
    Args:
        sentence: Input text string
        model: Trained transformer model
        
    Returns:
        Generated response text
    """
    try:
        tokenizer = get_tokenizer()
        prediction = evaluate(sentence, model)
        
        # Decode tokens back to text
        predicted_sentence = tokenizer.decode(
            [i for i in prediction if i < tokenizer.vocab_size]
        )
        
        return predicted_sentence
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return f"I encountered an error processing your question: {str(e)}"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_model_safe(model_path):
    """
    Safely load model from SavedModel or H5 format.
    
    Args:
        model_path: Path to model directory or file
        
    Returns:
        Loaded keras model
    """
    custom_objects = {
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
        "create_padding_mask": create_padding_mask,
        "create_look_ahead_mask": create_look_ahead_mask,
        "scaled_dot_product_attention": scaled_dot_product_attention,
        "add_pos_enc": add_pos_enc,
    }
    
    try:
        # Try SavedModel format first
        if os.path.isdir(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Loaded SavedModel from {model_path}")
            return model
        
        # Fall back to H5 format
        elif model_path.endswith('.h5') or model_path.endswith('.keras'):
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print(f"✅ Loaded H5 model from {model_path}")
            return model
        
        else:
            raise ValueError(f"Unknown model format: {model_path}")
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

print("✅ All helper functions initialized!")