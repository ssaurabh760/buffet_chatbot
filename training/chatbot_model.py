# -*- coding: utf-8 -*-
"""
Warren Buffett Investment Advisor - Transformer Chatbot Training
=================================================================
Keras 3 Compatible Version

Run this notebook in Google Colab to train the chatbot model.

Instructions:
1. Upload this file to Google Colab
2. Upload your Q&A CSV file to Colab
3. Run all cells
4. Download the model files from the 'model' folder

Google Colab Setup:
- Go to Runtime > Change runtime type > Select GPU
"""

# ============================================================================
# STEP 1: Imports
# ============================================================================

import os
import re
import json
import numpy as np
import pandas as pd
from time import time
import tensorflow as tf
import zipfile

print(f"TensorFlow version: {tf.__version__}")

# Set random seed
tf.keras.utils.set_random_seed(1234)

# ============================================================================
# STEP 2: GPU Setup
# ============================================================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("No GPU detected. Training will be slower on CPU.")

# ============================================================================
# STEP 3: Hyperparameters
# ============================================================================

MAX_LENGTH = 60
BATCH_SIZE = 64
BUFFER_SIZE = 10000

NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

EPOCHS = 120
OUTPUT_DIR = "./model"

print(f"""
Hyperparameters:
- MAX_LENGTH: {MAX_LENGTH}
- BATCH_SIZE: {BATCH_SIZE}
- NUM_LAYERS: {NUM_LAYERS}
- D_MODEL: {D_MODEL}
- NUM_HEADS: {NUM_HEADS}
- UNITS: {UNITS}
- DROPOUT: {DROPOUT}
- EPOCHS: {EPOCHS}
""")

# ============================================================================
# STEP 4: Upload CSV File
# ============================================================================

from google.colab import files

print("Please upload your Q&A CSV file...")
uploaded = files.upload()
CSV_FILENAME = list(uploaded.keys())[0]
print(f"\nUploaded file: {CSV_FILENAME}")

# ============================================================================
# STEP 5: Text Preprocessing
# ============================================================================

def preprocess_sentence(sentence):
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
    sentence = re.sub(r"[^a-zA-Z0-9?.!,%]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

# ============================================================================
# STEP 6: Load Data
# ============================================================================

def load_conversations_from_csv(csv_path):
    print(f"Loading data from {csv_path}...")
    
    # Try tab delimiter first (common for Q&A files)
    for delimiter in ['\t', ',', ';']:
        try:
            df = pd.read_csv(csv_path, delimiter=delimiter, on_bad_lines='skip')
            if len(df.columns) >= 2:
                print(f"Loaded with delimiter: {repr(delimiter)}")
                break
        except:
            continue
    
    print(f"Columns: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    
    # Get column names
    q_col = df.columns[0]
    a_col = df.columns[1]
    
    df = df.dropna(subset=[q_col, a_col])
    
    questions = [preprocess_sentence(q) for q in df[q_col].tolist()]
    answers = [preprocess_sentence(a) for a in df[a_col].tolist()]
    
    # Filter empty
    pairs = [(q, a) for q, a in zip(questions, answers) if q and a]
    questions = [q for q, a in pairs]
    answers = [a for q, a in pairs]
    
    print(f"Loaded {len(questions)} Q&A pairs")
    return questions, answers

questions, answers = load_conversations_from_csv(CSV_FILENAME)

print(f"\nSample Q&A pairs:")
for i in range(min(3, len(questions))):
    print(f"  Q: {questions[i][:60]}...")
    print(f"  A: {answers[i][:60]}...")
    print()

# ============================================================================
# STEP 7: Tokenizer
# ============================================================================

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def build_from_corpus(self, sentences, vocab_size=8192):
        word_counts = {}
        for sentence in sentences:
            for word in sentence.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, _ in sorted_words[:vocab_size - 2]]
        
        self.word2idx = {'<PAD>': 0, '<OOV>': 1}
        for idx, word in enumerate(vocab_words, start=2):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary size: {self.vocab_size}")
        return self
    
    def encode(self, sentence):
        return [self.word2idx.get(word, 1) for word in sentence.split()]
    
    def decode(self, tokens):
        return ' '.join([self.idx2word.get(idx, '<OOV>') for idx in tokens if idx > 0])
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'word2idx': self.word2idx, 'idx2word': {int(k): v for k, v in self.idx2word.items()}, 'vocab_size': self.vocab_size}, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        t = cls()
        t.word2idx = data['word2idx']
        t.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        t.vocab_size = data['vocab_size']
        return t

tokenizer = SimpleTokenizer()
tokenizer.build_from_corpus(questions + answers)

START_TOKEN = tokenizer.vocab_size
END_TOKEN = tokenizer.vocab_size + 1
VOCAB_SIZE = tokenizer.vocab_size + 2

print(f"Total vocab size (with special tokens): {VOCAB_SIZE}")

# ============================================================================
# STEP 8: Tokenize Data
# ============================================================================

def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []
    
    for q, a in zip(inputs, outputs):
        q_tok = [START_TOKEN] + tokenizer.encode(q) + [END_TOKEN]
        a_tok = [START_TOKEN] + tokenizer.encode(a) + [END_TOKEN]
        
        if len(q_tok) <= MAX_LENGTH and len(a_tok) <= MAX_LENGTH:
            tokenized_inputs.append(q_tok)
            tokenized_outputs.append(a_tok)
    
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=MAX_LENGTH, padding="post")
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=MAX_LENGTH, padding="post")
    
    return tokenized_inputs, tokenized_outputs

questions_tok, answers_tok = tokenize_and_filter(questions, answers)
print(f"Samples after filtering: {len(questions_tok)}")

# ============================================================================
# STEP 9: Create Dataset
# ============================================================================

# Create encoder inputs, decoder inputs, and labels
encoder_inputs = questions_tok
decoder_inputs = answers_tok[:, :-1]
labels = answers_tok[:, 1:]

dataset = tf.data.Dataset.from_tensor_slices((
    (encoder_inputs, decoder_inputs),
    labels
))
dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Dataset created: {dataset}")

# ============================================================================
# STEP 10: Transformer Components
# ============================================================================

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

# ============================================================================
# STEP 11: Loss and Metrics
# ============================================================================

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.int32))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

# ============================================================================
# STEP 12: Learning Rate Schedule
# ============================================================================

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=400):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model_float = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model_float) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

# ============================================================================
# STEP 13: Build Model
# ============================================================================

print("\nBuilding Transformer model...")

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=UNITS,
    input_vocab_size=VOCAB_SIZE,
    target_vocab_size=VOCAB_SIZE,
    pe_input=VOCAB_SIZE,
    pe_target=VOCAB_SIZE,
    rate=DROPOUT
)

transformer.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy_function])

# Build model by calling it once
sample_input = (tf.zeros((1, MAX_LENGTH), dtype=tf.int32), tf.zeros((1, MAX_LENGTH-1), dtype=tf.int32))
_ = transformer(sample_input)
transformer.summary()

# ============================================================================
# STEP 14: Train
# ============================================================================

print(f"\n{'='*60}")
print(f"Starting training for {EPOCHS} epochs...")
print(f"{'='*60}\n")

start_time = time()

history = transformer.fit(dataset, epochs=EPOCHS)

training_time = time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# ============================================================================
# STEP 15: Save Model
# ============================================================================

print(f"\n{'='*60}")
print("Saving model and configuration files...")
print(f"{'='*60}\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save weights only (more compatible)
weights_path = os.path.join(OUTPUT_DIR, "transformer_weights.weights.h5")
transformer.save_weights(weights_path)
print(f"✓ Weights saved to {weights_path}")

# Save tokenizer
tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
tokenizer.save(tokenizer_path)
print(f"✓ Tokenizer saved")

# Save config
config = {
    "vocab_size": VOCAB_SIZE,
    "max_length": MAX_LENGTH,
    "num_layers": NUM_LAYERS,
    "d_model": D_MODEL,
    "num_heads": NUM_HEADS,
    "units": UNITS,
    "dropout": DROPOUT,
    "start_token": START_TOKEN,
    "end_token": END_TOKEN,
}
config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"✓ Config saved")

# Save history
history_path = os.path.join(OUTPUT_DIR, "training_history.json")
with open(history_path, 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
print(f"✓ History saved")

# ============================================================================
# STEP 16: Test Model
# ============================================================================

print(f"\n{'='*60}")
print("Testing the trained model...")
print(f"{'='*60}\n")

def evaluate(sentence):

    sentence = preprocess_sentence(sentence)

    sentence = [START_TOKEN] + tokenizer.encode(sentence) + [END_TOKEN]

    sentence = tf.keras.preprocessing.sequence.pad_sequences([sentence], maxlen=MAX_LENGTH, padding="post")

    encoder_input = tf.cast(sentence, tf.int32)

    

    decoder_input = [START_TOKEN]

    output = tf.expand_dims(decoder_input, 0)

    output = tf.cast(output, tf.int32)

    

    for i in range(MAX_LENGTH):

        predictions = transformer((encoder_input, output), training=False)

        predictions = predictions[:, -1:, :]

        predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)

        predicted_id_val = int(predicted_id.numpy()[0][0])

        

        if predicted_id_val == END_TOKEN:

            break

        

        output = tf.concat([output, predicted_id], axis=-1)

    

    return tf.squeeze(output, axis=0)



def predict(sentence):

    prediction = evaluate(sentence)

    predicted_tokens = [i for i in prediction.numpy() if i < tokenizer.vocab_size and i > 0]

    return tokenizer.decode(predicted_tokens)



test_questions = [

    "What is gross margin?",

    "How do you select stocks?",

    "What is value investing?",

    "Who are you?",

    "What is intrinsic value?",

]



for q in test_questions:

    print(f"Q: {q}")

    print(f"A: {predict(q)}")

    print()



# ============================================================================

# STEP 17: Create ZIP and Download

# ============================================================================



print(f"\n{'='*60}")

print("Creating ZIP file for download...")

print(f"{'='*60}\n")



zip_filename = "buffett_chatbot_model.zip"

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:

    for root, dirs, files_list in os.walk(OUTPUT_DIR):

        for file in files_list:

            file_path = os.path.join(root, file)

            arcname = os.path.relpath(file_path, OUTPUT_DIR)

            zipf.write(file_path, arcname)



print(f"✓ Created {zip_filename}")

print(f"\nFiles in ZIP:")

with zipfile.ZipFile(zip_filename, 'r') as zipf:

    for name in zipf.namelist():

        print(f"  - {name}")



files.download(zip_filename)



print("""

✅ TRAINING COMPLETE!



To use in your AppleBee app:

1. Unzip buffett_chatbot_model.zip

2. Place files in: applebee/model/

3. Run: streamlit run app.py

""")



# ============================================================================

# STEP 18: Interactive Testing

# ============================================================================



print(f"\n{'='*60}")

print("Interactive Testing - Type 'quit' to exit")

print(f"{'='*60}\n")



while True:

    user_input = input("You: ").strip()

    if user_input.lower() in ['quit', 'exit', 'q', '']:

        print("Goodbye!")

        break

    print(f"Buffett Bot: {predict(user_input)}\n")

