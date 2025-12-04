# 🐝 AppleBee - Warren Buffett Stock Analysis Dashboard

A comprehensive stock analysis platform that combines Warren Buffett's value investing principles with AI-powered chatbots. Built as a final project for the Master's in Information Systems program at Northeastern University.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Training Details](#-training-details)
- [API Configuration](#-api-configuration)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

## 🎯 Overview

**AppleBee** (a playful nod to Apple Inc., one of Buffett's favorite holdings) is a Streamlit-based application that helps investors analyze stocks using Warren Buffett's time-tested investment criteria. The platform features:

1. **Stock Analysis Dashboard** - Evaluates stocks against 12 key Buffett ratios
2. **Dual AI Chatbots** - Two different approaches to conversational AI:
   - **Groq Chatbot**: LLaMA 3.1 70B model via Groq API for comprehensive responses
   - **Custom Transformer**: A from-scratch transformer model trained on 1,153 Buffett Q&A pairs
3. **Educational Content** - Learn the reasoning behind each of Buffett's investment criteria

### Why Two Chatbots?

This project demonstrates the trade-offs in AI approaches:

| Aspect | Groq (LLaMA 3.1 70B) | Custom Transformer |
|--------|---------------------|-------------------|
| **Strength** | Broad knowledge, nuanced responses | Specialized, authentic Buffett voice |
| **Training** | Pre-trained on internet scale | Trained on curated 1,153 Q&A pairs |
| **Response Style** | Comprehensive, general | Focused, personality-driven |
| **Infrastructure** | Requires API key | Runs locally |

## ✨ Features

### 📊 Stock Analysis Dashboard
- Real-time stock data via Yahoo Finance API
- 12 Warren Buffett investment criteria evaluation
- Visual Buffett Score gauge (0-100%)
- Detailed ratio explanations with pass/fail indicators
- Interactive candlestick price charts
- Sample data available for AAPL, MSFT, and BRK-B

### 🤖 Groq API Chatbot
- Powered by LLaMA 3.1 70B via Groq's ultra-fast LPU
- Maintains conversation context (last 10 messages)
- Warren Buffett persona with authentic voice
- Free API access available

### 🎩 Custom Transformer Chatbot
- Built from scratch using TensorFlow/Keras
- Trained on 1,153 curated Warren Buffett Q&A pairs
- 99.69% training accuracy
- Runs entirely locally without API calls
- Seq2Seq architecture with attention mechanism

### 📚 Educational Content
- Detailed explanations of all 12 Buffett ratios
- Income statement, balance sheet, and cash flow analysis
- Investment philosophy and key takeaways

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Data Source** | Yahoo Finance API (yfinance) |
| **Visualization** | Plotly |
| **LLM Chatbot** | Groq API (LLaMA 3.1 70B) |
| **Custom Model** | TensorFlow/Keras Transformer |
| **Training Environment** | Google Colab (GPU) |

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/applebee.git
   cd applebee
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the model** (for Custom Chatbot)
   
   The `model/` folder should contain:
   - `transformer_weights.weights.h5`
   - `tokenizer.json`
   - `config.json`
   
   If training your own model, see [Training Details](#-training-details).

5. **Configure Groq API** (optional, for Groq Chatbot)
   
   Get a free API key from [console.groq.com](https://console.groq.com/keys)

## 🚀 Usage

### Running Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Deploying to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add your Groq API key in **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_your_api_key_here"
   ```

### Using the Dashboard

1. **Stock Analysis**: Enter a ticker symbol (e.g., AAPL) and click "Analyze Stock"
2. **Groq Chatbot**: Enter your API key in the sidebar, then chat about investing
3. **Custom Chatbot**: Ask questions about Buffett's investment principles
4. **Learn**: Explore detailed explanations of each ratio

## 📁 Project Structure

```
applebee/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore rules
├── .streamlit/
│   └── secrets.toml.example # Secrets template
├── model/
│   ├── config.json          # Model configuration
│   ├── tokenizer.json       # Custom tokenizer vocabulary
│   ├── transformer_weights.weights.h5  # Trained model weights
│   └── training_history.json # Training metrics
└── training/
    ├── train_chatbot_colab.py        # Google Colab training script
    └── warren_buffett_qa_augmented.csv # Training dataset (1,153 Q&A pairs)
```

## 🧠 Model Architecture

### Custom Transformer Specifications

| Parameter | Value |
|-----------|-------|
| **Architecture** | Encoder-Decoder Transformer |
| **Encoder Layers** | 2 |
| **Decoder Layers** | 2 |
| **Model Dimension (d_model)** | 256 |
| **Attention Heads** | 8 |
| **Feed-Forward Units** | 512 |
| **Dropout Rate** | 0.1 |
| **Vocabulary Size** | 3,388 tokens |
| **Max Sequence Length** | 60 tokens |

### Key Components
- **Multi-Head Self-Attention**: 8 parallel attention heads
- **Positional Encoding**: Sinusoidal position embeddings
- **Layer Normalization**: Applied after each sub-layer
- **Custom Learning Rate Schedule**: Warm-up followed by decay

## 📈 Training Details

### Dataset
- **Total Q&A Pairs**: 1,153
- **Sources**: Curated from Buffett's letters, interviews, and books
- **Topics Covered**:
  - Value investing principles
  - Intrinsic value calculation
  - Margin of safety
  - Circle of competence
  - Economic moats
  - Management evaluation
  - Investment psychology
  - Financial ratios

### Data Augmentation
To improve question-matching accuracy, 108 question variations were added:
- Multiple phrasings for the same concept
- Synonym substitution
- Question reformulation

### Training Configuration
```python
EPOCHS = 120
BATCH_SIZE = 64
warmup_steps = 400  # Critical for small datasets
```

### Results

| Metric | Value |
|--------|-------|
| **Final Accuracy** | 99.69% |
| **Training Time** | ~25-35 minutes (Colab GPU) |
| **Model Size** | ~3 MB |

### Training on Google Colab

1. Upload `train_chatbot_colab.py` and `warren_buffett_qa_augmented.csv` to Colab
2. Select **Runtime → Change runtime type → GPU**
3. Run all cells
4. Download the generated `buffett_chatbot_model.zip`
5. Extract to the `model/` folder

## 🔑 API Configuration

### Groq API Setup

1. Visit [console.groq.com](https://console.groq.com)
2. Create a free account
3. Navigate to **API Keys**
4. Create a new key (starts with `gsk_`)
5. Add to Streamlit:
   - **Local**: Create `.streamlit/secrets.toml`
   - **Cloud**: Add in app Settings → Secrets

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

## 🎓 Key Learnings

### Technical Challenges Solved

1. **Keras 3 Compatibility**
   - Issue: TensorFlow Keras 3 broke training code
   - Solution: Fixed keyword arguments, tensor type casting, Lambda layer issues

2. **Learning Rate Warmup**
   - Issue: Model stuck at 60% accuracy
   - Root Cause: `warmup_steps=4000` designed for Wikipedia-scale data
   - Solution: Reduced to `warmup_steps=400` for 1,000-sample dataset

3. **Question-Answer Matching**
   - Issue: Model generated fluent but sometimes incorrect answers
   - Solution: Data augmentation with 108 question variations

## 📄 Warren Buffett's 12 Investment Criteria

### Income Statement Ratios
| Ratio | Buffett's Rule | Logic |
|-------|---------------|-------|
| Gross Margin | ≥ 40% | Signals the company isn't competing on price |
| SG&A Margin | ≤ 30% | Wide-moat companies don't need high overhead |
| R&D Margin | ≤ 30% | R&D doesn't always create shareholder value |
| Depreciation Margin | ≤ 10% | Avoid capital-intensive businesses |
| Interest Expense Margin | ≤ 15% | Great businesses don't need debt |
| Net Margin | ≥ 20% | Great companies convert 20%+ revenue to profit |
| EPS Growth | Positive | Great companies grow profits every year |

### Balance Sheet Ratios
| Ratio | Buffett's Rule | Logic |
|-------|---------------|-------|
| Debt to Equity | ≤ 80% | Conservative leverage for stability |
| Cash to Debt | ≥ 1.0x | More cash than debt indicates strength |
| Retained Earnings Growth | Positive | Reinvesting profits effectively |

### Cash Flow Ratios
| Ratio | Buffett's Rule | Logic |
|-------|---------------|-------|
| CapEx to Net Income | ≤ 50% | Low capital needs = more free cash flow |

## 🔮 Future Enhancements

- [ ] **RAG Implementation**: Retrieve relevant Q&A pairs at query time for improved accuracy
- [ ] **Stock Comparison**: Side-by-side analysis of multiple stocks
- [ ] **Backtesting**: Historical performance of Buffett criteria
- [ ] **Fine-tuned LLM**: Train a larger model on expanded Buffett corpus
- [ ] **Portfolio Tracker**: Save and monitor favorite stocks
- [ ] **News Integration**: Real-time news sentiment analysis

## 👤 Author

**Saurabh**  
Master's in Information Systems  
Northeastern University

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Warren Buffett for decades of investment wisdom
- Charlie Munger for complementary insights
- Berkshire Hathaway shareholder letters
- Groq for providing free API access
- Google Colab for GPU training resources

## ⚠️ Disclaimer

This application is for educational purposes only. It does not constitute financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.

---

<p align="center">
  <i>"Price is what you pay. Value is what you get."</i> — Warren Buffett
</p>
