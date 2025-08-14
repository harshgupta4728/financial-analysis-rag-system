# 📊 Financial Analysis RAG System

A comprehensive **Retrieval-Augmented Generation (RAG)** system for financial analysis that integrates financial reports, real-time market data, and financial news to provide AI-powered investment insights and risk assessments.

## 🎯 Project Overview

This system combines multiple data sources to create a powerful financial analysis platform:

- **📄 Financial Reports**: SEC filings (10-K, 10-Q), annual reports, and other financial documents
- **📈 Market Data**: Real-time and historical stock data with technical indicators
- **📰 Financial News**: Latest news articles with sentiment analysis
- **🤖 AI Analysis**: Advanced LLM-powered insights and risk assessments

## ✨ Key Features

### 🔍 Multi-Source Data Integration
- **PDF Processing**: Extract and analyze financial reports
- **Market Data**: Real-time stock prices, volumes, and technical indicators
- **News Aggregation**: Financial news from multiple sources with sentiment analysis

### 📊 Advanced Analytics
- **Time-Series Analysis**: Trend identification and volatility analysis
- **Technical Indicators**: RSI, MACD, Bollinger Bands, moving averages
- **Risk Metrics**: VaR, Sharpe ratio, maximum drawdown, beta analysis
- **Sentiment Analysis**: News sentiment scoring and trend analysis

### 🤖 AI-Powered Insights
- **RAG System**: Context-aware responses using retrieved documents
- **Multi-Modal Analysis**: Combine reports, news, and market data
- **Risk Assessment**: Comprehensive risk analysis and mitigation strategies
- **Investment Recommendations**: Data-driven investment insights

### 🎨 Modern Web Interface
- **Streamlit UI**: Clean, intuitive, and professional interface
- **Interactive Charts**: Plotly-powered market visualizations
- **Real-time Updates**: Live data fetching and analysis
- **Responsive Design**: Works on desktop and mobile devices

## 🏗️ System Architecture

```
Financial Analysis RAG System
├── 📥 Data Ingestion Layer
│   ├── PDF Processing (PyPDF2)
│   ├── Market Data (yfinance)
│   └── News Aggregation (RSS feeds)
├── 🔧 Processing Layer
│   ├── Text Chunking & Cleaning
│   ├── Embedding Generation (Sentence Transformers)
│   └── Vector Storage (ChromaDB)
├── 🤖 AI Layer
│   ├── Document Retrieval
│   ├── LLM Integration (OpenAI)
│   └── Response Generation
├── 📊 Analysis Layer
│   ├── Technical Analysis
│   ├── Risk Metrics
│   └── Sentiment Analysis
└── 🎨 Presentation Layer
    └── Streamlit Web Interface
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **OpenAI API Key** (required for LLM responses)
- **Internet Connection** (for market data and news)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd financial-rag-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   # Required
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional
   NEWS_API_KEY=your_news_api_key_here
   
   # Application Configuration
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   TEMPERATURE=0.7
   MAX_TOKENS=2000
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📖 Usage Guide

### 1. Data Ingestion

#### Upload Financial Reports
- Go to the **"Data Ingestion"** tab
- Upload PDF financial reports (10-K, 10-Q, annual reports)
- The system will extract text and create searchable chunks

#### Fetch Market Data
- Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
- Select the time period (1 month to 5 years)
- Click "Fetch Market Data" to get historical prices and metrics

#### Collect News Articles
- Optionally filter news by ticker symbol
- Select the number of days to look back
- Click "Fetch News" to get recent financial news

### 2. Query & Analysis

#### Ask Questions
- Go to the **"Query & Analysis"** tab
- Enter your question in natural language
- Examples:
  - "What are the key risk factors for AAPL?"
  - "Analyze the market sentiment for Tesla based on recent news"
  - "What are the main financial highlights from the latest 10-K report?"

#### Choose Analysis Type
- **Analysis**: Comprehensive financial analysis
- **Summary**: Executive summary of findings
- **Risk Assessment**: Detailed risk analysis

#### View Results
- AI-generated responses with source citations
- Confidence levels and relevance scores
- Metadata about the analysis

### 3. Market Analysis

#### Technical Analysis
- Enter a stock ticker for detailed analysis
- View interactive price charts with moving averages
- Analyze technical indicators (RSI, MACD, Bollinger Bands)

#### Risk Metrics
- Value at Risk (VaR) calculations
- Sharpe ratio and maximum drawdown
- Volatility analysis and beta calculations

#### Market Insights
- Automated trend analysis
- Technical signal interpretation
- Risk assessment and recommendations

### 4. News Sentiment

#### Sentiment Analysis
- Analyze sentiment for specific stocks
- View overall sentiment scores and trends
- Examine individual article sentiments

#### Sentiment Trends
- Track sentiment changes over time
- Identify positive/negative news patterns
- Correlate sentiment with market performance

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
NEWS_API_KEY=your_news_api_key_here

# Application Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=all-MiniLM-L6-v2
TEMPERATURE=0.7
MAX_TOKENS=2000
```

### Model Configuration

The system uses the following models by default:

- **Embedding Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **LLM Model**: `gpt-3.5-turbo` (OpenAI)
- **Vector Database**: ChromaDB

You can modify these in the respective module files.

## 📊 Data Sources

### Market Data
- **Source**: Yahoo Finance (via yfinance)
- **Data Types**: Historical prices, volumes, financial ratios
- **Update Frequency**: Real-time during market hours

### Financial News
- **Sources**: Reuters, Bloomberg, Yahoo Finance RSS feeds
- **Coverage**: Global financial news
- **Update Frequency**: Real-time

### Financial Reports
- **Formats**: PDF documents
- **Types**: SEC filings, annual reports, quarterly reports
- **Processing**: Text extraction and intelligent chunking

## 🏗️ Project Structure

```
financial-rag-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── SETUP.md              # Quick setup guide
└── src/                  # Source code modules
    ├── __init__.py
    ├── data_ingestion.py    # Data loading and processing
    ├── text_processing.py   # Text chunking and cleaning
    ├── vector_store.py      # ChromaDB integration
    ├── llm_interface.py     # OpenAI API integration
    └── analysis.py          # Financial analysis and charts
```

## 🔍 Technical Details

### RAG Implementation

1. **Document Processing**
   - Intelligent chunking based on document type
   - Financial report section extraction
   - Text cleaning and normalization

2. **Vector Storage**
   - ChromaDB for efficient similarity search
   - Separate collections for different data types
   - Metadata tracking for source attribution

3. **Retrieval & Generation**
   - Semantic search across all collections
   - Context-aware prompt engineering
   - Confidence scoring based on relevance

### Analysis Capabilities

#### Technical Analysis
- **Trend Analysis**: Price trends, moving averages, golden/death crosses
- **Volatility Analysis**: Historical volatility, volatility clustering
- **Technical Indicators**: RSI, MACD, Bollinger Bands, volume analysis

#### Risk Assessment
- **Value at Risk**: 95% and 99% VaR calculations
- **Risk Metrics**: Sharpe ratio, maximum drawdown, beta
- **Portfolio Risk**: Correlation analysis and diversification metrics

#### Sentiment Analysis
- **Keyword-based Analysis**: Positive/negative keyword scoring
- **Article-level Sentiment**: Individual article sentiment scores
- **Aggregate Sentiment**: Overall sentiment trends and volatility

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Set environment variables in Streamlit Cloud dashboard
4. Deploy automatically

### Hugging Face Spaces
1. Create a new Space on Hugging Face
2. Upload your code and requirements.txt
3. Configure environment variables
4. Deploy and share your app

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models and API
- **Hugging Face** for sentence transformers
- **ChromaDB** for vector database
- **Streamlit** for the web framework
- **Yahoo Finance** for market data
- **Financial news sources** for content

## 📞 Support

For questions, issues, or contributions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed description
3. Contact the maintainers

## 🔮 Future Enhancements

- [ ] **Real-time Data Streaming**: Live market data updates
- [ ] **Portfolio Analysis**: Multi-stock portfolio management
- [ ] **Advanced ML Models**: Custom financial prediction models
- [ ] **API Endpoints**: REST API for external integrations
- [ ] **Mobile App**: Native mobile application
- [ ] **Advanced Charts**: More technical indicators and chart types
- [ ] **Backtesting**: Historical strategy testing
- [ ] **Alert System**: Price and news alerts
- [ ] **Export Features**: PDF reports and data exports
- [ ] **User Authentication**: Multi-user support with roles

---

**Built with ❤️ for the financial analysis community**
