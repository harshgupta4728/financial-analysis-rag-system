# PATCH: For Streamlit Cloud compatibility with ChromaDB
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# END PATCH

"""
Financial Analysis RAG System - Main Application
Streamlit web interface for financial analysis with RAG capabilities
"""

import os
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Financial Analysis RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set environment variables to disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
os.environ["POSTHOG_DISABLED"] = "True"

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from data_ingestion import FinancialDataIngestion
from text_processing import FinancialTextProcessor
from vector_store import FinancialVectorStore
from llm_interface import FinancialLLMInterface
from analysis import FinancialAnalyzer

# Load environment variables
load_dotenv()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_components():
    """Initialize all components with caching"""
    try:
        # Initialize components
        data_ingestion = FinancialDataIngestion()
        text_processor = FinancialTextProcessor()
        vector_store = FinancialVectorStore()
        llm_interface = FinancialLLMInterface()
        analyzer = FinancialAnalyzer()

        return data_ingestion, text_processor, vector_store, llm_interface, analyzer
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None, None, None

def main():
    """Main application function"""

    # Header
    st.markdown('<h1 class="main-header">📊 Financial Analysis RAG System</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Financial Analysis with Multi-Source Data Integration")

    # Initialize components
    data_ingestion, text_processor, vector_store, llm_interface, analyzer = initialize_components()

    if not all([data_ingestion, text_processor, vector_store, llm_interface, analyzer]):
        st.error("Failed to initialize system components. Please check your configuration.")
        return

    # Sidebar
    with st.sidebar:
        st.header("🔧 System Configuration")

        # API Key check
        gemini_key = os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            st.error("⚠️ Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        else:
            st.success("✅ Gemini API key configured")

        # System stats
        st.subheader("📈 System Statistics")
        try:
            stats = vector_store.get_collection_stats()
            for collection_name, stat in stats.items():
                st.metric(f"{collection_name.replace('_', ' ').title()}", stat['document_count'])
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")

        # Reset collections
        if st.button("🔄 Reset All Collections"):
            if vector_store.reset_all_collections():
                st.success("Collections reset successfully!")
                st.rerun()
            else:
                st.error("Failed to reset collections")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 Data Ingestion", 
        "🔍 Query & Analysis", 
        "📊 Market Analysis", 
        "📰 News Sentiment", 
        "📋 System Info"
    ])

    # Tab 1: Data Ingestion
    with tab1:
        st.header("📥 Data Ingestion")
        st.markdown("Upload financial reports and fetch market data to build your knowledge base.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Financial Reports")

            # PDF upload
            uploaded_file = st.file_uploader(
                "Upload Financial Report (PDF)", 
                type=['pdf'],
                help="Upload SEC filings, annual reports, or other financial documents"
            )

            if uploaded_file is not None:
                if st.button("Process PDF"):
                    with st.spinner("Processing PDF..."):
                        try:
                            # Extract text from PDF
                            text = data_ingestion.process_uploaded_pdf(uploaded_file)

                            if text:
                                # Process and chunk text
                                chunks = text_processor.chunk_text(text, "financial_report")

                                # Add to vector store
                                if vector_store.add_documents(chunks, "reports"):
                                    st.success(f"✅ Successfully processed and stored {len(chunks)} document chunks")

                                    # Show sample chunks
                                    with st.expander("View Sample Chunks"):
                                        for i, chunk in enumerate(chunks[:3]):
                                            st.write(f"**Chunk {i+1}:**")
                                            st.write(chunk['content'][:300] + "...")
                                            st.divider()
                                else:
                                    st.error("Failed to store documents in vector database")
                            else:
                                st.error("Failed to extract text from PDF")
                        except Exception as e:
                            st.error(f"Error processing PDF: {str(e)}")

        with col2:
            st.subheader("📈 Market Data")

            # Stock ticker input
            ticker = st.text_input(
                "Stock Ticker Symbol",
                placeholder="e.g., AAPL, MSFT, GOOGL",
                help="Enter a valid stock ticker symbol"
            )

            if ticker:
                period = st.selectbox(
                    "Data Period",
                    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                    index=3
                )

                if st.button("Fetch Market Data"):
                    with st.spinner(f"Fetching data for {ticker}..."):
                        try:
                            # Fetch market data
                            market_data = data_ingestion.get_market_data(ticker, period)

                            if market_data and 'historical_data' in market_data:
                                # Convert to text for storage
                                market_text = f"""
                                Market Data for {ticker}:
                                Current Price: ${market_data.get('current_price', 'N/A')}
                                Market Cap: ${market_data.get('market_cap', 'N/A'):,.0f}
                                P/E Ratio: {market_data.get('pe_ratio', 'N/A')}
                                Beta: {market_data.get('beta', 'N/A')}

                                Historical Performance:
                                {market_data['historical_data'].tail(10).to_string()}
                                """

                                # Process and store
                                chunks = text_processor.chunk_text(market_text, "market_data")

                                if vector_store.add_documents(chunks, "market"):
                                    st.success(f"✅ Successfully stored market data for {ticker}")

                                    # Show market summary
                                    st.metric("Current Price", f"${market_data.get('current_price', 0):.2f}")
                                    st.metric("Market Cap", f"${market_data.get('market_cap', 0):,.0f}")
                                    st.metric("P/E Ratio", f"{market_data.get('pe_ratio', 0):.2f}")
                                else:
                                    st.error("Failed to store market data")
                            else:
                                st.error(f"Failed to fetch data for {ticker}")
                        except Exception as e:
                            st.error(f"Error fetching market data: {str(e)}")

            # News fetching
            st.subheader("📰 Financial News")

            news_ticker = st.text_input(
                "News Ticker Filter (Optional)",
                placeholder="e.g., AAPL",
                help="Filter news by specific ticker"
            )

            days_back = st.slider("Days Back", 1, 30, 7)

            if st.button("Fetch News"):
                with st.spinner("Fetching financial news..."):
                    try:
                        news_articles = data_ingestion.get_financial_news(news_ticker, days_back)

                        if news_articles:
                            # Process and store news
                            news_chunks = []
                            for article in news_articles:
                                article_text = f"Title: {article.get('title', '')}\nSummary: {article.get('summary', '')}\nFull Text: {article.get('full_text', '')}"
                                chunks = text_processor.chunk_text(article_text, "news")
                                news_chunks.extend(chunks)

                            if vector_store.add_documents(news_chunks, "news"):
                                st.success(f"✅ Successfully stored {len(news_articles)} news articles")

                                # Show recent news
                                with st.expander("Recent News"):
                                    for article in news_articles[:5]:
                                        st.write(f"**{article.get('title', 'No title')}**")
                                        st.write(f"*{article.get('source', 'Unknown source')} - {article.get('published_date', 'Unknown date')}*")
                                        st.write(article.get('summary', 'No summary')[:200] + "...")
                                        st.divider()
                            else:
                                st.error("Failed to store news articles")
                        else:
                            st.warning("No news articles found")
                    except Exception as e:
                        st.error(f"Error fetching news: {str(e)}")

    # Tab 2: Query & Analysis
    with tab2:
        st.header("🔍 Query & Analysis")
        st.markdown("Ask questions about your financial data and get AI-powered insights.")

        # Query input
        query = st.text_area(
            "Enter your question",
            placeholder="e.g., What are the key risk factors for AAPL based on recent news and financial reports?",
            height=100
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            response_type = st.selectbox(
                "Response Type",
                ["analysis", "summary", "risk_assessment"],
                help="Choose the type of analysis you want"
            )

        with col2:
            n_results = st.slider("Number of Results", 3, 15, 8)

        with col3:
            search_collection = st.selectbox(
                "Search Collection",
                ["all", "reports", "news", "market"],
                help="Choose which data sources to search"
            )

        if st.button("🔍 Generate Analysis", type="primary"):
            if query:
                with st.spinner("Searching and generating analysis..."):
                    try:
                        # Search for relevant documents
                        if search_collection == "all":
                            relevant_docs = vector_store.search_across_all_collections(query, n_results)
                        else:
                            relevant_docs = vector_store.search_documents(query, n_results, search_collection)

                        if relevant_docs:
                            # Generate response
                            response = llm_interface.generate_response(query, relevant_docs, response_type)

                            # Display results
                            st.subheader("📋 Analysis Results")

                            # Confidence indicator
                            confidence = response.get('confidence', 'unknown')
                            if confidence == 'high':
                                st.success("✅ High Confidence Analysis")
                            elif confidence == 'medium':
                                st.warning("⚠️ Medium Confidence Analysis")
                            else:
                                st.error("❌ Low Confidence Analysis")

                            # Main response
                            st.markdown("### Response")
                            st.write(response.get('response', 'No response generated'))

                            # Sources
                            st.markdown("### 📚 Sources Used")
                            sources = response.get('sources', [])
                            for i, source in enumerate(sources, 1):
                                with st.expander(f"Source {i}: {source.get('type', 'unknown').title()}"):
                                    st.write(f"**Type:** {source.get('type', 'unknown')}")
                                    st.write(f"**Relevance Score:** {source.get('relevance_score', 0):.3f}")
                                    if source.get('section'):
                                        st.write(f"**Section:** {source['section']}")

                            # Metadata
                            st.markdown("### 📊 Analysis Metadata")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Model Used", response.get('model_used', 'Unknown'))
                            with col2:
                                st.metric("Tokens Used", response.get('tokens_used', 'Unknown'))
                            with col3:
                                st.metric("Sources Found", len(sources))
                        else:
                            st.warning("No relevant documents found. Try rephrasing your question or adding more data.")

                    except Exception as e:
                        st.error(f"Error generating analysis: {str(e)}")
            else:
                st.warning("Please enter a question to analyze.")

    # Tab 3: Market Analysis
    with tab3:
        st.header("📊 Market Analysis")
        st.markdown("Comprehensive market analysis with technical indicators and risk metrics.")

        # Market data input
        analysis_ticker = st.text_input(
            "Stock Ticker for Analysis",
            placeholder="e.g., AAPL",
            help="Enter ticker symbol for detailed market analysis"
        )

        if analysis_ticker:
            if st.button("📊 Analyze Market"):
                with st.spinner(f"Analyzing {analysis_ticker}..."):
                    try:
                        # Fetch market data
                        market_data = data_ingestion.get_market_data(analysis_ticker, "1y")

                        if market_data and 'historical_data' in market_data:
                            # Perform analysis
                            analysis = analyzer.analyze_market_trends(market_data)
                            risk_metrics = analyzer.calculate_risk_metrics(market_data)
                            charts = analyzer.create_market_charts(market_data)

                            # Display results
                            st.subheader(f"📈 Market Analysis for {analysis_ticker}")

                            # Key metrics
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                if 'trends' in analysis and '20_day_change' in analysis['trends']:
                                    change = analysis['trends']['20_day_change']
                                    st.metric(
                                        "20-Day Change",
                                        f"{change['percentage']:.2f}%",
                                        delta=f"{change['percentage']:.2f}%"
                                    )

                            with col2:
                                if 'volatility' in analysis and 'current_volatility' in analysis['volatility']:
                                    vol = analysis['volatility']['current_volatility']
                                    st.metric("Volatility", f"{vol['value']:.2f}%")

                            with col3:
                                if 'technical_indicators' in analysis and 'rsi' in analysis['technical_indicators']:
                                    rsi = analysis['technical_indicators']['rsi']
                                    st.metric("RSI", f"{rsi['value']:.2f}")

                            with col4:
                                if risk_metrics and 'volatility' in risk_metrics:
                                    st.metric("Annual Volatility", f"{risk_metrics['volatility']['annualized']:.2f}%")

                            # Charts
                            st.subheader("📊 Market Charts")

                            if 'price_chart' in charts:
                                st.plotly_chart(charts['price_chart'], use_container_width=True)

                            col1, col2 = st.columns(2)

                            with col1:
                                if 'volume_chart' in charts:
                                    st.plotly_chart(charts['volume_chart'], use_container_width=True)

                            with col2:
                                if 'rsi_chart' in charts:
                                    st.plotly_chart(charts['rsi_chart'], use_container_width=True)

                            # Technical Analysis
                            st.subheader("🔧 Technical Analysis")

                            if 'trends' in analysis:
                                trends = analysis['trends']

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**Price Trends**")
                                    for period in ['5_day_change', '10_day_change', '20_day_change']:
                                        if period in trends:
                                            change = trends[period]
                                            st.write(f"{period.replace('_', ' ').title()}: {change['percentage']:.2f}% ({change['direction']})")

                                with col2:
                                    st.markdown("**Moving Averages**")
                                    if 'moving_averages' in trends:
                                        ma = trends['moving_averages']
                                        st.write(f"20-day SMA: ${ma['sma_20']:.2f}")
                                        st.write(f"50-day SMA: ${ma['sma_50']:.2f}")
                                        st.write(f"Golden Cross: {'Yes' if ma.get('golden_cross') else 'No'}")

                            # Risk Metrics
                            st.subheader("⚠️ Risk Metrics")

                            if risk_metrics and 'error' not in risk_metrics:
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("95% VaR", f"{risk_metrics['var_95']:.2f}%")
                                    st.metric("99% VaR", f"{risk_metrics['var_99']:.2f}%")

                                with col2:
                                    st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2f}%")
                                    st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.3f}")

                                with col3:
                                    st.metric("Beta", f"{risk_metrics['beta']}")

                            # Insights
                            if 'insights' in analysis and analysis['insights']:
                                st.subheader("💡 Market Insights")
                                for insight in analysis['insights']:
                                    st.write(f"• {insight}")
                        else:
                            st.error(f"Failed to fetch data for {analysis_ticker}")

                    except Exception as e:
                        st.error(f"Error analyzing market: {str(e)}")

    # Tab 4: News Sentiment
    with tab4:
        st.header("📰 News Sentiment Analysis")
        st.markdown("Analyze sentiment from financial news articles.")

        sentiment_ticker = st.text_input(
            "Ticker for Sentiment Analysis",
            placeholder="e.g., AAPL",
            help="Enter ticker to analyze news sentiment"
        )

        if sentiment_ticker:
            if st.button("📊 Analyze Sentiment"):
                with st.spinner(f"Analyzing sentiment for {sentiment_ticker}..."):
                    try:
                        # Fetch news
                        news_articles = data_ingestion.get_financial_news(sentiment_ticker, 30)

                        if news_articles:
                            # Analyze sentiment
                            sentiment_analysis = analyzer.analyze_news_sentiment(news_articles)

                            if 'error' not in sentiment_analysis:
                                st.subheader(f"📰 Sentiment Analysis for {sentiment_ticker}")

                                # Overall sentiment
                                overall = sentiment_analysis['overall_sentiment']
                                st.markdown("### Overall Sentiment")

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    sentiment_color = "green" if overall['sentiment'] == 'positive' else "red" if overall['sentiment'] == 'negative' else "gray"
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 1rem; background-color: {sentiment_color}; color: white; border-radius: 0.5rem;">
                                        <h3>{overall['sentiment'].title()}</h3>
                                        <p>Score: {overall['score']:.4f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                with col2:
                                    st.metric("Sentiment Score", f"{overall['score']:.4f}")

                                with col3:
                                    st.metric("Sentiment Volatility", f"{overall['volatility']:.4f}")

                                # Article breakdown
                                breakdown = sentiment_analysis['article_breakdown']
                                st.markdown("### Article Breakdown")

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("Positive Articles", breakdown['positive'])

                                with col2:
                                    st.metric("Negative Articles", breakdown['negative'])

                                with col3:
                                    st.metric("Neutral Articles", breakdown['neutral'])

                                # Individual article sentiments
                                st.markdown("### Individual Article Sentiments")

                                article_sentiments = sentiment_analysis['article_sentiments']
                                for article in article_sentiments[:10]:  # Show top 10
                                    with st.expander(f"{article['title'][:50]}..."):
                                        sentiment_color = "green" if article['sentiment'] == 'positive' else "red" if article['sentiment'] == 'negative' else "gray"
                                        st.markdown(f"""
                                        <span style="color: {sentiment_color}; font-weight: bold;">
                                            {article['sentiment'].title()}
                                        </span>
                                        """, unsafe_allow_html=True)
                                        st.write(f"**Score:** {article['sentiment_score']:.4f}")
                                        st.write(f"**Date:** {article['date']}")
                                        st.write(f"**Title:** {article['title']}")
                            else:
                                st.error(sentiment_analysis['error'])
                        else:
                            st.warning(f"No news articles found for {sentiment_ticker}")

                    except Exception as e:
                        st.error(f"Error analyzing sentiment: {str(e)}")

    # Tab 5: System Info
    with tab5:
        st.header("📋 System Information")
        st.markdown("System configuration and usage information.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔧 Configuration")

            # Environment variables
            st.markdown("**Environment Variables**")
            gemini_key = os.getenv('GEMINI_API_KEY')
            st.write(f"Gemini API Key: {'✅ Configured' if gemini_key else '❌ Not configured'}")

            news_api_key = os.getenv('NEWS_API_KEY')
            st.write(f"News API Key: {'✅ Configured' if news_api_key else '❌ Not configured'}")

            # Model information
            st.markdown("**Model Configuration**")
            st.write(f"Embedding Model: all-MiniLM-L6-v2")
            st.write(f"LLM Model: gemini-pro")
            st.write(f"Vector Database: ChromaDB")

        with col2:
            st.subheader("📊 System Statistics")

            try:
                stats = vector_store.get_collection_stats()

                for collection_name, stat in stats.items():
                    st.metric(
                        f"{collection_name.replace('_', ' ').title()}",
                        stat['document_count']
                    )
            except Exception as e:
                st.error(f"Error loading statistics: {str(e)}")

        # Usage instructions
        st.subheader("📖 Usage Instructions")

        st.markdown("""
        ### How to Use the System

        1. **Data Ingestion** (Tab 1):
           - Upload financial reports (PDFs)
           - Fetch market data for specific tickers
           - Collect financial news articles

        2. **Query & Analysis** (Tab 2):
           - Ask questions about your data
           - Choose response type (analysis, summary, risk assessment)
           - View AI-generated insights with sources

        3. **Market Analysis** (Tab 3):
           - Get comprehensive technical analysis
           - View interactive charts
           - Analyze risk metrics

        4. **News Sentiment** (Tab 4):
           - Analyze sentiment from financial news
           - Track sentiment trends over time

        ### Example Queries

        - "What are the key risk factors for AAPL?"
        - "Analyze the market sentiment for Tesla based on recent news"
        - "What are the main financial highlights from the latest 10-K report?"
        - "Compare the performance of tech stocks in the current market"
        """)

        # System requirements
        st.subheader("⚙️ System Requirements")

        st.markdown("""
        - **Python 3.8+**
        - **Gemini API Key** (required for LLM responses)
        - **Internet Connection** (for market data and news)
        - **Sufficient Storage** (for vector database)

        ### Dependencies

        All required packages are listed in `requirements.txt`. Install with:
        ```bash
        pip install -r requirements.txt
        ```
        """)

if __name__ == "__main__":
    main()