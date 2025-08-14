"""
Analysis Module for Financial Analysis RAG System
Handles time-series analysis, trend identification, and financial insights
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """Handles financial analysis and time-series processing"""
    
    def __init__(self):
        """Initialize the financial analyzer"""
        self.scaler = StandardScaler()
    
    def analyze_market_trends(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market trends from historical data
        
        Args:
            market_data: Market data dictionary with historical data
            
        Returns:
            Analysis results with trends and insights
        """
        try:
            if not market_data or 'historical_data' not in market_data:
                return {'error': 'No historical data available'}
            
            df = market_data['historical_data']
            if df.empty:
                return {'error': 'Historical data is empty'}
            
            analysis = {
                'ticker': market_data.get('ticker', 'Unknown'),
                'analysis_date': datetime.now().isoformat(),
                'trends': {},
                'volatility': {},
                'technical_indicators': {},
                'correlations': {},
                'insights': []
            }
            
            # Price trends
            analysis['trends'] = self._calculate_price_trends(df)
            
            # Volatility analysis
            analysis['volatility'] = self._analyze_volatility(df)
            
            # Technical indicators
            analysis['technical_indicators'] = self._calculate_technical_indicators(df)
            
            # Volume analysis
            analysis['volume_analysis'] = self._analyze_volume(df)
            
            # Generate insights
            analysis['insights'] = self._generate_trend_insights(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market trends: {str(e)}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _calculate_price_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various price trends"""
        trends = {}
        
        try:
            # Short-term trends (5, 10, 20 days)
            for period in [5, 10, 20]:
                if len(df) >= period:
                    current_price = df['Close'].iloc[-1]
                    past_price = df['Close'].iloc[-period]
                    change = ((current_price - past_price) / past_price) * 100
                    
                    trends[f'{period}_day_change'] = {
                        'percentage': round(change, 2),
                        'direction': 'up' if change > 0 else 'down',
                        'strength': 'strong' if abs(change) > 5 else 'moderate' if abs(change) > 2 else 'weak'
                    }
            
            # Moving averages
            if len(df) >= 50:
                sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
                sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
                current_price = df['Close'].iloc[-1]
                
                trends['moving_averages'] = {
                    'sma_20': round(sma_20, 2),
                    'sma_50': round(sma_50, 2),
                    'current_price': round(current_price, 2),
                    'above_sma_20': current_price > sma_20,
                    'above_sma_50': current_price > sma_50,
                    'golden_cross': sma_20 > sma_50 and len(df) >= 50
                }
            
            # Trend strength using linear regression
            if len(df) >= 30:
                x = np.arange(len(df))
                y = df['Close'].values
                slope, intercept = np.polyfit(x, y, 1)
                r_squared = np.corrcoef(x, y)[0, 1] ** 2
                
                trends['trend_strength'] = {
                    'slope': round(slope, 4),
                    'r_squared': round(r_squared, 4),
                    'trend_direction': 'upward' if slope > 0 else 'downward',
                    'trend_strength': 'strong' if r_squared > 0.7 else 'moderate' if r_squared > 0.4 else 'weak'
                }
            
        except Exception as e:
            logger.error(f"Error calculating price trends: {str(e)}")
        
        return trends
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price volatility"""
        volatility = {}
        
        try:
            # Calculate daily returns
            df['Daily_Return'] = df['Close'].pct_change()
            
            # Historical volatility (20-day rolling)
            if len(df) >= 20:
                rolling_vol = df['Daily_Return'].rolling(window=20).std()
                current_vol = rolling_vol.iloc[-1]
                avg_vol = rolling_vol.mean()
                
                volatility['current_volatility'] = {
                    'value': round(current_vol * 100, 2),  # Convert to percentage
                    'average': round(avg_vol * 100, 2),
                    'status': 'high' if current_vol > avg_vol * 1.5 else 'normal' if current_vol > avg_vol * 0.5 else 'low'
                }
            
            # Volatility clustering
            if len(df) >= 30:
                recent_vol = df['Daily_Return'].tail(10).std()
                earlier_vol = df['Daily_Return'].iloc[-30:-10].std()
                
                volatility['volatility_trend'] = {
                    'recent': round(recent_vol * 100, 2),
                    'earlier': round(earlier_vol * 100, 2),
                    'direction': 'increasing' if recent_vol > earlier_vol else 'decreasing'
                }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {str(e)}")
        
        return volatility
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        indicators = {}
        
        try:
            # RSI (Relative Strength Index)
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                current_rsi = rsi.iloc[-1]
                indicators['rsi'] = {
                    'value': round(current_rsi, 2),
                    'status': 'overbought' if current_rsi > 70 else 'oversold' if current_rsi < 30 else 'neutral'
                }
            
            # MACD
            if len(df) >= 26:
                ema_12 = df['Close'].ewm(span=12).mean()
                ema_26 = df['Close'].ewm(span=26).mean()
                macd = ema_12 - ema_26
                signal = macd.ewm(span=9).mean()
                
                current_macd = macd.iloc[-1]
                current_signal = signal.iloc[-1]
                
                indicators['macd'] = {
                    'macd_line': round(current_macd, 4),
                    'signal_line': round(current_signal, 4),
                    'histogram': round(current_macd - current_signal, 4),
                    'signal': 'bullish' if current_macd > current_signal else 'bearish'
                }
            
            # Bollinger Bands
            if len(df) >= 20:
                sma_20 = df['Close'].rolling(window=20).mean()
                std_20 = df['Close'].rolling(window=20).std()
                upper_band = sma_20 + (std_20 * 2)
                lower_band = sma_20 - (std_20 * 2)
                
                current_price = df['Close'].iloc[-1]
                current_upper = upper_band.iloc[-1]
                current_lower = lower_band.iloc[-1]
                
                indicators['bollinger_bands'] = {
                    'upper_band': round(current_upper, 2),
                    'lower_band': round(current_lower, 2),
                    'middle_band': round(sma_20.iloc[-1], 2),
                    'current_price': round(current_price, 2),
                    'position': 'above_upper' if current_price > current_upper else 'below_lower' if current_price < current_lower else 'within_bands'
                }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
        
        return indicators
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading volume patterns"""
        volume_analysis = {}
        
        try:
            if 'Volume' in df.columns:
                # Volume trends
                recent_volume = df['Volume'].tail(10).mean()
                avg_volume = df['Volume'].mean()
                
                volume_analysis['volume_trend'] = {
                    'recent_average': round(recent_volume, 0),
                    'historical_average': round(avg_volume, 0),
                    'ratio': round(recent_volume / avg_volume, 2),
                    'status': 'high' if recent_volume > avg_volume * 1.5 else 'normal' if recent_volume > avg_volume * 0.5 else 'low'
                }
                
                # Volume-price relationship
                price_change = df['Close'].pct_change()
                volume_change = df['Volume'].pct_change()
                
                # Calculate correlation
                correlation = price_change.corr(volume_change)
                volume_analysis['price_volume_correlation'] = {
                    'correlation': round(correlation, 3),
                    'relationship': 'positive' if correlation > 0.3 else 'negative' if correlation < -0.3 else 'weak'
                }
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {str(e)}")
        
        return volume_analysis
    
    def _generate_trend_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from analysis results"""
        insights = []
        
        try:
            trends = analysis.get('trends', {})
            volatility = analysis.get('volatility', {})
            indicators = analysis.get('technical_indicators', {})
            
            # Price trend insights
            if '20_day_change' in trends:
                change = trends['20_day_change']
                if change['direction'] == 'up' and change['strength'] == 'strong':
                    insights.append("Strong upward momentum over the past 20 days suggests bullish sentiment.")
                elif change['direction'] == 'down' and change['strength'] == 'strong':
                    insights.append("Significant downward pressure over the past 20 days indicates bearish sentiment.")
            
            # Moving average insights
            if 'moving_averages' in trends:
                ma = trends['moving_averages']
                if ma.get('golden_cross'):
                    insights.append("Golden cross detected (20-day SMA above 50-day SMA) - bullish technical signal.")
                elif ma.get('current_price', 0) < ma.get('sma_20', 0):
                    insights.append("Price below 20-day moving average suggests short-term bearish pressure.")
            
            # Volatility insights
            if 'current_volatility' in volatility:
                vol = volatility['current_volatility']
                if vol['status'] == 'high':
                    insights.append("Elevated volatility suggests increased market uncertainty and potential trading opportunities.")
                elif vol['status'] == 'low':
                    insights.append("Low volatility indicates stable market conditions with reduced risk.")
            
            # RSI insights
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi['status'] == 'overbought':
                    insights.append("RSI indicates overbought conditions - potential for price correction.")
                elif rsi['status'] == 'oversold':
                    insights.append("RSI shows oversold conditions - potential for price rebound.")
            
            # MACD insights
            if 'macd' in indicators:
                macd = indicators['macd']
                if macd['signal'] == 'bullish':
                    insights.append("MACD bullish signal suggests upward momentum.")
                else:
                    insights.append("MACD bearish signal indicates downward momentum.")
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
        
        return insights
    
    def analyze_news_sentiment(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment from news articles
        
        Args:
            news_articles: List of news articles
            
        Returns:
            Sentiment analysis results
        """
        try:
            if not news_articles:
                return {'error': 'No news articles provided'}
            
            # Simple keyword-based sentiment analysis
            positive_keywords = [
                'positive', 'growth', 'increase', 'profit', 'gain', 'up', 'higher',
                'strong', 'excellent', 'outperform', 'beat', 'surge', 'rally'
            ]
            
            negative_keywords = [
                'negative', 'decline', 'decrease', 'loss', 'down', 'lower',
                'weak', 'poor', 'underperform', 'miss', 'fall', 'drop'
            ]
            
            sentiment_scores = []
            article_sentiments = []
            
            for article in news_articles:
                title = article.get('title', '').lower()
                summary = article.get('summary', '').lower()
                full_text = article.get('full_text', '').lower()
                
                text = f"{title} {summary} {full_text}"
                
                positive_count = sum(1 for keyword in positive_keywords if keyword in text)
                negative_count = sum(1 for keyword in negative_keywords if keyword in text)
                
                # Calculate sentiment score (-1 to 1)
                total_words = len(text.split())
                if total_words > 0:
                    sentiment_score = (positive_count - negative_count) / total_words
                else:
                    sentiment_score = 0
                
                sentiment_scores.append(sentiment_score)
                
                article_sentiments.append({
                    'title': article.get('title', ''),
                    'sentiment_score': round(sentiment_score, 4),
                    'sentiment': 'positive' if sentiment_score > 0.001 else 'negative' if sentiment_score < -0.001 else 'neutral',
                    'date': article.get('published_date', '')
                })
            
            # Overall sentiment analysis
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            sentiment_std = np.std(sentiment_scores) if sentiment_scores else 0
            
            positive_articles = sum(1 for score in sentiment_scores if score > 0.001)
            negative_articles = sum(1 for score in sentiment_scores if score < -0.001)
            neutral_articles = len(sentiment_scores) - positive_articles - negative_articles
            
            return {
                'overall_sentiment': {
                    'score': round(avg_sentiment, 4),
                    'sentiment': 'positive' if avg_sentiment > 0.001 else 'negative' if avg_sentiment < -0.001 else 'neutral',
                    'volatility': round(sentiment_std, 4)
                },
                'article_breakdown': {
                    'positive': positive_articles,
                    'negative': negative_articles,
                    'neutral': neutral_articles,
                    'total': len(news_articles)
                },
                'article_sentiments': article_sentiments,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {'error': f'Sentiment analysis failed: {str(e)}'}
    
    def create_market_charts(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create interactive market charts
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Dictionary with chart objects
        """
        try:
            if not market_data or 'historical_data' not in market_data:
                return {'error': 'No historical data available'}
            
            df = market_data['historical_data']
            if df.empty:
                return {'error': 'Historical data is empty'}
            
            charts = {}
            
            # Price chart with moving averages
            fig_price = go.Figure()
            
            fig_price.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            
            if len(df) >= 20:
                sma_20 = df['Close'].rolling(window=20).mean()
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=sma_20,
                    mode='lines',
                    name='20-day SMA',
                    line=dict(color='orange', width=1)
                ))
            
            if len(df) >= 50:
                sma_50 = df['Close'].rolling(window=50).mean()
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=sma_50,
                    mode='lines',
                    name='50-day SMA',
                    line=dict(color='red', width=1)
                ))
            
            fig_price.update_layout(
                title=f"{market_data.get('ticker', 'Stock')} Price Chart",
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified'
            )
            
            charts['price_chart'] = fig_price
            
            # Volume chart
            if 'Volume' in df.columns:
                fig_volume = go.Figure()
                
                fig_volume.add_trace(go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ))
                
                fig_volume.update_layout(
                    title=f"{market_data.get('ticker', 'Stock')} Volume",
                    xaxis_title='Date',
                    yaxis_title='Volume',
                    hovermode='x unified'
                )
                
                charts['volume_chart'] = fig_volume
            
            # RSI chart
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                fig_rsi = go.Figure()
                
                fig_rsi.add_trace(go.Scatter(
                    x=df.index,
                    y=rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ))
                
                # Add overbought/oversold lines
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                
                fig_rsi.update_layout(
                    title=f"{market_data.get('ticker', 'Stock')} RSI",
                    xaxis_title='Date',
                    yaxis_title='RSI',
                    yaxis=dict(range=[0, 100]),
                    hovermode='x unified'
                )
                
                charts['rsi_chart'] = fig_rsi
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating charts: {str(e)}")
            return {'error': f'Chart creation failed: {str(e)}'}
    
    def calculate_risk_metrics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk metrics for the stock
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Risk metrics
        """
        try:
            if not market_data or 'historical_data' not in market_data:
                return {'error': 'No historical data available'}
            
            df = market_data['historical_data']
            if df.empty:
                return {'error': 'Historical data is empty'}
            
            # Calculate daily returns
            df['Daily_Return'] = df['Close'].pct_change()
            
            # Remove NaN values
            returns = df['Daily_Return'].dropna()
            
            if len(returns) < 30:
                return {'error': 'Insufficient data for risk calculation'}
            
            # Risk metrics
            risk_metrics = {
                'volatility': {
                    'annualized': round(returns.std() * np.sqrt(252) * 100, 2),  # Annualized volatility
                    'daily': round(returns.std() * 100, 2)
                },
                'var_95': round(np.percentile(returns, 5) * 100, 2),  # 95% VaR
                'var_99': round(np.percentile(returns, 1) * 100, 2),  # 99% VaR
                'max_drawdown': self._calculate_max_drawdown(df['Close']),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'beta': market_data.get('beta', 'N/A'),
                'analysis_date': datetime.now().isoformat()
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {'error': f'Risk calculation failed: {str(e)}'}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = prices.expanding(min_periods=1).max()
            drawdown = (prices - peak) / peak
            max_drawdown = drawdown.min()
            return round(max_drawdown * 100, 2)
        except:
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            if returns.std() == 0:
                return 0.0
            sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
            return round(sharpe, 3)
        except:
            return 0.0
