import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="DASHBOARD ANALISIS EMAS",
    page_icon="💫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Neon Gold Theme CSS
st.markdown("""
<style>
    /* Import Futuristic Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Audiowide&display=swap');
    
    /* Neon Gold Theme Variables */
    :root {
        --neon-gold: #FFD700;
        --neon-gold-bright: #FFFF00;
        --neon-gold-dark: #DAA520;
        --neon-orange: #FF8C00;
        --dark-bg: #0a0a0a;
        --darker-bg: #000000;
        --accent-blue: #00BFFF;
        --neon-green: #00FF41;
        --neon-pink: #FF1493;
    }
    
    /* Global Background */
    .stApp {
        background:"white";
        background-attachment: fixed;
    }
    
    /* Main Header with Neon Effect */
    .neon-header {
        font-family: 'Audiowide', cursive;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        margin: 2rem 0;
        color: #FFD700;
        text-shadow: 
            0 0 5px #FFD700,
            0 0 10px #FFD700,
            0 0 15px #FFD700,
            0 0 20px #FFD700,
            0 0 35px #FFD700,
            0 0 40px #FFD700;
        animation: neonGlow 1.5s ease-in-out infinite alternate;
        letter-spacing: 3px;
    }
    
    @keyframes neonGlow {
        from {
            text-shadow: 
                0 0 5px #FFD700,
                0 0 10px #FFD700,
                0 0 15px #FFD700,
                0 0 20px #FFD700,
                0 0 35px #FFD700,
                0 0 40px #FFD700;
        }
        to {
            text-shadow: 
                0 0 2px #FFD700,
                0 0 5px #FFD700,
                0 0 8px #FFD700,
                0 0 12px #FFD700,
                0 0 25px #FFD700,
                0 0 30px #FFD700;
        }
    }
    
    /* Team Info Card - Neon Style 
    .neon-team-card {
        background: linear-gradient(135deg, #000000, #0a0a0a);
        border: 2px solid #FFD700;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 
            0 0 20px rgba(255, 215, 0, 0.5),
            inset 0 0 20px rgba(255, 215, 0, 0.1);
        font-family: 'Rajdhani', sans-serif;
        color: #FFD700;
        position: relative;
        overflow: hidden;
    }*/
    
    .neon-team-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 215, 0, 0.2), 
            transparent);
        animation: neonSweep 3s infinite;
    }
    
    @keyframes neonSweep {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Sidebar Styling 
    .css-1d391kg, .css-1aumxhk {
        background: linear-gradient(180deg, #000000, #0a0a0a) !important;
        border-right: 2px solid #FFD700;
    }*/
    
    /* Metric Cards with Neon Glow 
    .metric-container {
        background: linear-gradient(145deg, #0a0a0a, #1a1a1a);
        border: 1px solid #FFD700;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 
            0 0 15px rgba(255, 215, 0, 0.3),
            inset 0 1px 0 rgba(255, 215, 0, 0.2);
        font-family: 'Orbitron', monospace;
        color: #FFD700;
        transition: all 0.3s ease;
    }*/
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 0 25px rgba(255, 215, 0, 0.5),
            inset 0 1px 0 rgba(255, 215, 0, 0.3);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', monospace !important;
        color: #FFD700 !important;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    /* Streamlit Components 
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: linear-gradient(145deg, #0a0a0a, #1a1a1a) !important;
        border: 1px solid #FFD700 !important;
        color: #FFD700 !important;
        border-radius: 8px;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #FFD700, #FF8C00) !important;
    }*/
    
    .stCheckbox > label {
        color: #FFD700 !important;
        font-family: 'Rajdhani', sans-serif !important;
    }
    
    /* Success/Info Messages */
    .stSuccess, .stInfo {
        background: linear-gradient(90deg, 
            rgba(255, 215, 0, 0.1), 
            rgba(255, 140, 0, 0.1)) !important;
        border: 1px solid #FFD700 !important;
        border-radius: 8px;
        color: #FFD700 !important;
    }
    
    /* Data Tables */
    .stDataFrame {
        border: 1px solid #FFD700;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #FFD700, #FF8C00) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px;
        font-family: 'Orbitron', monospace !important;
        font-weight: bold;
        text-transform: uppercase;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(255, 215, 0, 0.6);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #FFD700, #FF8C00);
        border-radius: 6px;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #FFD700;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #FFD700 transparent #FFD700 transparent !important;
    }
    
    /* Section Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #FFD700, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Neon Header
st.markdown('<h1 >💫 DASHBOARD ANALISIS EMAS🚀</h1>', unsafe_allow_html=True)

# Team Information Card
team_members = [
    {"name": "Ikhsanudin", "nim": "23.11.5502"},
    {"name": "Ikhsanuddin Ahmad Fauzi", "nim": "23.11.5506"},
    {"name": "Muhammad Zaidan Elha Rasyad", "nim": "23.11.5519"},
    {"name": "Wahid Nurrohim", "nim": "23.11.5521"}
]

st.markdown('<div class="neon-team-card">', unsafe_allow_html=True)
st.markdown("### 👨‍💻 DEVELOPMENT TEAM")
cols = st.columns(4)
for i, member in enumerate(team_members):
    with cols[i]:
        st.markdown(f"**⚡ {member['name']}**")
        st.markdown(f"🎯 `{member['nim']}`")
st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Data Processing Functions
@st.cache_data
def clean_price_format_robust(price):
    """Clean price format to handle various formats like 384.08.00"""
    if isinstance(price, (int, float)):
        return price
    if not isinstance(price, str):
        try:
            return float(price)
        except:
            return np.nan

    price = price.strip()
    
    # Handle multiple decimal points
    if price.count('.') > 1:
        parts = price.split('.')
        if len(parts) >= 2:
            cleaned_price = f"{parts[0]}.{parts[1]}"
            try:
                return float(cleaned_price)
            except:
                return np.nan

    # Handle comma as decimal separator
    if ',' in price and '.' not in price:
        price = price.replace(',', '.')
    
    try:
        return float(price)
    except ValueError:
        return np.nan

@st.cache_data
def convert_volume_to_numeric(volume_str):
    """Convert volume strings like '1.5K', '2.3M' to numeric"""
    if isinstance(volume_str, (int, float)):
        return volume_str
    if isinstance(volume_str, str):
        volume_str = volume_str.strip().upper()
        if 'K' in volume_str:
            return float(volume_str.replace('K', '')) * 1000
        elif 'M' in volume_str:
            return float(volume_str.replace('M', '')) * 1000000
        elif 'B' in volume_str:
            return float(volume_str.replace('B', '')) * 1000000000
    try:
        return float(volume_str)
    except:
        return np.nan

@st.cache_data
def calculate_rsi(prices, window=14):
    """Calculate RSI exactly as in Colab"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data
def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

@st.cache_data
def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

@st.cache_data
def load_data_from_file():
    """Load and process data exactly as in Colab"""
    try:
        # Try different possible file paths
        possible_paths = [
            'data/EMAS(XAU)_1W_data.csv',
            'EMAS(XAU)_1W_data.csv',
            'EMASXAU_1W_data.csv',
            'emas_data.csv'
        ]
        
        df = None
        for path in possible_paths:
            try:
                # Read the file
                df = pd.read_csv(path, sep=';', skiprows=6)
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            # Create sample data if file not found
            st.warning("⚠️ Data file not found. Using sample data for demonstration.")
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='W')
            np.random.seed(42)
            
            # Generate realistic gold price data
            n = len(dates)
            trend = np.linspace(1800, 2000, n)
            noise = np.random.normal(0, 50, n)
            seasonal = 30 * np.sin(2 * np.pi * np.arange(n) / 52)
            
            close_prices = trend + noise + seasonal
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': close_prices + np.random.normal(0, 5, n),
                'High': close_prices + np.abs(np.random.normal(10, 5, n)),
                'Low': close_prices - np.abs(np.random.normal(10, 5, n)),
                'Close': close_prices,
                'Volume': np.random.randint(10000, 100000, n)
            })
        
        # Clean the data
        initial_rows = len(df)
        
        # Fix date format
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            # If no Date column, create one
            df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='W')
        
        # Clean price columns
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_price_format_robust)
        
        # Fix volume format
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].apply(convert_volume_to_numeric)
        
        # Remove invalid data
        df.dropna(inplace=True)
        cleaned_rows = len(df)
        
        # Set index
        df = df.set_index('Date').sort_index()
        
        # Calculate technical indicators (exactly as in Colab)
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        
        # RSI
        df['RSI'] = calculate_rsi(df['Close'])
        
        # Price changes and volatility
        df['Price_Change'] = df['Close'].pct_change() * 100
        df['Volatility'] = df['Close'].rolling(20).std()
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'])
        
        # Support and Resistance
        df['Support'] = df['Low'].rolling(50).min()
        df['Resistance'] = df['High'].rolling(50).max()
        
        return df, initial_rows - cleaned_rows
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, 0

# Sidebar with Neon Styling
st.sidebar.markdown(" ## CONTROL PANEL")


# Load data
with st.spinner('🔄 Loading neon gold data...'):
    df, removed_rows = load_data_from_file()

if df is not None:
    st.success(f"✅ Data loaded successfully! {len(df):,} data points from {df.index.year.min()} - {df.index.year.max()}")
    if removed_rows > 0:
        st.info(f"📝 {removed_rows} invalid rows were removed.")
    
    # Enhanced Sidebar Controls
    st.sidebar.markdown("###  ANALYSIS SETTINGS")
    
    # Date range selector
    min_year = int(df.index.year.min())
    max_year = int(df.index.year.max())
    
    date_range = st.sidebar.slider(
        "📅 TIME RANGE",
        min_value=min_year,
        max_value=max_year,
        value=(max(min_year, max_year - 3), max_year),
        help="Select analysis time range"
    )
    
    # Filter data
    df_filtered = df[
        (df.index.year >= date_range[0]) & 
        (df.index.year <= date_range[1])
    ]
    
    # Analysis type selector
    analysis_type = st.sidebar.selectbox(
        "🎯 ANALYSIS MODULE",
        [
            "📊 MAIN DASHBOARD", 
            "📈 TREND ANALYSIS", 
            "🕯️ CANDLESTICK CHART", 
            "🧮 CORRELATION MATRIX", 
            "🤖 MACHINE LEARNING",
            "📊 TECHNICAL ANALYSIS",
            "🎯 BOLLINGER BANDS",
            "⚡ MACD ANALYSIS"
        ],
        help="Select analysis module"
    )
    
    # Display period info
    st.sidebar.markdown("### 📊 DATA INFO")
    st.sidebar.metric("📅 Start Year", date_range[0])
    st.sidebar.metric("📅 End Year", date_range[1])
    st.sidebar.metric("📊 Data Points", f"{len(df_filtered):,}")
    st.sidebar.metric("📈 Price Range", f"${df_filtered['Low'].min():.0f} - ${df_filtered['High'].max():.0f}")

    # Main Analysis Modules
    if analysis_type == "📊 MAIN DASHBOARD":
        st.header("💫 NEON GOLD MAIN DASHBOARD")
        
        # Enhanced Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_price = df_filtered['Close'].iloc[-1]
        prev_price = df_filtered['Close'].iloc[-2] if len(df_filtered) > 1 else current_price
        price_change_1w = ((current_price / prev_price) - 1) * 100
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                "💰 CURRENT PRICE",
                f"${current_price:,.2f}",
                delta=f"{price_change_1w:+.2f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                "📈 ALL-TIME HIGH", 
                f"${df_filtered['High'].max():,.2f}",
                delta=f"{((df_filtered['High'].max() / current_price) - 1) * 100:+.1f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                "📉 ALL-TIME LOW",
                f"${df_filtered['Low'].min():,.2f}",
                delta=f"{((current_price / df_filtered['Low'].min()) - 1) * 100:+.1f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            avg_volume = df_filtered['Volume'].mean()
            recent_volume = df_filtered['Volume'].iloc[-5:].mean()
            volume_change = ((recent_volume / avg_volume) - 1) * 100
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                "📊 AVG VOLUME",
                f"{avg_volume:,.0f}",
                delta=f"{volume_change:+.1f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            volatility = df_filtered['Price_Change'].std()
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(
                "⚡ VOLATILITY",
                f"{volatility:.2f}%",
                delta="Std Dev"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Interactive Chart
        st.subheader("📈 Interactive Price Chart")
        
        # Chart options
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_type = st.selectbox("Chart Type", ["Line", "Candlestick", "OHLC"])
        with col2:
            timeframe = st.selectbox("Timeframe", ["3 Months", "6 Months", "1 Year", "All Data"])
        with col3:
            show_ma = st.selectbox("Moving Averages", ["None", "MA20+MA50", "All MAs"])
        
        # Filter data based on timeframe
        if timeframe == "3 Months":
            chart_data = df_filtered.tail(13)
        elif timeframe == "6 Months":
            chart_data = df_filtered.tail(26)
        elif timeframe == "1 Year":
            chart_data = df_filtered.tail(52)
        else:
            chart_data = df_filtered
        
        # Create chart based on type
        fig = go.Figure()
        
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name='Gold Price',
                increasing_line_color='#00FF41',
                decreasing_line_color='#FF1493'
            ))
        elif chart_type == "OHLC":
            fig.add_trace(go.Ohlc(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name='Gold Price'
            ))
        else:  # Line chart
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['Close'],
                name='Gold Price',
                line=dict(color='#FFD700', width=3),
                hovertemplate='<b>%{x}</b><br>Price: $%{y:,.2f}<extra></extra>'
            ))
        
        # Add moving averages
        if show_ma == "MA20+MA50":
            if len(chart_data) > 20:
                fig.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data['MA20'],
                    name='MA 20',
                    line=dict(color='#FF8C00', width=2, dash='dash')
                ))
            if len(chart_data) > 50:
                fig.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data['MA50'],
                    name='MA 50',
                    line=dict(color='#00BFFF', width=2, dash='dash')
                ))
        elif show_ma == "All MAs":
            ma_configs = [
                ('MA10', '#00FF41', 10),
                ('MA20', '#FF8C00', 20),
                ('MA50', '#00BFFF', 50),
                ('MA200', '#FF1493', 200)
            ]
            for ma_name, color, period in ma_configs:
                if len(chart_data) > period:
                    fig.add_trace(go.Scatter(
                        x=chart_data.index,
                        y=chart_data[ma_name],
                        name=ma_name,
                        line=dict(color=color, width=2, dash='dash'),
                        opacity=0.8
                    ))
        
        fig.update_layout(
            title='Gold Price Analysis',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=600,
            hovermode='x unified',
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Statistical Summary")
            summary_stats = df_filtered[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
            st.dataframe(
                summary_stats.style.format({
                    'Open': '${:,.2f}',
                    'High': '${:,.2f}',
                    'Low': '${:,.2f}',
                    'Close': '${:,.2f}',
                    'Volume': '{:,.0f}'
                })
            )
        
        with col2:
            st.subheader("🔥 Recent Data")
            recent_data = df_filtered[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
            st.dataframe(
                recent_data.style.format({
                    'Open': '${:,.2f}',
                    'High': '${:,.2f}',
                    'Low': '${:,.2f}',
                    'Close': '${:,.2f}',
                    'Volume': '{:,.0f}'
                })
            )

    elif analysis_type == "📈 TREND ANALYSIS":
        st.header("📈 ADVANCED TREND ANALYSIS")
        
        # Trend Indicators
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate various trend metrics
        trend_1w = ((df_filtered['Close'].iloc[-1] / df_filtered['Close'].iloc[-2]) - 1) * 100 if len(df_filtered) >= 2 else 0
        trend_1m = ((df_filtered['Close'].iloc[-1] / df_filtered['Close'].iloc[-5]) - 1) * 100 if len(df_filtered) >= 5 else 0
        trend_3m = ((df_filtered['Close'].iloc[-1] / df_filtered['Close'].iloc[-13]) - 1) * 100 if len(df_filtered) >= 13 else 0
        
        with col1:
            st.metric("📊 1W Trend", f"{trend_1w:+.2f}%", "Bullish" if trend_1w > 0 else "Bearish")
        with col2:
            st.metric("📈 1M Trend", f"{trend_1m:+.2f}%", "Bullish" if trend_1m > 0 else "Bearish")
        with col3:
            st.metric("🎯 3M Trend", f"{trend_3m:+.2f}%", "Bullish" if trend_3m > 0 else "Bearish")
        with col4:
            current_rsi = df_filtered['RSI'].iloc[-1] if not pd.isna(df_filtered['RSI'].iloc[-1]) else 50
            rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            st.metric("⚡ RSI Signal", f"{current_rsi:.1f}", rsi_signal)
        
        # Trend Chart with All Indicators
        fig = go.Figure()
        
        # Main price line
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered['Close'],
            name='Close Price',
            line=dict(color='#FFD700', width=3)
        ))
        
        # Moving averages
        ma_configs = [
            ('MA10', '#00FF41', 10),
            ('MA20', '#FF8C00', 20),
            ('MA50', '#00BFFF', 50)
        ]
        
        for ma_name, color, period in ma_configs:
            if len(df_filtered) > period:
                fig.add_trace(go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered[ma_name],
                    name=ma_name,
                    line=dict(color=color, width=2, dash='dash'),
                    opacity=0.8
                ))
        
        fig.update_layout(
            title='Trend Analysis with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=600,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price Distribution Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                df_filtered.dropna(),
                x='Price_Change',
                nbins=50,
                title='Daily Price Change Distribution',
                template="plotly_dark"
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#FFD700")
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_vol = px.line(
                df_filtered.reset_index(),
                x='Date',
                y='Volatility',
                title='Price Volatility Over Time',
                template="plotly_dark"
            )
            fig_vol.update_traces(line_color='#FF8C00')
            fig_vol.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_vol, use_container_width=True)

    elif analysis_type == "🕯️ CANDLESTICK CHART":
        st.header("🕯️ ADVANCED CANDLESTICK ANALYSIS")
        
        # Enhanced Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            period = st.selectbox("Period", ["1 Month", "3 Months", "6 Months", "1 Year", "All"])
        with col2:
            show_volume = st.checkbox("Show Volume", True)
        with col3:
            show_indicators = st.checkbox("Technical Indicators", True)
        
        # Filter data
        if period == "1 Month":
            chart_data = df_filtered.tail(4)
        elif period == "3 Months":
            chart_data = df_filtered.tail(13)
        elif period == "6 Months":
            chart_data = df_filtered.tail(26)
        elif period == "1 Year":
            chart_data = df_filtered.tail(52)
        else:
            chart_data = df_filtered
        
        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=('Gold Price', 'Volume')
            )
        else:
            fig = make_subplots(rows=1, cols=1)
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name="Gold Price",
                increasing_line_color='#00FF41',
                decreasing_line_color='#FF1493',
                increasing_fillcolor='#00FF41',
                decreasing_fillcolor='#FF1493'
            ),
            row=1, col=1
        )
        
        # Add technical indicators
        if show_indicators:
            if len(chart_data) > 20:
                fig.add_trace(
                    go.Scatter(
                        x=chart_data.index,
                        y=chart_data['MA20'],
                        name='MA 20',
                        line=dict(color='#FFD700', width=2)
                    ),
                    row=1, col=1
                )
            
            if len(chart_data) > 50:
                fig.add_trace(
                    go.Scatter(
                        x=chart_data.index,
                        y=chart_data['MA50'],
                        name='MA 50',
                        line=dict(color='#00BFFF', width=2)
                    ),
                    row=1, col=1
                )
        
        # Volume chart
        if show_volume:
            colors = ['#00FF41' if close >= open else '#FF1493' 
                     for close, open in zip(chart_data['Close'], chart_data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=chart_data.index,
                    y=chart_data['Volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f'Gold Candlestick Chart - {period}',
            xaxis_rangeslider_visible=False,
            height=700 if show_volume else 500,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Candlestick Pattern Analysis
        st.subheader("📊 Candlestick Pattern Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        bullish_count = (chart_data['Close'] > chart_data['Open']).sum()
        bearish_count = (chart_data['Close'] < chart_data['Open']).sum()
        doji_count = (abs(chart_data['Close'] - chart_data['Open']) < (chart_data['High'] - chart_data['Low']) * 0.1).sum()
        
        with col1:
            st.metric("🟢 Bullish Candles", bullish_count, f"{(bullish_count/len(chart_data))*100:.1f}%")
        with col2:
            st.metric("🔴 Bearish Candles", bearish_count, f"{(bearish_count/len(chart_data))*100:.1f}%")
        with col3:
            st.metric("⚪ Doji Candles", doji_count, f"{(doji_count/len(chart_data))*100:.1f}%")
        with col4:
            avg_body = abs(chart_data['Close'] - chart_data['Open']).mean()
            st.metric("📏 Avg Body Size", f"${avg_body:.2f}", "Price units")

    elif analysis_type == "🧮 CORRELATION MATRIX":
        st.header("🧮 ADVANCED CORRELATION ANALYSIS")
        
        # Prepare correlation data (exactly as in Colab)
        correlation_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'MA50']
        available_columns = [col for col in correlation_columns if col in df_filtered.columns]
        correlation_data = df_filtered[available_columns].dropna()
        correlation_matrix = correlation_data.corr()
        
        # Interactive Heatmap (Viridis colorscale like Colab)
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix Heatmap",
            color_continuous_scale="Viridis",  # Same as Colab
            zmin=-1, zmax=1
        )
        fig.update_layout(
            height=600,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Insights
        st.subheader("🔍 Correlation Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔥 Highest Correlations:**")
            # Get upper triangle to avoid duplicates
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            high_corr = upper_triangle.unstack().sort_values(ascending=False).head(5)
            
            for idx, corr in high_corr.items():
                if not pd.isna(corr):
                    strength = "Very Strong" if abs(corr) > 0.8 else "Strong" if abs(corr) > 0.6 else "Moderate"
                    st.write(f"• **{idx[0]}** ↔ **{idx[1]}**: `{corr:.3f}` ({strength})")
        
        with col2:
            st.markdown("**❄️ Lowest Correlations:**")
            low_corr = upper_triangle.unstack().sort_values(ascending=True).head(5)
            
            for idx, corr in low_corr.items():
                if not pd.isna(corr):
                    strength = "Weak" if abs(corr) < 0.3 else "Moderate"
                    st.write(f"• **{idx[0]}** ↔ **{idx[1]}**: `{corr:.3f}` ({strength})")
        
        # Scatter Plot Matrix
        st.subheader("📊 Scatter Plot Matrix")
        
        selected_vars = st.multiselect(
            "Select variables for scatter plot analysis:",
            available_columns,
            default=available_columns[:4] if len(available_columns) >= 4 else available_columns
        )
        
        if len(selected_vars) >= 2:
            # Sample data for performance
            sample_size = min(1000, len(correlation_data))
            sample_data = correlation_data[selected_vars].sample(sample_size, random_state=42)
            
            fig_scatter = px.scatter_matrix(
                sample_data,
                title=f"Scatter Plot Matrix (n={sample_size})",
                height=600
            )
            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    elif analysis_type == "🤖 MACHINE LEARNING":
        st.header("🤖 AI-POWERED PRICE PREDICTION")
        
        # Feature Selection
        available_features = ['Open', 'High', 'Low', 'Volume', 'MA10', 'MA20', 'MA50']
        available_features = [f for f in available_features if f in df_filtered.columns]
        
        selected_features = st.multiselect(
            "🎯 Select Features for ML Model:",
            available_features,
            default=['Open', 'High', 'Low', 'Volume'] if len(available_features) >= 4 else available_features[:3]
        )
        
        if len(selected_features) == 0:
            st.error("⚠️ Please select at least one feature!")
            st.stop()
        
        # ML Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        with col2:
            random_state = st.number_input("Random State", 1, 100, 42)
        with col3:
            model_type = st.selectbox("Model Type", ["Linear Regression", "Ridge", "Lasso"])
        
        # Prepare data
        df_model = df_filtered[selected_features + ['Close']].dropna()
        
        if len(df_model) < 10:
            st.error("⚠️ Insufficient data for machine learning!")
            st.stop()
        
        X = df_model[selected_features]
        y = df_model['Close']
        
        # Train model
        with st.spinner('🤖 Training AI model...'):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Choose model
            if model_type == "Ridge":
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
            elif model_type == "Lasso":
                from sklearn.linear_model import Lasso
                model = Lasso(alpha=1.0)
            else:
                model = LinearRegression()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = metrics.r2_score(y_test, y_pred)
            rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Model Performance
        st.subheader("📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 R² Score", f"{r2:.4f}", f"{r2:.2%} variance explained")
        with col2:
            st.metric("📏 RMSE", f"${rmse:.2f}", "Root Mean Square Error")
        with col3:
            st.metric("📐 MAE", f"${mae:.2f}", "Mean Absolute Error")
        with col4:
            st.metric("📊 MAPE", f"{mape:.2f}%", "Mean Absolute % Error")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            fig_pred = go.Figure()
            
            fig_pred.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(
                    color=np.abs(y_test - y_pred),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Absolute Error")
                ),
                hovertemplate='Actual: $%{x:,.2f}<br>Predicted: $%{y:,.2f}<extra></extra>'
            ))
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='#FFD700', dash='dash', width=2)
            ))
            
            fig_pred.update_layout(
                title="Actual vs Predicted Prices",
                xaxis_title="Actual Price (USD)",
                yaxis_title="Predicted Price (USD)",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col2:
            # Feature Importance
            if hasattr(model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'Feature': selected_features,
                    'Coefficient': model.coef_,
                    'Abs_Coefficient': np.abs(model.coef_)
                }).sort_values('Abs_Coefficient', ascending=True)
                
                fig_importance = px.bar(
                    feature_importance,
                    x='Abs_Coefficient',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance',
                    color='Coefficient',
                    color_continuous_scale='RdBu_r',
                    template="plotly_dark"
                )
                fig_importance.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
        
        
        
        # Recent Predictions
        st.subheader("🔮 Recent Predictions Sample")
        
        sample_size = min(10, len(y_test))
        recent_indices = y_test.tail(sample_size).index
        
        prediction_df = pd.DataFrame({
            'Date': recent_indices,
            'Actual Price': y_test.tail(sample_size).values,
            'Predicted Price': y_pred[-sample_size:],
            'Error': y_test.tail(sample_size).values - y_pred[-sample_size:],
            'Error %': ((y_test.tail(sample_size).values - y_pred[-sample_size:]) / y_test.tail(sample_size).values) * 100
        })
        
        st.dataframe(
            prediction_df.style.format({
                'Actual Price': '${:,.2f}',
                'Predicted Price': '${:,.2f}',
                'Error': '${:+,.2f}',
                'Error %': '{:+.2f}%'
            })
        )

    elif analysis_type == "📊 TECHNICAL ANALYSIS":
        st.header("📊 COMPREHENSIVE TECHNICAL ANALYSIS")
        
        # Technical Indicators Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_rsi = df_filtered['RSI'].iloc[-1] if not pd.isna(df_filtered['RSI'].iloc[-1]) else 50
            rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            st.metric("📈 RSI (14)", f"{current_rsi:.1f}", rsi_signal)
        
        with col2:
            if len(df_filtered) > 50:
                ma_cross = "Bullish" if df_filtered['Close'].iloc[-1] > df_filtered['MA50'].iloc[-1] else "Bearish"
                st.metric("🎯 MA Cross", ma_cross, "Price vs MA50")
            else:
                st.metric("🎯 MA Cross", "N/A", "Insufficient data")
        
        with col3:
            if 'MACD' in df_filtered.columns:
                current_macd = df_filtered['MACD'].iloc[-1]
                macd_signal = "Bullish" if current_macd > 0 else "Bearish"
                st.metric("⚡ MACD", f"{current_macd:.2f}", macd_signal)
            else:
                st.metric("⚡ MACD", "N/A", "Not calculated")
        
        with col4:
            momentum = ((df_filtered['Close'].iloc[-1] / df_filtered['Close'].iloc[-10]) - 1) * 100 if len(df_filtered) >= 10 else 0
            momentum_signal = "Strong" if abs(momentum) > 5 else "Weak"
            st.metric("💫 Momentum", f"{momentum:+.2f}%", momentum_signal)
        
        # RSI Analysis
        st.subheader("📈 RSI Analysis")
        
        fig_rsi = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Gold Price with Moving Averages', 'RSI (14)')
        )
        
        # Price and MAs
        fig_rsi.add_trace(
            go.Scatter(
                x=df_filtered.index,
                y=df_filtered['Close'],
                name='Close Price',
                line=dict(color='#FFD700', width=3)
            ),
            row=1, col=1
        )
        
        if len(df_filtered) > 20:
            fig_rsi.add_trace(
                go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered['MA20'],
                    name='MA 20',
                    line=dict(color='#FF8C00', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        if len(df_filtered) > 50:
            fig_rsi.add_trace(
                go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered['MA50'],
                    name='MA 50',
                    line=dict(color='#00BFFF', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # RSI
        fig_rsi.add_trace(
            go.Scatter(
                x=df_filtered.index,
                y=df_filtered['RSI'],
                name='RSI',
                line=dict(color='#FF1493', width=2)
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#FF4444", row=2, col=1, annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#44FF44", row=2, col=1, annotation_text="Oversold")
        fig_rsi.add_hline(y=50, line_dash="dash", line_color="#888888", row=2, col=1, annotation_text="Neutral")
        
        fig_rsi.update_layout(
            title="Technical Analysis - RSI",
            height=600,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_rsi.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig_rsi.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Support and Resistance
        st.subheader("🎯 Support & Resistance Analysis")
        
        # Calculate S&R levels
        lookback_period = min(100, len(df_filtered))
        recent_data = df_filtered.tail(lookback_period)
        
        resistance_level = recent_data['High'].quantile(0.95)
        support_level = recent_data['Low'].quantile(0.05)
        current_price = df_filtered['Close'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🔴 Resistance",
                f"${resistance_level:.2f}",
                f"{((resistance_level / current_price) - 1) * 100:+.1f}%"
            )
        
        with col2:
            st.metric(
                "💰 Current Price",
                f"${current_price:.2f}",
                "Market Position"
            )
        
        with col3:
            st.metric(
                "🟢 Support",
                f"${support_level:.2f}",
                f"{((support_level / current_price) - 1) * 100:+.1f}%"
            )
        
        # S&R Chart
        fig_sr = go.Figure()
        
        fig_sr.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered['Close'],
            name='Gold Price',
            line=dict(color='#FFD700', width=3)
        ))
        
        fig_sr.add_hline(
            y=resistance_level,
            line_dash="dash",
            line_color="#FF4444",
            annotation_text="Resistance",
            annotation_position="bottom right"
        )
        
        fig_sr.add_hline(
            y=support_level,
            line_dash="dash",
            line_color="#44FF44",
            annotation_text="Support",
            annotation_position="top right"
        )
        
        fig_sr.update_layout(
            title="Support & Resistance Levels",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_sr, use_container_width=True)

    elif analysis_type == "🎯 BOLLINGER BANDS":
        st.header("🎯 BOLLINGER BANDS ANALYSIS")
        
        # Bollinger Bands parameters
        col1, col2 = st.columns(2)
        with col1:
            bb_period = st.slider("BB Period", 10, 50, 20)
        with col2:
            bb_std = st.slider("Standard Deviation", 1.0, 3.0, 2.0, 0.1)
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df_filtered['Close'], bb_period, bb_std)
        
        # BB Chart
        fig_bb = go.Figure()
        
        # Price line
        fig_bb.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered['Close'],
            name='Close Price',
            line=dict(color='#FFD700', width=3)
        ))
        
        # Upper band
        fig_bb.add_trace(go.Scatter(
            x=df_filtered.index,
            y=bb_upper,
            name='Upper Band',
            line=dict(color='#FF4444', width=2, dash='dash')
        ))
        
        # Middle band (MA)
        fig_bb.add_trace(go.Scatter(
            x=df_filtered.index,
            y=bb_middle,
            name='Middle Band (SMA)',
            line=dict(color='#FFFFFF', width=2)
        ))
        
        # Lower band
        fig_bb.add_trace(go.Scatter(
            x=df_filtered.index,
            y=bb_lower,
            name='Lower Band',
            line=dict(color='#44FF44', width=2, dash='dash')
        ))
        
        # Fill between bands
        fig_bb.add_trace(go.Scatter(
            x=df_filtered.index.tolist() + df_filtered.index.tolist()[::-1],
            y=bb_upper.tolist() + bb_lower.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 215, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='BB Range',
            showlegend=False
        ))
        
        fig_bb.update_layout(
            title=f'Bollinger Bands (Period: {bb_period}, Std: {bb_std})',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=600,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_bb, use_container_width=True)
        
        # BB Analysis
        st.subheader("📊 Bollinger Bands Analysis")
        
        # Calculate BB indicators
        current_price = df_filtered['Close'].iloc[-1]
        current_upper = bb_upper.iloc[-1]
        current_lower = bb_lower.iloc[-1]
        current_middle = bb_middle.iloc[-1]
        
        bb_position = (current_price - current_lower) / (current_upper - current_lower) * 100
        bb_width = ((current_upper - current_lower) / current_middle) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            position_signal = "Overbought" if bb_position > 80 else "Oversold" if bb_position < 20 else "Normal"
            st.metric("📍 BB Position", f"{bb_position:.1f}%", position_signal)
        
        with col2:
            width_signal = "High Volatility" if bb_width > 10 else "Low Volatility" if bb_width < 5 else "Normal"
            st.metric("📏 BB Width", f"{bb_width:.1f}%", width_signal)
        
        with col3:
            distance_to_upper = ((current_upper - current_price) / current_price) * 100
            st.metric("🔴 Distance to Upper", f"{distance_to_upper:.1f}%", "Resistance")
        
        with col4:
            distance_to_lower = ((current_price - current_lower) / current_price) * 100
            st.metric("🟢 Distance to Lower", f"{distance_to_lower:.1f}%", "Support")

    elif analysis_type == "⚡ MACD ANALYSIS":
        st.header("⚡ MACD ANALYSIS")
        
        # MACD Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            fast_period = st.slider("Fast EMA", 5, 20, 12)
        with col2:
            slow_period = st.slider("Slow EMA", 20, 40, 26)
        with col3:
            signal_period = st.slider("Signal EMA", 5, 15, 9)
        
        # Calculate MACD
        macd_line, signal_line, histogram = calculate_macd(df_filtered['Close'], fast_period, slow_period, signal_period)
        
        # MACD Chart
        fig_macd = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Gold Price', 'MACD')
        )
        
        # Price chart
        fig_macd.add_trace(
            go.Scatter(
                x=df_filtered.index,
                y=df_filtered['Close'],
                name='Close Price',
                line=dict(color='#FFD700', width=3)
            ),
            row=1, col=1
        )
        
        # MACD Line
        fig_macd.add_trace(
            go.Scatter(
                x=df_filtered.index,
                y=macd_line,
                name='MACD',
                line=dict(color='#00BFFF', width=2)
            ),
            row=2, col=1
        )
        
        # Signal Line
        fig_macd.add_trace(
            go.Scatter(
                x=df_filtered.index,
                y=signal_line,
                name='Signal',
                line=dict(color='#FF8C00', width=2)
            ),
            row=2, col=1
        )
        
        # Histogram
        colors = ['#00FF41' if h >= 0 else '#FF1493' for h in histogram]
        fig_macd.add_trace(
            go.Bar(
                x=df_filtered.index,
                y=histogram,
                name='Histogram',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig_macd.add_hline(y=0, line_dash="dash", line_color="#888888", row=2, col=1)
        
        fig_macd.update_layout(
            title=f'MACD Analysis ({fast_period}, {slow_period}, {signal_period})',
            height=600,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_macd.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig_macd.update_yaxes(title_text="MACD", row=2, col=1)
        
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # MACD Signals
        st.subheader("📊 MACD Signals")
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
            st.metric("📈 MACD Signal", macd_trend, f"MACD: {current_macd:.3f}")
        
        with col2:
            hist_trend = "Increasing" if current_hist > 0 else "Decreasing"
            st.metric("📊 Histogram", hist_trend, f"{current_hist:.3f}")
        
        with col3:
            zero_cross = "Above Zero" if current_macd > 0 else "Below Zero"
            st.metric("⚡ Zero Line", zero_cross, "MACD Position")
        
        with col4:
            momentum = "Strong" if abs(current_hist) > 0.1 else "Weak"
            st.metric("💫 Momentum", momentum, f"Histogram: {abs(current_hist):.3f}")

else:
    st.error("❌ Unable to load data. Please ensure the data file exists.")
    
    st.markdown("""
    ## 📋 Data Loading Options:
    
    1. **Upload your CSV file** with gold price data
    2. **Ensure proper format**: Date, Open, High, Low, Close, Volume columns
    3. **Check file encoding**: Use UTF-8 encoding
    
    ## 🚀 Demo Mode:
    Sample data has been generated for demonstration purposes.
    """)

# Enhanced Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 3rem; 
                border: 2px solid #FFD700; border-radius: 20px; margin-top: 2rem;
                box-shadow: 0 0 30px rgba(255, 215, 0, 0.3);'>
        <h2 style='color: #FFD700; font-family: Audiowide, cursive; margin-bottom: 1rem;
                   text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);'>
            💫 DASHBOARD ANALISIS EMAS🚀
        </h2>
        
    </div>
    """,
    unsafe_allow_html=True
)