# Enhanced Momentum Pipeline

A comprehensive cryptocurrency analysis pipeline that computes momentum features, technical indicators, and risk metrics using data from CoinGecko Pro API.

## 🚀 Features

### Core Momentum Features
- **7-day momentum**: Short-term price momentum
- **21-day momentum**: Medium-term price momentum  
- **60-day momentum**: Long-term price momentum
- **Momentum score**: Composite momentum indicator

### Technical Indicators
- **Moving Averages**: SMA (20, 50) and EMA (20, 50)
- **RSI (14)**: Relative Strength Index for overbought/oversold conditions
- **MACD**: Moving Average Convergence Divergence with signal line and histogram
- **Bollinger Bands**: Upper, middle, lower bands with width and position metrics

### Risk Metrics
- **Volatility**: 20-day and 60-day rolling volatility (annualized)
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Rolling maximum drawdown calculation
- **Value at Risk (VaR)**: 95% and 99% confidence levels
- **Return Metrics**: Daily, 7-day, and 30-day returns

## 📦 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd phase1-factor-model
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
# Create .env file
echo "COINGECKO_API_KEY=your_api_key_here" > .env
```

## 🔧 Usage

### Basic Momentum Pipeline
```bash
python momentum_pipeline.py --start_date 2021-01-01 --end_date 2025-06-23
```

### With Technical Indicators
```bash
python momentum_pipeline.py --start_date 2021-01-01 --end_date 2025-06-23 --include_technical
```

### With Risk Metrics
```bash
python momentum_pipeline.py --start_date 2021-01-01 --end_date 2025-06-23 --include_risk
```

### Complete Analysis (All Features)
```bash
python momentum_pipeline.py --start_date 2021-01-01 --end_date 2025-06-23 --include_technical --include_risk
```

### Custom Output Path
```bash
python momentum_pipeline.py --start_date 2021-01-01 --end_date 2025-06-23 --output data/my_analysis.csv
```

## 📊 Output Data Structure

The pipeline generates a CSV file with the following columns:

### Base Data
- `date`: Date of the observation
- `coin_id`: Cryptocurrency identifier
- `open`, `high`, `low`, `close`: OHLC price data
- `volume`: Trading volume (currently NaN - CoinGecko OHLC doesn't include volume)

### Momentum Features
- `mom_7d`: 7-day percentage change
- `mom_21d`: 21-day percentage change
- `mom_60d`: 60-day percentage change
- `momentum_score`: Average of momentum features

### Technical Indicators (with `--include_technical`)
- `sma_20`, `sma_50`: Simple Moving Averages
- `ema_20`, `ema_50`: Exponential Moving Averages
- `rsi_14`: Relative Strength Index
- `macd_line`, `macd_signal`, `macd_histogram`: MACD components
- `bb_upper`, `bb_middle`, `bb_lower`: Bollinger Bands
- `bb_width`: Bollinger Band width percentage
- `bb_position`: Price position within Bollinger Bands

### Risk Metrics (with `--include_risk`)
- `volatility_20d`, `volatility_60d`: Rolling volatility
- `sharpe_ratio`: Risk-adjusted return
- `max_drawdown`: Maximum drawdown percentage
- `var_95`, `var_99`: Value at Risk at different confidence levels
- `daily_return`: Daily percentage return
- `return_7d`, `return_30d`: Rolling returns
- `return_volatility_20d`: Volatility of returns

## 🔍 API Configuration

### CoinGecko Pro API
- **Rate Limit**: 30 requests per minute
- **Authentication**: API key required via `COINGECKO_API_KEY` environment variable
- **Data**: Daily OHLCV data for cryptocurrencies

### Supported Coins
Currently hardcoded to analyze:
- Bitcoin (bitcoin)
- Ethereum (ethereum)
- Ripple (ripple)

## 📈 Technical Analysis Details

### RSI (Relative Strength Index)
- **Range**: 0-100
- **Overbought**: >70
- **Oversold**: <30
- **Period**: 14 days (standard)

### MACD (Moving Average Convergence Divergence)
- **Fast EMA**: 12 periods
- **Slow EMA**: 26 periods
- **Signal Line**: 9-period EMA of MACD line
- **Histogram**: MACD line - Signal line

### Bollinger Bands
- **Period**: 20 days
- **Standard Deviation**: 2.0
- **Width**: Percentage distance between upper and lower bands
- **Position**: Price position within bands (0-100%)

### Risk Metrics
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: (Return - Risk-free rate) / Volatility
- **Maximum Drawdown**: Largest peak-to-trough decline
- **VaR**: Historical simulation method

## 🛠️ Development

### Project Structure
```
phase1-factor-model/
├── momentum_pipeline.py      # Main pipeline with CLI
├── technical_indicators.py   # Technical analysis functions
├── test_api.py              # API testing script
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (gitignored)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

### Adding New Features
1. **Technical Indicators**: Add functions to `technical_indicators.py`
2. **Risk Metrics**: Extend `compute_risk_metrics()` function
3. **CLI Options**: Add arguments to `main()` function
4. **Data Processing**: Update pipeline workflow

### Testing
```bash
# Test API connection
python test_api.py

# Test with small date range
python momentum_pipeline.py --start_date 2024-01-01 --end_date 2024-01-31 --include_technical --include_risk
```

## 📋 Requirements

- Python 3.8+
- pandas >= 1.5.0
- requests >= 2.28.0
- numpy >= 1.21.0
- python-dotenv >= 0.19.0
- urllib3 >= 1.26.0

## 🔒 Security

- API keys stored in `.env` file (gitignored)
- No hardcoded credentials in source code
- Rate limiting and exponential backoff for API calls
- Comprehensive error handling

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Support

For issues or questions, please check the documentation or create an issue in the repository.