#!/usr/bin/env python3
"""
Momentum Pipeline Module

A rapid prototype for computing momentum features on cryptocurrency price data.
Fetches daily OHLCV data from CoinGecko Pro REST API and computes momentum indicators.

Usage:
    python momentum_pipeline.py --start_date 2021-01-01 --end_date 2025-06-23
"""

import os
import sys
import argparse
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Import technical indicators module
from technical_indicators import compute_all_features, get_feature_summary

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hard-coded list of coins to analyze
COINS = ['bitcoin', 'ethereum', 'ripple']

# CoinGecko API configuration
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_DEMO_API_KEY_PARAM = "x_cg_demo_api_key"
COINGECKO_PRO_BASE_URL = "https://pro-api.coingecko.com/api/v3"


def create_session_with_retries() -> requests.Session:
    """
    Create a requests session with retry strategy for handling rate limits and HTTP errors.
    
    Returns:
        requests.Session: Configured session with exponential backoff retry strategy
    """
    session = requests.Session()
    
    # Configure retry strategy with exponential backoff
    retry_strategy = Retry(
        total=5,  # Maximum number of retries
        backoff_factor=1,  # Base delay between retries (1, 2, 4, 8, 16 seconds)
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        allowed_methods=["GET", "POST"]  # HTTP methods to retry
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def fetch_price_data(coin_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV price data from CoinGecko Pro REST API.
    
    Args:
        coin_id (str): Coin identifier (e.g., 'bitcoin', 'ethereum')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: DataFrame with columns [date, open, high, low, close, volume]
    
    Raises:
        ValueError: If API key is not found or invalid dates provided
        requests.RequestException: If API request fails after retries
    """
    # Validate API key
    api_key = os.getenv('COINGECKO_API_KEY')
    if not api_key:
        raise ValueError("COINGECKO_API_KEY environment variable not set")
    
    # Validate date format
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format")
    
    # Convert dates to Unix timestamps (milliseconds)
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    # Prepare API request
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/ohlc"
    params = {
        'vs_currency': 'usd',
        'days': 'max',  # Get maximum available data
        COINGECKO_DEMO_API_KEY_PARAM: api_key
    }
    
    logger.info(f"Fetching price data for {coin_id} from {start_date} to {end_date}")
    
    # Make API request with retry logic
    session = create_session_with_retries()
    
    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            logger.warning(f"No data returned for {coin_id}")
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Parse OHLCV data
        # CoinGecko returns: [timestamp, open, high, low, close]
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Filter by date range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Add volume column (CoinGecko OHLC endpoint doesn't include volume)
        # We'll set it to NaN for now - in a real implementation, you'd fetch this separately
        df['volume'] = float('nan')
        
        # Select and reorder columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Successfully fetched {len(df)} records for {coin_id}")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {coin_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching data for {coin_id}: {e}")
        raise


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum features from price data.
    
    Args:
        df (pd.DataFrame): DataFrame with columns [date, open, high, low, close, volume]
    
    Returns:
        pd.DataFrame: Original DataFrame with added momentum columns
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to compute_momentum")
        return df
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure close prices are numeric
    result_df['close'] = pd.to_numeric(result_df['close'], errors='coerce')
    
    # Sort by date to ensure proper calculation
    result_df = result_df.sort_values('date').reset_index(drop=True)
    
    # Compute momentum features (percentage change over different periods)
    logger.info("Computing momentum features...")
    
    # 7-day momentum
    result_df['mom_7d'] = result_df['close'].pct_change(periods=7) * 100
    
    # 21-day momentum (approximately 1 month)
    result_df['mom_21d'] = result_df['close'].pct_change(periods=21) * 100
    
    # 60-day momentum (approximately 3 months)
    result_df['mom_60d'] = result_df['close'].pct_change(periods=60) * 100
    
    # Compute momentum score as row-wise average of the three momentum features
    momentum_columns = ['mom_7d', 'mom_21d', 'mom_60d']
    result_df['momentum_score'] = result_df[momentum_columns].mean(axis=1)
    
    logger.info(f"Computed momentum features for {len(result_df)} records")
    
    return result_df


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to CSV file, creating directories as needed.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path where to save the CSV file
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved momentum features to {output_path}")


def main():
    """
    Main entry point for the momentum pipeline CLI.
    """
    parser = argparse.ArgumentParser(
        description="Momentum Pipeline - Compute momentum features for cryptocurrency data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python momentum_pipeline.py --start_date 2021-01-01 --end_date 2025-06-23
  python momentum_pipeline.py --start_date 2023-01-01 --end_date 2023-12-31
        """
    )
    
    parser.add_argument(
        '--start_date',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        '--end_date',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/momentum_features.csv',
        help='Output file path (default: data/processed/momentum_features.csv)'
    )
    
    parser.add_argument(
        '--include_technical',
        action='store_true',
        help='Include technical indicators (RSI, MACD, Bollinger Bands, etc.)'
    )
    
    parser.add_argument(
        '--include_risk',
        action='store_true',
        help='Include risk metrics (volatility, Sharpe ratio, VaR, etc.)'
    )
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        if start_dt >= end_dt:
            logger.error("Start date must be before end date")
            sys.exit(1)
            
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD")
        sys.exit(1)
    
    logger.info(f"Starting momentum pipeline for {len(COINS)} coins")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Include technical indicators: {args.include_technical}")
    logger.info(f"Include risk metrics: {args.include_risk}")
    
    # List to store all coin data
    all_data = []
    
    # Process each coin
    for coin_id in COINS:
        try:
            logger.info(f"Processing {coin_id}...")
            
            # Fetch price data
            price_df = fetch_price_data(coin_id, args.start_date, args.end_date)
            
            if price_df.empty:
                logger.warning(f"No data available for {coin_id}")
                continue
            
            # Compute momentum features
            momentum_df = compute_momentum(price_df)
            
            # Add technical indicators and risk metrics if requested
            if args.include_technical or args.include_risk:
                logger.info(f"Computing technical indicators and risk metrics for {coin_id}...")
                momentum_df = compute_all_features(momentum_df)
            
            # Add coin identifier
            momentum_df['coin_id'] = coin_id
            
            # Append to results
            all_data.append(momentum_df)
            
            # Add delay to respect rate limits
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing {coin_id}: {e}")
            continue
    
    if not all_data:
        logger.error("No data was successfully processed")
        sys.exit(1)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date and coin_id
    combined_df = combined_df.sort_values(['date', 'coin_id']).reset_index(drop=True)
    
    logger.info(f"Combined data shape: {combined_df.shape}")
    logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    logger.info(f"Coins processed: {combined_df['coin_id'].unique()}")
    
    # Generate feature summary if technical indicators or risk metrics are included
    if args.include_technical or args.include_risk:
        feature_summary = get_feature_summary(combined_df)
        logger.info("Feature summary generated")
    
    # Save results
    save_dataframe(combined_df, args.output)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ENHANCED MOMENTUM PIPELINE SUMMARY")
    print("="*60)
    print(f"Total records: {len(combined_df)}")
    print(f"Date range: {combined_df['date'].min().strftime('%Y-%m-%d')} to {combined_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Coins: {', '.join(combined_df['coin_id'].unique())}")
    print(f"Output file: {args.output}")
    print(f"Features included:")
    print(f"  - Momentum features: ✓")
    print(f"  - Technical indicators: {'✓' if args.include_technical else '✗'}")
    print(f"  - Risk metrics: {'✓' if args.include_risk else '✗'}")
    
    # Show momentum score statistics
    if 'momentum_score' in combined_df.columns:
        print(f"\nMomentum Score Statistics:")
        print(f"  Mean: {combined_df['momentum_score'].mean():.2f}%")
        print(f"  Std:  {combined_df['momentum_score'].std():.2f}%")
        print(f"  Min:  {combined_df['momentum_score'].min():.2f}%")
        print(f"  Max:  {combined_df['momentum_score'].max():.2f}%")
    
    # Show technical indicator statistics if included
    if args.include_technical and 'rsi_14' in combined_df.columns:
        print(f"\nTechnical Indicators Summary:")
        print(f"  RSI (14): Mean={combined_df['rsi_14'].mean():.1f}, Std={combined_df['rsi_14'].std():.1f}")
        print(f"  MACD: Mean={combined_df['macd_line'].mean():.2f}, Std={combined_df['macd_line'].std():.2f}")
        print(f"  BB Width: Mean={combined_df['bb_width'].mean():.1f}%, Std={combined_df['bb_width'].std():.1f}%")
    
    # Show risk metrics statistics if included
    if args.include_risk and 'sharpe_ratio' in combined_df.columns:
        print(f"\nRisk Metrics Summary:")
        print(f"  Sharpe Ratio: Mean={combined_df['sharpe_ratio'].mean():.2f}, Std={combined_df['sharpe_ratio'].std():.2f}")
        print(f"  Volatility (20d): Mean={combined_df['volatility_20d'].mean():.1f}%, Std={combined_df['volatility_20d'].std():.1f}%")
        print(f"  Max Drawdown: Mean={combined_df['max_drawdown'].mean():.1f}%, Min={combined_df['max_drawdown'].min():.1f}%")
        print(f"  VaR (95%): Mean={combined_df['var_95'].mean():.1f}%, Std={combined_df['var_95'].std():.1f}%")
    
    print("="*60)


if __name__ == "__main__":
    main() 