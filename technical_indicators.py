#!/usr/bin/env python3
"""
Technical Indicators and Risk Metrics Module

Provides comprehensive technical analysis functions for cryptocurrency data.
Includes RSI, MACD, Bollinger Bands, moving averages, and risk metrics.

Usage:
    from technical_indicators import compute_technical_indicators, compute_risk_metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute Simple Moving Average (SMA).
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' column
        period (int): Period for SMA calculation (default: 20)
    
    Returns:
        pd.Series: Simple Moving Average values
    """
    return df['close'].rolling(window=period).mean()


def compute_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute Exponential Moving Average (EMA).
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' column
        period (int): Period for EMA calculation (default: 20)
    
    Returns:
        pd.Series: Exponential Moving Average values
    """
    return df['close'].ewm(span=period).mean()


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' column
        period (int): Period for RSI calculation (default: 14)
    
    Returns:
        pd.Series: RSI values (0-100)
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence).
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' column
        fast_period (int): Fast EMA period (default: 12)
        slow_period (int): Slow EMA period (default: 26)
        signal_period (int): Signal line period (default: 9)
    
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (MACD line, Signal line, Histogram)
    """
    ema_fast = df['close'].ewm(span=fast_period).mean()
    ema_slow = df['close'].ewm(span=slow_period).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' column
        period (int): Period for SMA calculation (default: 20)
        std_dev (float): Number of standard deviations (default: 2.0)
    
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (Upper band, Middle band, Lower band)
    """
    middle_band = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def compute_volatility(df: pd.DataFrame, period: int = 20, annualized: bool = True) -> pd.Series:
    """
    Compute price volatility (standard deviation of returns).
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' column
        period (int): Rolling window period (default: 20)
        annualized (bool): Whether to annualize volatility (default: True)
    
    Returns:
        pd.Series: Volatility values
    """
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=period).std()
    
    if annualized:
        # Annualize by multiplying by sqrt(252) for daily data
        volatility = volatility * np.sqrt(252)
    
    return volatility


def compute_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.02, period: int = 252) -> pd.Series:
    """
    Compute Sharpe ratio (risk-adjusted return).
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' column
        risk_free_rate (float): Annual risk-free rate (default: 2%)
        period (int): Rolling window period (default: 252 days)
    
    Returns:
        pd.Series: Sharpe ratio values
    """
    returns = df['close'].pct_change()
    
    # Rolling mean return
    rolling_return = returns.rolling(window=period).mean() * 252  # Annualize
    
    # Rolling volatility
    rolling_vol = returns.rolling(window=period).std() * np.sqrt(252)  # Annualize
    
    # Sharpe ratio
    sharpe = (rolling_return - risk_free_rate) / rolling_vol
    
    return sharpe


def compute_max_drawdown(df: pd.DataFrame, period: int = 252) -> pd.Series:
    """
    Compute maximum drawdown over a rolling window.
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' column
        period (int): Rolling window period (default: 252 days)
    
    Returns:
        pd.Series: Maximum drawdown values (negative percentages)
    """
    rolling_max = df['close'].rolling(window=period).max()
    drawdown = (df['close'] - rolling_max) / rolling_max * 100
    return drawdown


def compute_var(df: pd.DataFrame, confidence_level: float = 0.05, period: int = 252) -> pd.Series:
    """
    Compute Value at Risk (VaR) using historical simulation.
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' column
        confidence_level (float): VaR confidence level (default: 5%)
        period (int): Rolling window period (default: 252 days)
    
    Returns:
        pd.Series: VaR values (negative percentages)
    """
    returns = df['close'].pct_change()
    
    def rolling_var(returns_series):
        if len(returns_series.dropna()) < 30:  # Need minimum observations
            return np.nan
        return np.percentile(returns_series.dropna(), confidence_level * 100) * 100
    
    var = returns.rolling(window=period).apply(rolling_var)
    return var


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with columns [date, open, high, low, close, volume]
    
    Returns:
        pd.DataFrame: Original DataFrame with added technical indicator columns
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to compute_technical_indicators")
        return df
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure close prices are numeric
    result_df['close'] = pd.to_numeric(result_df['close'], errors='coerce')
    
    # Sort by date to ensure proper calculation
    result_df = result_df.sort_values('date').reset_index(drop=True)
    
    logger.info("Computing technical indicators...")
    
    # Moving Averages
    result_df['sma_20'] = compute_sma(result_df, period=20)
    result_df['sma_50'] = compute_sma(result_df, period=50)
    result_df['ema_20'] = compute_ema(result_df, period=20)
    result_df['ema_50'] = compute_ema(result_df, period=50)
    
    # RSI
    result_df['rsi_14'] = compute_rsi(result_df, period=14)
    
    # MACD
    macd_line, signal_line, histogram = compute_macd(result_df)
    result_df['macd_line'] = macd_line
    result_df['macd_signal'] = signal_line
    result_df['macd_histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(result_df)
    result_df['bb_upper'] = bb_upper
    result_df['bb_middle'] = bb_middle
    result_df['bb_lower'] = bb_lower
    result_df['bb_width'] = (bb_upper - bb_lower) / bb_middle * 100  # Percentage width
    result_df['bb_position'] = (result_df['close'] - bb_lower) / (bb_upper - bb_lower) * 100  # Position within bands
    
    logger.info(f"Computed technical indicators for {len(result_df)} records")
    
    return result_df


def compute_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all risk metrics for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with columns [date, open, high, low, close, volume]
    
    Returns:
        pd.DataFrame: Original DataFrame with added risk metric columns
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to compute_risk_metrics")
        return df
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure close prices are numeric
    result_df['close'] = pd.to_numeric(result_df['close'], errors='coerce')
    
    # Sort by date to ensure proper calculation
    result_df = result_df.sort_values('date').reset_index(drop=True)
    
    logger.info("Computing risk metrics...")
    
    # Volatility
    result_df['volatility_20d'] = compute_volatility(result_df, period=20)
    result_df['volatility_60d'] = compute_volatility(result_df, period=60)
    
    # Sharpe Ratio
    result_df['sharpe_ratio'] = compute_sharpe_ratio(result_df)
    
    # Maximum Drawdown
    result_df['max_drawdown'] = compute_max_drawdown(result_df)
    
    # Value at Risk
    result_df['var_95'] = compute_var(result_df, confidence_level=0.05)  # 95% VaR
    result_df['var_99'] = compute_var(result_df, confidence_level=0.01)  # 99% VaR
    
    # Additional risk metrics
    returns = result_df['close'].pct_change()
    result_df['daily_return'] = returns * 100  # Convert to percentage
    
    # Rolling return metrics
    result_df['return_7d'] = result_df['close'].pct_change(periods=7) * 100
    result_df['return_30d'] = result_df['close'].pct_change(periods=30) * 100
    
    # Rolling volatility of returns
    result_df['return_volatility_20d'] = returns.rolling(window=20).std() * np.sqrt(252) * 100
    
    logger.info(f"Computed risk metrics for {len(result_df)} records")
    
    return result_df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and risk metrics in one function.
    
    Args:
        df (pd.DataFrame): DataFrame with columns [date, open, high, low, close, volume]
    
    Returns:
        pd.DataFrame: Original DataFrame with all technical and risk features
    """
    logger.info("Computing all technical indicators and risk metrics...")
    
    # Compute technical indicators first
    df_with_technical = compute_technical_indicators(df)
    
    # Then compute risk metrics
    df_with_all_features = compute_risk_metrics(df_with_technical)
    
    logger.info(f"Completed computation of all features for {len(df_with_all_features)} records")
    
    return df_with_all_features


def get_feature_summary(df: pd.DataFrame) -> Dict:
    """
    Generate a summary of all computed features.
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators and risk metrics
    
    Returns:
        Dict: Summary statistics for all features
    """
    # Define feature categories
    technical_features = [
        'sma_20', 'sma_50', 'ema_20', 'ema_50', 'rsi_14',
        'macd_line', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position'
    ]
    
    risk_features = [
        'volatility_20d', 'volatility_60d', 'sharpe_ratio',
        'max_drawdown', 'var_95', 'var_99',
        'daily_return', 'return_7d', 'return_30d', 'return_volatility_20d'
    ]
    
    momentum_features = ['mom_7d', 'mom_21d', 'mom_60d', 'momentum_score']
    
    summary = {}
    
    # Technical indicators summary
    for feature in technical_features:
        if feature in df.columns:
            summary[feature] = {
                'mean': df[feature].mean(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max(),
                'null_count': df[feature].isnull().sum()
            }
    
    # Risk metrics summary
    for feature in risk_features:
        if feature in df.columns:
            summary[feature] = {
                'mean': df[feature].mean(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max(),
                'null_count': df[feature].isnull().sum()
            }
    
    # Momentum features summary
    for feature in momentum_features:
        if feature in df.columns:
            summary[feature] = {
                'mean': df[feature].mean(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max(),
                'null_count': df[feature].isnull().sum()
            }
    
    return summary 