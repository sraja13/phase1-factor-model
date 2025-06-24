#!/usr/bin/env python3
"""
Test script to verify CoinGecko Demo API key functionality.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key():
    """Test the CoinGecko demo API key with a simple price request."""
    
    api_key = os.getenv('COINGECKO_API_KEY')
    if not api_key:
        print("❌ COINGECKO_API_KEY not found in environment variables")
        return False
    
    print(f"🔑 API Key found: {api_key[:10]}...")
    
    # Test with simple price endpoint
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        'ids': 'bitcoin',
        'vs_currencies': 'usd',
        'x_cg_demo_api_key': api_key
    }
    
    try:
        print("🌐 Testing API connection...")
        response = requests.get(url, params=params, timeout=10)
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API key is working!")
            print(f"📈 Bitcoin price: ${data['bitcoin']['usd']:,.2f}")
            return True
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def test_ohlc_endpoint():
    """Test the OHLC endpoint that we'll use in the momentum pipeline."""
    
    api_key = os.getenv('COINGECKO_API_KEY')
    if not api_key:
        print("❌ COINGECKO_API_KEY not found")
        return False
    
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    params = {
        'vs_currency': 'usd',
        'days': '7',  # Just get 7 days for testing
        'x_cg_demo_api_key': api_key
    }
    
    try:
        print("📊 Testing OHLC endpoint...")
        response = requests.get(url, params=params, timeout=10)
        
        print(f"📊 OHLC Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ OHLC endpoint working! Got {len(data)} data points")
            if data:
                print(f"📅 Latest data point: {data[-1]}")
            return True
        else:
            print(f"❌ OHLC request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OHLC endpoint: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing CoinGecko Demo API Key")
    print("=" * 40)
    
    # Test simple price endpoint
    price_test = test_api_key()
    print()
    
    # Test OHLC endpoint
    ohlc_test = test_ohlc_endpoint()
    print()
    
    if price_test and ohlc_test:
        print("🎉 All tests passed! API key is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check your API key and limits.") 