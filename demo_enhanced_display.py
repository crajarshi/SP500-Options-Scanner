#!/usr/bin/env python3
"""
Demo script to showcase the enhanced dashboard with all indicators
"""
from dashboard import OptionsScannnerDashboard
from datetime import datetime
import config

def create_demo_analyses():
    """Create demo analysis data with all scores"""
    return [
        {
            'ticker': 'AAPL',
            'current_price': 175.32,
            'price_change_pct': -2.1,
            'scores': {
                'composite_score': 92.5,
                'rsi_score': 100.0,  # Oversold
                'macd_score': 100.0,  # Bullish
                'bollinger_score': 85.0,  # Near lower band
                'obv_score': 100.0,  # Above SMA
                'atr_score': 100.0  # Expanding
            },
            'signal': {
                'type': 'STRONG_BUY',
                'text': 'STRONG BUY',
                'emoji': '🟢'
            },
            'indicators': {
                'rsi': 28,
                'macd_bullish': True,
                'bb_position': 15,
                'obv_above_sma': True,
                'atr_value': 3.45,
                'atr_trend': 'Rising'
            }
        },
        {
            'ticker': 'MSFT',
            'current_price': 382.15,
            'price_change_pct': -1.8,
            'scores': {
                'composite_score': 78.5,
                'rsi_score': 75.0,
                'macd_score': 100.0,
                'bollinger_score': 70.0,
                'obv_score': 100.0,
                'atr_score': 0.0  # Contracting
            },
            'signal': {
                'type': 'BUY',
                'text': 'BUY',
                'emoji': '🟢'
            },
            'indicators': {
                'rsi': 38,
                'macd_bullish': True,
                'bb_position': 30,
                'obv_above_sma': True,
                'atr_value': 5.20,
                'atr_trend': 'Falling'
            }
        },
        {
            'ticker': 'NVDA',
            'current_price': 695.20,
            'price_change_pct': -3.2,
            'scores': {
                'composite_score': 55.0,
                'rsi_score': 50.0,
                'macd_score': 0.0,
                'bollinger_score': 50.0,
                'obv_score': 100.0,
                'atr_score': 100.0
            },
            'signal': {
                'type': 'HOLD',
                'text': 'HOLD',
                'emoji': '⚪'
            },
            'indicators': {
                'rsi': 50,
                'macd_bullish': False,
                'bb_position': 50,
                'obv_above_sma': True,
                'atr_value': 12.30,
                'atr_trend': 'Rising'
            }
        },
        {
            'ticker': 'TSLA',
            'current_price': 242.18,
            'price_change_pct': 1.5,
            'scores': {
                'composite_score': 25.0,
                'rsi_score': 0.0,  # Overbought
                'macd_score': 0.0,
                'bollinger_score': 10.0,  # Near upper band
                'obv_score': 0.0,
                'atr_score': 0.0
            },
            'signal': {
                'type': 'AVOID',
                'text': 'AVOID',
                'emoji': '🔴'
            },
            'indicators': {
                'rsi': 75,
                'macd_bullish': False,
                'bb_position': 90,
                'obv_above_sma': False,
                'atr_value': 8.75,
                'atr_trend': 'Falling'
            }
        },
        {
            'ticker': 'META',
            'current_price': 485.62,
            'price_change_pct': -0.8,
            'scores': {
                'composite_score': 72.0,
                'rsi_score': 60.0,
                'macd_score': 100.0,
                'bollinger_score': 60.0,
                'obv_score': 100.0,
                'atr_score': 0.0
            },
            'signal': {
                'type': 'BUY',
                'text': 'BUY',
                'emoji': '🟢'
            },
            'indicators': {
                'rsi': 42,
                'macd_bullish': True,
                'bb_position': 40,
                'obv_above_sma': True,
                'atr_value': 7.15,
                'atr_trend': 'Falling'
            }
        }
    ]

def main():
    """Run demo display"""
    dashboard = OptionsScannnerDashboard()
    
    # Create demo data
    analyses = create_demo_analyses()
    scan_time = datetime.now()
    errors = []
    
    # Display full dashboard
    dashboard.display_results(analyses, scan_time, errors)
    
    print("\n" + "="*80)
    print("ENHANCED DISPLAY FEATURES:")
    print("="*80)
    print("1. Scoring Explanation Panel - Shows how composite score is calculated")
    print("2. Individual Indicator Scores - Each indicator shows 0-100 score")
    print("3. Color Coding:")
    print("   - Green (80-100): Strong signal")
    print("   - Yellow (1-79): Moderate signal")
    print("   - Red (0): No signal")
    print("4. ATR Trend - Shows ↑ for expanding, ↓ for contracting volatility")
    print("5. Complete Transparency - All components of the score visible")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()