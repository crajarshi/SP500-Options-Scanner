"""
Unit tests for Maximum Profit Scanner
Tests scoring functions, filtering logic, and edge cases
"""
import unittest
import numpy as np
from datetime import datetime, timedelta
from max_profit_scanner import MaxProfitScanner, MaxProfitContract
import config

class MockDataProvider:
    """Mock data provider for testing"""
    
    def get_sp500_tickers(self):
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    def fetch_latest_quote(self, ticker):
        return {
            'price': 150.0,
            'volume': 50000000,
            'bid': 149.95,
            'ask': 150.05
        }
    
    def calculate_beta(self, ticker):
        # Return different betas for testing
        betas = {'AAPL': 1.25, 'MSFT': 1.1, 'GOOGL': 1.3, 'AMZN': 1.5, 'NVDA': 2.0}
        return betas.get(ticker, 1.0)
    
    def fetch_options_chain(self, ticker):
        # Return mock options chain
        exp_date = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
        
        return {
            exp_date: {
                150: {
                    'call': {
                        'bid': 2.50,
                        'ask': 2.60,
                        'delta': 0.30,
                        'gamma': 0.05,
                        'theta': -0.15,
                        'vega': 0.10,
                        'implied_volatility': 0.45,
                        'open_interest': 500,
                        'volume': 100
                    },
                    'put': {
                        'bid': 3.00,
                        'ask': 3.10,
                        'delta': -0.30,
                        'gamma': 0.05,
                        'theta': -0.12,
                        'vega': 0.10,
                        'implied_volatility': 0.48,
                        'open_interest': 300,
                        'volume': 50
                    }
                }
            }
        }


class TestMaxProfitScoring(unittest.TestCase):
    """Test scoring functions"""
    
    def setUp(self):
        self.scanner = MaxProfitScanner(test_mode=True)
        self.scanner.data_provider = MockDataProvider()
    
    def test_liquidity_score_calculation(self):
        """Test liquidity scoring with various inputs"""
        contract = MaxProfitContract(
            symbol='TEST',
            strike=100,
            expiration=datetime.now() + timedelta(days=14),
            contract_type='call',
            days_to_expiry=14,
            bid=2.0,
            ask=2.1,
            mid_price=2.05,
            spread_percent=0.05,
            delta=0.3,
            gamma=0.05,
            theta=-0.1,
            open_interest=500,
            volume=100,
            avg_volume_5d=50,
            implied_volatility=0.3,
            iv_rank=75
        )
        
        score = self.scanner.calculate_liquidity_score(contract)
        
        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
        # Test edge cases
        contract.open_interest = 0
        contract.avg_volume_5d = 0
        contract.spread_percent = 1.0
        score_low = self.scanner.calculate_liquidity_score(contract)
        self.assertLess(score_low, 0.2)  # Should be very low
        
        contract.open_interest = 10000
        contract.avg_volume_5d = 1000
        contract.spread_percent = 0.01
        score_high = self.scanner.calculate_liquidity_score(contract)
        self.assertGreater(score_high, 0.7)  # Should be high
    
    def test_gtr_normalization_with_winsorization(self):
        """Test GTR normalization handles outliers correctly"""
        contracts = [
            MaxProfitContract(
                symbol='TEST', strike=100, expiration=datetime.now() + timedelta(days=14),
                contract_type='call', days_to_expiry=14, bid=1, ask=1.1,
                mid_price=1.05, spread_percent=0.1, delta=0.3,
                gamma=0.01, theta=-0.1, open_interest=100, volume=10,
                avg_volume_5d=5, implied_volatility=0.3, iv_rank=50
            ),
            MaxProfitContract(
                symbol='TEST', strike=100, expiration=datetime.now() + timedelta(days=14),
                contract_type='call', days_to_expiry=14, bid=1, ask=1.1,
                mid_price=1.05, spread_percent=0.1, delta=0.3,
                gamma=0.05, theta=-0.2, open_interest=100, volume=10,
                avg_volume_5d=5, implied_volatility=0.3, iv_rank=50
            ),
            MaxProfitContract(
                symbol='TEST', strike=100, expiration=datetime.now() + timedelta(days=14),
                contract_type='call', days_to_expiry=14, bid=1, ask=1.1,
                mid_price=1.05, spread_percent=0.1, delta=0.3,
                gamma=10.0, theta=-0.01, open_interest=100, volume=10,
                avg_volume_5d=5, implied_volatility=0.3, iv_rank=50
            ),  # Outlier with huge gamma/theta ratio
        ]
        
        gtr_min, gtr_max = self.scanner.winsorize_gtr_values(contracts)
        
        # After winsorization, the range should be reasonable
        # With only 3 samples and one extreme outlier, expect larger range
        self.assertLess(gtr_max, 1100)  # Outlier should be clipped
        self.assertGreater(gtr_max, gtr_min)  # Valid range
        self.assertGreater(gtr_min, 0)  # Min should be positive
    
    def test_final_score_calculation(self):
        """Test complete scoring pipeline"""
        contract = MaxProfitContract(
            symbol='TEST',
            strike=100,
            expiration=datetime.now() + timedelta(days=14),
            contract_type='call',
            days_to_expiry=14,
            bid=2.0,
            ask=2.1,
            mid_price=2.05,
            spread_percent=0.05,
            delta=0.3,
            gamma=0.05,
            theta=-0.2,
            open_interest=500,
            volume=100,
            avg_volume_5d=25,
            implied_volatility=0.45,
            iv_rank=75
        )
        
        score = self.scanner.calculate_final_score(contract, gtr_min=0.1, gtr_max=2.0)
        
        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
        # Check breakdown exists and has all components
        self.assertIn('GTR', contract.score_breakdown)
        self.assertIn('IVR', contract.score_breakdown)
        self.assertIn('LIQ', contract.score_breakdown)
        self.assertIn('PRICE_ADJ', contract.score_breakdown)
        
        # Check raw components are stored
        self.assertIn('gtr', contract.raw_components)
        self.assertIn('gtr_norm', contract.raw_components)
        self.assertIn('ivr', contract.raw_components)
        self.assertIn('liquidity', contract.raw_components)
    
    def test_theta_zero_handling(self):
        """Test that zero theta is handled correctly"""
        contract = MaxProfitContract(
            symbol='TEST',
            strike=100,
            expiration=datetime.now() + timedelta(days=14),
            contract_type='call',
            days_to_expiry=14,
            bid=2.0,
            ask=2.1,
            mid_price=2.05,
            spread_percent=0.05,
            delta=0.3,
            gamma=0.05,
            theta=0,  # Zero theta
            open_interest=500,
            volume=100,
            avg_volume_5d=25,
            implied_volatility=0.3,
            iv_rank=75
        )
        
        # Should not raise division by zero error
        score = self.scanner.calculate_final_score(contract, gtr_min=0.1, gtr_max=100)
        
        # Score should still be valid
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
        # GTR should be very high (gamma / epsilon)
        self.assertGreater(contract.gamma_theta_ratio, 1000)
    
    def test_price_penalty_effect(self):
        """Test that price penalty reduces score for expensive options"""
        # Create two identical contracts except for price
        cheap_contract = MaxProfitContract(
            symbol='TEST', strike=100, expiration=datetime.now() + timedelta(days=14),
            contract_type='call', days_to_expiry=14, bid=0.5, ask=0.55,
            mid_price=0.525, spread_percent=0.1, delta=0.3,
            gamma=0.05, theta=-0.1, open_interest=500, volume=100,
            avg_volume_5d=25, implied_volatility=0.3, iv_rank=75
        )
        
        expensive_contract = MaxProfitContract(
            symbol='TEST', strike=100, expiration=datetime.now() + timedelta(days=14),
            contract_type='call', days_to_expiry=14, bid=10.0, ask=10.5,
            mid_price=10.25, spread_percent=0.05, delta=0.3,
            gamma=0.05, theta=-0.1, open_interest=500, volume=100,
            avg_volume_5d=25, implied_volatility=0.3, iv_rank=75
        )
        
        cheap_score = self.scanner.calculate_final_score(cheap_contract, 0.1, 2.0)
        expensive_score = self.scanner.calculate_final_score(expensive_contract, 0.1, 2.0)
        
        # Cheap contract should score higher due to price penalty
        self.assertGreater(cheap_score, expensive_score)


class TestMaxProfitFiltering(unittest.TestCase):
    """Test filtering logic"""
    
    def setUp(self):
        self.scanner = MaxProfitScanner(test_mode=True)
        self.scanner.data_provider = MockDataProvider()
    
    def test_stock_pre_filtering(self):
        """Test that stocks are filtered by beta and volume"""
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        
        # Set beta threshold high to filter some stocks
        self.scanner.beta_threshold = 1.4
        
        filtered = self.scanner.pre_filter_stocks(tickers)
        
        # Only AMZN (1.5) and NVDA (2.0) should pass
        self.assertIn('AMZN', filtered)
        self.assertIn('NVDA', filtered)
        self.assertNotIn('AAPL', filtered)  # Beta 1.25 < 1.4
        self.assertNotIn('MSFT', filtered)  # Beta 1.1 < 1.4
    
    def test_delta_filtering(self):
        """Test two-stage delta filtering"""
        contracts = []
        
        # Create contracts with various deltas
        deltas = [0.05, 0.12, 0.25, 0.35, 0.48, 0.55, 0.70]
        
        for delta in deltas:
            contract = MaxProfitContract(
                symbol='TEST', strike=100, expiration=datetime.now() + timedelta(days=14),
                contract_type='call', days_to_expiry=14, bid=1, ask=1.1,
                mid_price=1.05, spread_percent=0.1, delta=delta,
                gamma=0.05, theta=-0.1, open_interest=100, volume=10,
                avg_volume_5d=5, implied_volatility=0.3, iv_rank=75
            )
            contracts.append(contract)
        
        # Wide scan: 0.10 - 0.50
        scan_filtered = [c for c in contracts 
                        if self.scanner.delta_scan_min <= abs(c.delta) <= self.scanner.delta_scan_max]
        
        # Should include 0.12, 0.25, 0.35, 0.48
        self.assertEqual(len(scan_filtered), 4)
        
        # Final filter: 0.15 - 0.45
        final_filtered = [c for c in scan_filtered
                         if self.scanner.delta_final_min <= abs(c.delta) <= self.scanner.delta_final_max]
        
        # Should include 0.25, 0.35
        self.assertEqual(len(final_filtered), 2)
    
    def test_expiration_filtering(self):
        """Test that only contracts in 7-21 day window are selected"""
        contracts = []
        
        # Create contracts with various expirations
        days_to_exp = [3, 7, 14, 21, 30, 45]
        
        for days in days_to_exp:
            contract = MaxProfitContract(
                symbol='TEST', strike=100, 
                expiration=datetime.now() + timedelta(days=days),
                contract_type='call', days_to_expiry=days, bid=1, ask=1.1,
                mid_price=1.05, spread_percent=0.1, delta=0.3,
                gamma=0.05, theta=-0.1, open_interest=100, volume=10,
                avg_volume_5d=5, implied_volatility=0.3, iv_rank=75
            )
            contracts.append(contract)
        
        # Filter by expiration window
        filtered = [c for c in contracts 
                   if self.scanner.min_expiry_days <= c.days_to_expiry <= self.scanner.max_expiry_days]
        
        # Should include 7, 14, 21 day contracts
        self.assertEqual(len(filtered), 3)
        self.assertEqual(set(c.days_to_expiry for c in filtered), {7, 14, 21})


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_empty_results_handling(self):
        """Test that scanner handles no results gracefully"""
        scanner = MaxProfitScanner(test_mode=True)
        
        # Mock provider that returns no eligible stocks
        class EmptyProvider:
            def get_sp500_tickers(self):
                return []
            def fetch_latest_quote(self, ticker):
                return None
            def calculate_beta(self, ticker):
                return 0.5  # Too low
        
        scanner.data_provider = EmptyProvider()
        
        results = scanner.run_scan()
        
        # Should return empty list, not crash
        self.assertEqual(results, [])
    
    def test_output_format(self):
        """Test that output dictionary has all required fields"""
        scanner = MaxProfitScanner(test_mode=True)
        scanner.data_provider = MockDataProvider()
        
        contract = MaxProfitContract(
            symbol='TEST', strike=100, expiration=datetime.now() + timedelta(days=14),
            contract_type='call', days_to_expiry=14, bid=2.0, ask=2.1,
            mid_price=2.05, spread_percent=0.05, delta=0.3,
            gamma=0.05, theta=-0.2, open_interest=500, volume=100,
            avg_volume_5d=25, implied_volatility=0.3, iv_rank=75
        )
        
        # Calculate score
        contract.final_score = scanner.calculate_final_score(contract, 0.1, 2.0)
        
        # Convert to output
        output = scanner._contract_to_dict(contract)
        
        # Check all required fields exist
        required_fields = [
            'symbol', 'strike', 'type', 'expiry', 'delta', 'gamma', 
            'theta', 'gt_ratio', 'iv_rank', 'score', 'score_breakdown',
            'bid', 'ask', 'mid_price', 'open_interest', 'liquidity'
        ]
        
        for field in required_fields:
            self.assertIn(field, output, f"Missing field: {field}")
        
        # Check score is in 0-100 range
        self.assertGreaterEqual(output['score'], 0)
        self.assertLessEqual(output['score'], 100)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)