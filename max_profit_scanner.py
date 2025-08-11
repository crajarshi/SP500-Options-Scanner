"""
Maximum Profit Scanner - High-gamma options scanner for explosive opportunities
Uses normalized scoring with robust filtering for production-ready recommendations
"""
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time
import os
import pickle

import config

logger = logging.getLogger(__name__)


@dataclass
class MaxProfitContract:
    """Extended contract data for Maximum Profit analysis"""
    # Required fields (no defaults)
    symbol: str
    strike: float
    expiration: datetime
    contract_type: str  # 'call' or 'put'
    days_to_expiry: int
    bid: float
    ask: float
    mid_price: float
    spread_percent: float
    delta: float
    gamma: float
    theta: float
    open_interest: int
    volume: int
    implied_volatility: float
    
    # Optional fields with defaults (must come after required fields)
    vega: float = 0
    avg_volume_5d: float = 0
    iv_rank: float = 50
    gamma_theta_ratio: float = 0
    gamma_theta_normalized: float = 0
    liquidity_score: float = 0
    price_penalty: float = 0
    final_score: float = 0
    max_loss: float = 0
    position_size: int = 1
    score_breakdown: Dict[str, str] = field(default_factory=dict)
    raw_components: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.mid_price = (self.bid + self.ask) / 2 if self.bid and self.ask else 0
        self.spread_percent = ((self.ask - self.bid) / self.mid_price 
                               if self.mid_price > 0 else 1.0)
        self.max_loss = self.mid_price * 100  # Per contract


class MaxProfitScanner:
    """
    Production-ready Maximum Profit Scanner
    Identifies high-gamma, short-dated options with explosive potential
    """
    
    def __init__(self, data_provider=None, risk_manager=None, test_mode=False):
        """
        Initialize scanner with data provider and optional risk manager
        
        Args:
            data_provider: AlpacaDataProvider instance
            risk_manager: RiskManager instance for position sizing
            test_mode: If True, use test configuration
        """
        self.data_provider = data_provider
        self.risk_manager = risk_manager
        self.test_mode = test_mode
        
        # Load configuration
        self._load_config()
        
        # Initialize caches
        self.stock_cache = {}
        self.options_cache = {}
        self.skip_counter = {'total': 0}
        self.skip_reasons = []
        
        # Adaptive filtering
        self.current_mode = 'strict'
        self.near_misses = []
        self.momentum_cache = {}
        
        logger.info("MaxProfitScanner initialized with adaptive filtering")
    
    def _load_config(self):
        """Load configuration from config module"""
        # Stock filters
        self.beta_threshold = config.MAX_PROFIT_BETA_THRESHOLD
        self.iv_rank_threshold = config.MAX_PROFIT_IV_RANK_THRESHOLD
        self.min_stock_volume = config.MAX_PROFIT_MIN_STOCK_DAILY_VOLUME
        self.min_stock_price = config.MAX_PROFIT_MIN_STOCK_PRICE
        
        # Options filters
        self.delta_scan_min = config.MAX_PROFIT_DELTA_SCAN_MIN
        self.delta_scan_max = config.MAX_PROFIT_DELTA_SCAN_MAX
        self.delta_final_min = config.MAX_PROFIT_DELTA_FINAL_MIN
        self.delta_final_max = config.MAX_PROFIT_DELTA_FINAL_MAX
        
        self.min_expiry_days = config.MAX_PROFIT_MIN_EXPIRY_DAYS
        self.max_expiry_days = config.MAX_PROFIT_MAX_EXPIRY_DAYS
        
        # Liquidity requirements
        self.min_oi = config.MAX_PROFIT_MIN_OPTION_OI
        self.min_avg_volume = config.MAX_PROFIT_MIN_OPTION_AVG_VOLUME_5D
        self.max_spread_pct = config.MAX_PROFIT_MAX_SPREAD_PCT
        self.min_bid = config.MAX_PROFIT_MIN_BID
        
        # Scoring parameters
        self.oi_ref = config.MAX_PROFIT_OI_REF
        self.vol_ref = config.MAX_PROFIT_VOL_REF
        self.epsilon = config.MAX_PROFIT_EPSILON
        
        self.gtr_weight = config.MAX_PROFIT_GTR_WEIGHT
        self.ivr_weight = config.MAX_PROFIT_IVR_WEIGHT
        self.liq_weight = config.MAX_PROFIT_LIQ_WEIGHT
        
        self.liq_oi_weight = config.MAX_PROFIT_LIQ_OI_WEIGHT
        self.liq_vol_weight = config.MAX_PROFIT_LIQ_VOL_WEIGHT
        self.liq_spread_weight = config.MAX_PROFIT_LIQ_SPREAD_WEIGHT
        
        self.price_penalty_alpha = config.MAX_PROFIT_PRICE_PENALTY_ALPHA
        
        # Performance settings
        self.max_workers = config.MAX_PROFIT_MAX_WORKERS
        self.winsorize_pct = config.MAX_PROFIT_WINSORIZE_PCT
        self.top_results = config.MAX_PROFIT_TOP_RESULTS
        
        # Rate limiting settings
        self.rate_limit_delay = config.MAX_PROFIT_RATE_LIMIT_DELAY
        self.batch_size = config.MAX_PROFIT_BATCH_SIZE
        self.concurrent_quotes = config.MAX_PROFIT_CONCURRENT_QUOTES
        self.concurrent_options = config.MAX_PROFIT_CONCURRENT_OPTIONS
        
        # Output settings
        self.output_dir = config.MAX_PROFIT_OUTPUT_DIR
        self.log_skipped = config.MAX_PROFIT_LOG_SKIPPED
        self.cache_minutes = config.MAX_PROFIT_CACHE_MINUTES
        
        # Adaptive settings
        self.auto_adapt = config.MAX_PROFIT_AUTO_ADAPT
        self.show_near_misses = config.MAX_PROFIT_SHOW_NEAR_MISSES
        self.near_miss_count = config.MAX_PROFIT_NEAR_MISS_COUNT
    
    def update_thresholds(self, mode: str):
        """Update thresholds based on adaptive mode"""
        if mode == 'moderate':
            logger.info("Switching to MODERATE mode thresholds")
            self.beta_threshold = config.MAX_PROFIT_MODERATE_BETA
            self.iv_rank_threshold = config.MAX_PROFIT_MODERATE_IV_RANK
            self.min_stock_volume = config.MAX_PROFIT_MODERATE_VOLUME
            self.delta_final_min = config.MAX_PROFIT_MODERATE_DELTA_MIN
            self.delta_final_max = config.MAX_PROFIT_MODERATE_DELTA_MAX
            self.min_expiry_days = config.MAX_PROFIT_MODERATE_DTE_MIN
            self.max_expiry_days = config.MAX_PROFIT_MODERATE_DTE_MAX
        elif mode == 'relaxed':
            logger.info("Switching to RELAXED mode thresholds")
            self.beta_threshold = config.MAX_PROFIT_RELAXED_BETA
            self.iv_rank_threshold = config.MAX_PROFIT_RELAXED_IV_RANK
            self.min_stock_volume = config.MAX_PROFIT_RELAXED_VOLUME
            self.delta_final_min = config.MAX_PROFIT_RELAXED_DELTA_MIN
            self.delta_final_max = config.MAX_PROFIT_RELAXED_DELTA_MAX
            self.min_expiry_days = config.MAX_PROFIT_RELAXED_DTE_MIN
            self.max_expiry_days = config.MAX_PROFIT_RELAXED_DTE_MAX
        else:  # strict mode
            logger.info("Using STRICT mode thresholds")
            self.beta_threshold = config.MAX_PROFIT_BETA_THRESHOLD
            self.iv_rank_threshold = config.MAX_PROFIT_IV_RANK_THRESHOLD
            self.min_stock_volume = config.MAX_PROFIT_MIN_STOCK_DAILY_VOLUME
            self.delta_final_min = config.MAX_PROFIT_DELTA_FINAL_MIN
            self.delta_final_max = config.MAX_PROFIT_DELTA_FINAL_MAX
            self.min_expiry_days = config.MAX_PROFIT_MIN_EXPIRY_DAYS
            self.max_expiry_days = config.MAX_PROFIT_MAX_EXPIRY_DAYS
        
        self.current_mode = mode
    
    def calculate_final_score(self, contract: MaxProfitContract, 
                             gtr_min: float, gtr_max: float, 
                             momentum_score: float = 0.5,
                             earnings_boost: float = 0.0) -> float:
        """
        Calculate final normalized score using exact formula with momentum/earnings
        
        Returns:
            Score in 0..1 range (multiply by 100 for display)
        """
        # 1. Calculate GTR with stability guard
        gtr = contract.gamma / max(abs(contract.theta), self.epsilon)
        contract.gamma_theta_ratio = gtr
        
        # 2. Normalize GTR using batch min/max
        gtr_range = gtr_max - gtr_min if gtr_max > gtr_min else 1.0
        gtr_norm = np.clip((gtr - gtr_min) / (gtr_range + self.epsilon), 0, 1)
        contract.gamma_theta_normalized = gtr_norm
        
        # 3. IV Rank (already 0..1)
        ivr = contract.iv_rank / 100.0
        
        # 4. Calculate liquidity components
        oi_score = min(1, np.log(1 + contract.open_interest) / np.log(1 + self.oi_ref))
        vol_score = min(1, np.log(1 + contract.avg_volume_5d) / np.log(1 + self.vol_ref))
        spread_score = max(0, 1 - contract.spread_percent / self.max_spread_pct)
        
        liquidity = (self.liq_oi_weight * oi_score + 
                    self.liq_vol_weight * vol_score + 
                    self.liq_spread_weight * spread_score)
        contract.liquidity_score = liquidity
        
        # 5. Calculate raw score with enhanced weights if momentum available
        if momentum_score > 0 or earnings_boost > 0:
            # Use enhanced weights
            raw = (config.MAX_PROFIT_GTR_WEIGHT_ENHANCED * gtr_norm + 
                   config.MAX_PROFIT_IVR_WEIGHT_ENHANCED * ivr + 
                   config.MAX_PROFIT_LIQ_WEIGHT_ENHANCED * liquidity +
                   config.MAX_PROFIT_MOMENTUM_WEIGHT * momentum_score +
                   config.MAX_PROFIT_EARNINGS_WEIGHT * earnings_boost)
        else:
            # Use original weights
            raw = (self.gtr_weight * gtr_norm + 
                   self.ivr_weight * ivr + 
                   self.liq_weight * liquidity)
        
        # 6. Apply price penalty (multiplicative)
        # Note: Price penalty should reduce score more for expensive options
        price_penalty = 1 / (1 + np.log(1 + contract.mid_price))
        contract.price_penalty = price_penalty
        
        # Apply penalty as reduction factor (higher price = lower score)
        penalty_factor = 1 - (self.price_penalty_alpha * (1 - price_penalty))
        final_score_0_1 = raw * penalty_factor
        
        # 7. Store breakdown for transparency (as percentages)
        contract.score_breakdown = {
            'GTR': f"{gtr_norm * self.gtr_weight * 100:.1f}%",
            'IVR': f"{ivr * self.ivr_weight * 100:.1f}%",
            'LIQ': f"{liquidity * self.liq_weight * 100:.1f}%",
            'PRICE_ADJ': f"-{price_penalty * self.price_penalty_alpha * 100:.1f}%"
        }
        
        # 8. Store raw components for backtesting
        contract.raw_components = {
            'gtr': gtr,
            'gtr_norm': gtr_norm,
            'ivr': ivr,
            'oi_score': oi_score,
            'vol_score': vol_score,
            'spread_score': spread_score,
            'liquidity': liquidity,
            'price_penalty': price_penalty
        }
        
        return final_score_0_1
    
    def calculate_liquidity_score(self, contract: MaxProfitContract) -> float:
        """
        Calculate transparent liquidity score with configurable weights
        """
        oi_score = min(1, np.log(1 + contract.open_interest) / np.log(1 + self.oi_ref))
        vol_score = min(1, np.log(1 + contract.avg_volume_5d) / np.log(1 + self.vol_ref))
        spread_score = max(0, 1 - contract.spread_percent / self.max_spread_pct)
        
        return (self.liq_oi_weight * oi_score + 
                self.liq_vol_weight * vol_score + 
                self.liq_spread_weight * spread_score)
    
    def winsorize_gtr_values(self, contracts: List[MaxProfitContract]) -> Tuple[float, float]:
        """
        Winsorize GTR values to handle outliers
        
        Returns:
            (min, max) after winsorization
        """
        if not contracts:
            return 0, 1
        
        gtr_values = [c.gamma / max(abs(c.theta), self.epsilon) for c in contracts]
        
        if len(gtr_values) < 3:
            # Not enough values for percentile calculation
            return float(np.min(gtr_values)), float(np.max(gtr_values))
        
        # Calculate percentiles for winsorization (2nd and 98th percentile)
        lower_pct = self.winsorize_pct * 100
        upper_pct = (1 - self.winsorize_pct) * 100
        
        # For small samples, use more aggressive winsorization
        if len(gtr_values) <= 10:
            lower_pct = 10  # 10th percentile
            upper_pct = 90  # 90th percentile
        
        lower = np.percentile(gtr_values, lower_pct)
        upper = np.percentile(gtr_values, upper_pct)
        
        # Clip values
        winsorized = np.clip(gtr_values, lower, upper)
        
        return float(np.min(winsorized)), float(np.max(winsorized))
    
    def pre_filter_stocks(self, tickers: List[str]) -> List[str]:
        """
        Pre-filter stocks by beta and volume with rate limiting
        """
        logger.info(f"Pre-filtering {len(tickers)} stocks...")
        filtered = []
        failed_symbols = []
        
        # Use ThreadPoolExecutor with limited workers for rate limiting
        max_concurrent = min(self.concurrent_quotes, self.max_workers)  # Use configured limit
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(self._check_stock_eligibility, ticker): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        filtered.append(ticker)
                    elif result is False:
                        failed_symbols.append(ticker)
                except Exception as e:
                    logger.warning(f"Error checking {ticker}: {e}")
                    failed_symbols.append(ticker)
                
                # Small delay between batches to avoid rate limiting
                time.sleep(0.05)  # 50ms between requests
        
        if failed_symbols:
            logger.info(f"Failed to check {len(failed_symbols)} symbols: {failed_symbols[:10]}...")
        
        logger.info(f"Pre-filtered to {len(filtered)} eligible stocks from {len(tickers)} total")
        return filtered
    
    def _check_stock_eligibility(self, ticker: str) -> bool:
        """
        Check if stock meets basic criteria (beta, volume, price)
        Returns True if eligible, False if not, None if error
        """
        try:
            # Check cached values first
            if ticker in self.stock_cache:
                return self.stock_cache[ticker]['eligible']
            
            # Get latest quote for price and volume check
            quote = self.data_provider.fetch_latest_quote(ticker)
            if not quote:
                self._log_skip(ticker, "no_quote")
                # Cache as ineligible to avoid re-checking
                self.stock_cache[ticker] = {'eligible': False, 'reason': 'no_quote'}
                return False
            
            # Extract price and volume, handling different quote formats
            # Handle both direct values and nested structures
            if isinstance(quote, dict):
                # Try different possible field names
                price = quote.get('price') or quote.get('ap') or quote.get('askPrice', 0)
                volume = quote.get('volume') or quote.get('v') or quote.get('dayVolume', 0)
            else:
                price = 0
                volume = 0
            
            # Check minimum price
            if price < self.min_stock_price:
                self._log_skip(ticker, "price_too_low", price)
                self.stock_cache[ticker] = {'eligible': False, 'reason': 'price_too_low'}
                return False
            
            # Check volume
            if volume < self.min_stock_volume:
                self._log_skip(ticker, "volume_too_low", volume)
                self.stock_cache[ticker] = {'eligible': False, 'reason': 'volume_too_low'}
                return False
            
            # Calculate beta (expensive operation) - do this last
            beta = self.data_provider.calculate_beta(ticker)
            if beta < self.beta_threshold:
                self._log_skip(ticker, "beta_too_low", beta)
                self.stock_cache[ticker] = {'eligible': False, 'reason': 'beta_too_low', 'beta': beta}
                return False
            
            # Cache successful result
            self.stock_cache[ticker] = {
                'eligible': True,
                'beta': beta,
                'volume': volume,
                'price': price
            }
            
            logger.debug(f"{ticker} eligible: beta={beta:.2f}, price=${price:.2f}, vol={volume:,.0f}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking eligibility for {ticker}: {e}")
            # Don't cache errors - might be temporary
            return False
    
    def fetch_options_parallel(self, tickers: List[str]) -> Dict[str, List[MaxProfitContract]]:
        """
        Fetch option chains in parallel with rate limiting
        """
        logger.info(f"Fetching options for {len(tickers)} stocks in parallel...")
        all_contracts = {}
        failed_tickers = []
        
        # Limit concurrent requests to avoid rate limiting
        max_concurrent = min(self.concurrent_options, self.max_workers)  # Use configured limit
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(self._fetch_and_filter_options, ticker): ticker 
                for ticker in tickers
            }
            
            completed = 0
            for future in as_completed(futures):
                ticker = futures[future]
                completed += 1
                
                try:
                    contracts = future.result()
                    if contracts:
                        all_contracts[ticker] = contracts
                        logger.debug(f"Found {len(contracts)} contracts for {ticker}")
                    else:
                        logger.debug(f"No valid contracts for {ticker}")
                except Exception as e:
                    logger.error(f"Error fetching options for {ticker}: {e}")
                    failed_tickers.append(ticker)
                
                # Progress update
                if completed % 10 == 0:
                    logger.info(f"Processed {completed}/{len(tickers)} stocks...")
                
                # Rate limiting between batches
                if completed % 5 == 0:
                    time.sleep(0.5)  # 500ms pause every 5 requests
        
        if failed_tickers:
            logger.warning(f"Failed to fetch options for {len(failed_tickers)} stocks")
        
        logger.info(f"Successfully fetched options for {len(all_contracts)} stocks")
        return all_contracts
    
    def _fetch_and_filter_options(self, ticker: str) -> List[MaxProfitContract]:
        """
        Fetch options for a ticker and apply initial filters
        """
        contracts = []
        
        try:
            # Get stock info from cache
            stock_info = self.stock_cache.get(ticker, {})
            current_price = stock_info.get('price', 100)
            
            # Fetch chain
            chain_data = self.data_provider.fetch_options_chain(ticker)
            if not chain_data:
                return contracts
            
            # Process each expiration and strike
            for exp_str, strikes in chain_data.items():
                # Parse expiration
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                except:
                    continue
                
                # Check expiration window
                days_to_exp = (exp_date - datetime.now()).days
                if not (self.min_expiry_days <= days_to_exp <= self.max_expiry_days):
                    continue
                
                for strike, data in strikes.items():
                    # Process both calls and puts
                    for contract_type in ['call', 'put']:
                        if contract_type not in data:
                            continue
                        
                        contract_data = data[contract_type]
                        
                        # Check delta (wide scan range)
                        delta = abs(contract_data.get('delta', 0))
                        if not (self.delta_scan_min <= delta <= self.delta_scan_max):
                            continue
                        
                        # Quick liquidity check
                        oi = contract_data.get('open_interest', 0)
                        if oi < self.min_oi:
                            self._log_skip(f"{ticker}_{strike}_{contract_type}", "low_oi", oi)
                            continue
                        
                        # Check bid/ask
                        bid = contract_data.get('bid', 0)
                        ask = contract_data.get('ask', 0)
                        
                        if bid < self.min_bid:
                            self._log_skip(f"{ticker}_{strike}_{contract_type}", "low_bid", bid)
                            continue
                        
                        if ask <= 0 or bid <= 0:
                            continue
                        
                        # Check spread
                        mid = (bid + ask) / 2
                        spread_pct = (ask - bid) / mid if mid > 0 else 1
                        
                        if spread_pct > self.max_spread_pct:
                            self._log_skip(f"{ticker}_{strike}_{contract_type}", "wide_spread", spread_pct)
                            continue
                        
                        # Check for required Greeks
                        if 'gamma' not in contract_data or 'theta' not in contract_data:
                            self._log_skip(f"{ticker}_{strike}_{contract_type}", "missing_greeks")
                            continue
                        
                        # Create contract object
                        contract = MaxProfitContract(
                            symbol=ticker,
                            strike=strike,
                            expiration=exp_date,
                            contract_type=contract_type,
                            days_to_expiry=days_to_exp,
                            bid=bid,
                            ask=ask,
                            mid_price=mid,
                            spread_percent=spread_pct,
                            delta=contract_data.get('delta', 0),
                            gamma=contract_data.get('gamma', 0),
                            theta=contract_data.get('theta', 0),
                            vega=contract_data.get('vega', 0),
                            open_interest=oi,
                            volume=contract_data.get('volume', 0),
                            avg_volume_5d=contract_data.get('volume', 0),  # TODO: Calculate 5-day average
                            implied_volatility=contract_data.get('implied_volatility', 0.3),
                            iv_rank=self._calculate_iv_rank(contract_data.get('implied_volatility', 0.3))
                        )
                        
                        contracts.append(contract)
            
        except Exception as e:
            logger.error(f"Error processing options for {ticker}: {e}")
        
        return contracts
    
    def _calculate_iv_rank(self, current_iv: float) -> float:
        """
        Calculate IV rank (Phase 1: approximation)
        TODO: Phase 2 - Store and calculate from historical IV data
        """
        # Rough approximation based on typical IV ranges
        if current_iv < 0.20:
            return 10
        elif current_iv < 0.30:
            return 30
        elif current_iv < 0.40:
            return 50
        elif current_iv < 0.60:
            return 70
        else:
            return 90
    
    def calculate_momentum_score(self, ticker: str) -> float:
        """
        Calculate momentum score based on technical indicators
        Returns: Score between 0 and 1
        """
        if ticker in self.momentum_cache:
            return self.momentum_cache[ticker]
        
        try:
            # Get recent price data
            df = self.data_provider.fetch_bars(ticker, timeframe='15Min', days_back=3)
            if df is None or len(df) < 30:
                return 0.5  # Neutral if no data
            
            # Simple momentum indicators
            close_prices = df['close'].values
            
            # 1. Price trend (compare to 10-period average)
            ma10 = np.mean(close_prices[-10:])
            current = close_prices[-1]
            trend_score = min(1.0, max(0, (current - ma10) / ma10 + 0.5))
            
            # 2. Relative strength (simplified RSI)
            gains = []
            losses = []
            for i in range(1, min(14, len(close_prices))):
                diff = close_prices[i] - close_prices[i-1]
                if diff > 0:
                    gains.append(diff)
                else:
                    losses.append(abs(diff))
            
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 1
            rs = avg_gain / avg_loss if avg_loss > 0 else 1
            rsi = 1 - (1 / (1 + rs))  # Normalized to 0-1
            
            # 3. Volume surge (if available)
            if 'volume' in df.columns:
                volumes = df['volume'].values
                recent_vol = np.mean(volumes[-5:])
                avg_vol = np.mean(volumes)
                vol_score = min(1.0, recent_vol / (avg_vol + 1))
            else:
                vol_score = 0.5
            
            # Combine scores
            momentum = (trend_score * 0.4 + rsi * 0.4 + vol_score * 0.2)
            
            self.momentum_cache[ticker] = momentum
            return momentum
            
        except Exception as e:
            logger.debug(f"Error calculating momentum for {ticker}: {e}")
            return 0.5  # Default neutral score
    
    def _log_skip(self, identifier: str, reason: str, value: any = None, contract: MaxProfitContract = None):
        """Log why a contract was skipped and track near misses"""
        if self.log_skipped:
            self.skip_counter['total'] += 1
            self.skip_counter[reason] = self.skip_counter.get(reason, 0) + 1
            
            if len(self.skip_reasons) < 1000:  # Limit memory usage
                self.skip_reasons.append({
                    'id': identifier,
                    'reason': reason,
                    'value': value,
                    'timestamp': datetime.now()
                })
            
            # Track near misses
            if contract and self.show_near_misses:
                self._check_near_miss(contract, reason, value)
    
    def _check_near_miss(self, contract: MaxProfitContract, reason: str, value: any):
        """Check if contract is a near miss (fails only 1-2 criteria)"""
        miss_count = 0
        miss_details = []
        
        # Check each criterion
        if abs(contract.delta) < self.delta_final_min * 0.9 or abs(contract.delta) > self.delta_final_max * 1.1:
            miss_count += 1
            miss_details.append(f"Delta: {contract.delta:.2f} (need {self.delta_final_min:.2f}-{self.delta_final_max:.2f})")
        
        if contract.iv_rank < self.iv_rank_threshold * 0.9:
            miss_count += 1
            miss_details.append(f"IVR: {contract.iv_rank:.0f}% (need {self.iv_rank_threshold:.0f}%)")
        
        if contract.days_to_expiry < self.min_expiry_days or contract.days_to_expiry > self.max_expiry_days:
            miss_count += 1
            miss_details.append(f"DTE: {contract.days_to_expiry} (need {self.min_expiry_days}-{self.max_expiry_days})")
        
        # If only 1-2 criteria failed, track as near miss
        if miss_count <= 2 and len(self.near_misses) < self.near_miss_count:
            self.near_misses.append({
                'contract': contract,
                'failures': miss_details,
                'miss_count': miss_count
            })
    
    def run_adaptive_scan(self) -> List[Dict]:
        """
        Adaptive scan that automatically relaxes criteria if no results found
        """
        logger.info("=" * 60)
        logger.info("Starting ADAPTIVE Maximum Profit Scanner")
        logger.info("Will automatically adjust thresholds if needed...")
        logger.info("=" * 60)
        
        modes = ['strict', 'moderate', 'relaxed']
        all_results = []
        
        for mode in modes:
            self.update_thresholds(mode)
            logger.info(f"\n🔍 Trying {mode.upper()} mode...")
            
            results = self.run_scan()
            
            if results:
                logger.info(f"✅ Found {len(results)} opportunities in {mode} mode")
                # Add mode tag to results
                for r in results:
                    r['filter_mode'] = mode.upper()
                all_results.extend(results)
                
                # If we found enough in strict/moderate, stop
                if mode != 'relaxed' and len(all_results) >= 3:
                    break
            else:
                logger.info(f"No results in {mode} mode, trying next level...")
        
        # Include ETFs if still no results
        if not all_results and config.MAX_PROFIT_ETFS:
            logger.info("\n🔍 Adding high-volatility ETFs to search...")
            etf_results = self._scan_etfs()
            if etf_results:
                for r in etf_results:
                    r['filter_mode'] = 'ETF'
                all_results.extend(etf_results)
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = all_results[:self.top_results]
        
        # Add near misses if enabled
        if self.show_near_misses and self.near_misses:
            logger.info(f"\n📊 Also found {len(self.near_misses)} near-miss contracts")
        
        return top_results
    
    def _scan_etfs(self) -> List[Dict]:
        """Scan high-volatility ETFs with relaxed criteria"""
        self.update_thresholds('relaxed')
        # Override beta requirement for ETFs
        original_beta = self.beta_threshold
        self.beta_threshold = 0.8  # Lower beta for ETFs
        
        results = []
        for etf in config.MAX_PROFIT_ETFS:
            try:
                contracts = self.fetch_option_contracts(etf)
                if contracts:
                    # Process similar to regular scan
                    for contract in contracts:
                        result = self._process_contract(contract)
                        if result:
                            results.append(result)
            except Exception as e:
                logger.debug(f"Error scanning ETF {etf}: {e}")
        
        self.beta_threshold = original_beta
        return results
    
    def _process_contract(self, contract: MaxProfitContract) -> Optional[Dict]:
        """Process a single contract and return formatted result if valid"""
        try:
            # Check basic criteria
            if contract.delta < self.delta_final_min or contract.delta > self.delta_final_max:
                return None
            if contract.days_to_expiry < self.min_expiry_days or contract.days_to_expiry > self.max_expiry_days:
                return None
            if contract.iv_rank < self.iv_rank_threshold:
                return None
            
            # Calculate score (simplified for single contract)
            gtr = contract.gamma / max(abs(contract.theta), self.epsilon)
            contract.gamma_theta_ratio = gtr
            contract.final_score = self.calculate_final_score(contract, gtr * 0.8, gtr * 1.2)
            
            # Format result
            return self._contract_to_dict(contract)
        except Exception as e:
            logger.debug(f"Error processing contract: {e}")
            return None
    
    def run_scan(self) -> List[Dict]:
        """
        Main scan execution with all optimizations
        """
        logger.info("Starting scan with current thresholds...")
        
        scan_start = time.time()
        
        # Reset counters
        self.skip_counter = {'total': 0}
        self.skip_reasons = []
        
        # 1. Get initial ticker list
        if not self.data_provider:
            logger.error("No data provider configured")
            return []
        
        tickers = self.data_provider.get_sp500_tickers()
        logger.info(f"Starting with {len(tickers)} S&P 500 stocks")
        
        # 2. Pre-filter by beta and volume
        eligible_tickers = self.pre_filter_stocks(tickers)
        
        if not eligible_tickers:
            logger.warning("No stocks passed pre-filtering")
            return []
        
        # 3. Fetch options in parallel
        all_contracts_by_ticker = self.fetch_options_parallel(eligible_tickers)
        
        # 4. Flatten all contracts
        all_contracts = []
        for ticker_contracts in all_contracts_by_ticker.values():
            all_contracts.extend(ticker_contracts)
        
        logger.info(f"Collected {len(all_contracts)} contracts for scoring")
        
        if not all_contracts:
            logger.warning("No contracts found matching criteria")
            return []
        
        # 5. Apply final delta filter
        final_contracts = [
            c for c in all_contracts 
            if self.delta_final_min <= abs(c.delta) <= self.delta_final_max
        ]
        
        logger.info(f"After final delta filter: {len(final_contracts)} contracts")
        
        if not final_contracts:
            logger.warning("No contracts passed final delta filter")
            return []
        
        # 6. Filter by IV rank threshold
        high_iv_contracts = [
            c for c in final_contracts
            if c.iv_rank >= self.iv_rank_threshold
        ]
        
        logger.info(f"After IV rank filter: {len(high_iv_contracts)} contracts")
        
        if not high_iv_contracts:
            logger.warning("No contracts with sufficient IV rank")
            return []
        
        # 7. Calculate GTR min/max with winsorization
        gtr_min, gtr_max = self.winsorize_gtr_values(high_iv_contracts)
        logger.info(f"GTR range after winsorization: {gtr_min:.2f} - {gtr_max:.2f}")
        
        # 8. Score all contracts with momentum
        for contract in high_iv_contracts:
            # Get momentum score for the ticker
            momentum = self.calculate_momentum_score(contract.symbol)
            contract.final_score = self.calculate_final_score(contract, gtr_min, gtr_max, momentum_score=momentum)
        
        # 9. Select top contract per ticker
        best_by_ticker = {}
        for contract in high_iv_contracts:
            if (contract.symbol not in best_by_ticker or 
                contract.final_score > best_by_ticker[contract.symbol].final_score):
                best_by_ticker[contract.symbol] = contract
        
        # 10. Sort and take top N
        top_contracts = sorted(
            best_by_ticker.values(), 
            key=lambda x: x.final_score, 
            reverse=True
        )[:self.top_results]
        
        # 11. Convert to output format
        results = [self._contract_to_dict(c) for c in top_contracts]
        
        # 12. Save results
        if results:
            self._save_results(results, all_contracts)
        
        # Log summary
        scan_duration = time.time() - scan_start
        logger.info("=" * 60)
        logger.info(f"Scan completed in {scan_duration:.1f} seconds")
        logger.info(f"Evaluated {len(all_contracts)} total contracts")
        logger.info(f"Skipped {self.skip_counter['total']} contracts")
        
        if self.skip_counter:
            logger.info("Skip reasons:")
            for reason, count in self.skip_counter.items():
                if reason != 'total':
                    logger.info(f"  {reason}: {count}")
        
        logger.info(f"Found {len(results)} top opportunities")
        logger.info("=" * 60)
        
        return results
    
    def _contract_to_dict(self, contract: MaxProfitContract) -> Dict:
        """Convert contract object to dictionary for output"""
        return {
            'symbol': contract.symbol,
            'strike': contract.strike,
            'type': contract.contract_type,
            'expiry': contract.expiration.strftime('%Y-%m-%d'),
            'days_to_expiry': contract.days_to_expiry,
            'delta': abs(contract.delta),
            'gamma': contract.gamma,
            'theta': contract.theta,
            'gt_ratio': contract.gamma_theta_ratio,
            'iv': contract.implied_volatility,
            'iv_rank': contract.iv_rank,
            'bid': contract.bid,
            'ask': contract.ask,
            'mid_price': contract.mid_price,
            'spread_pct': contract.spread_percent,
            'open_interest': contract.open_interest,
            'volume': contract.volume,
            'liquidity': contract.liquidity_score * 100,
            'score': contract.final_score * 100,  # Convert to 0-100 scale
            'score_breakdown': contract.score_breakdown,
            'max_loss': contract.max_loss,
            'position_size': contract.position_size
        }
    
    def _save_results(self, top_results: List[Dict], all_contracts: List[MaxProfitContract]):
        """
        Save results and raw data for analysis
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            
            # Save top results to CSV
            if top_results:
                top_df = pd.DataFrame(top_results)
                csv_path = os.path.join(self.output_dir, f"top_opportunities_{timestamp}.csv")
                top_df.to_csv(csv_path, index=False)
                logger.info(f"Saved top results to {csv_path}")
            
            # Save all evaluated contracts with raw components for backtesting
            all_data = []
            for contract in all_contracts:
                data = {
                    'symbol': contract.symbol,
                    'strike': contract.strike,
                    'expiry': contract.expiration.strftime('%Y-%m-%d'),
                    'type': contract.contract_type,
                    'final_score': contract.final_score,
                }
                # Add raw components if available
                if hasattr(contract, 'raw_components'):
                    data.update(contract.raw_components)
                all_data.append(data)
            
            if all_data:
                all_df = pd.DataFrame(all_data)
                parquet_path = os.path.join(self.output_dir, f"all_contracts_{timestamp}.parquet")
                all_df.to_parquet(parquet_path, index=False)
                logger.info(f"Saved all contracts to {parquet_path}")
                
                # Log summary statistics
                logger.info(f"Score statistics:")
                logger.info(f"  Min: {all_df['final_score'].min():.3f}")
                logger.info(f"  Max: {all_df['final_score'].max():.3f}")
                logger.info(f"  Mean: {all_df['final_score'].mean():.3f}")
                logger.info(f"  Median: {all_df['final_score'].median():.3f}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")