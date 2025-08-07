"""
Options chain analysis module for recommending specific contracts
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import time

import config

logger = logging.getLogger(__name__)


@dataclass
class OptionsContract:
    """Data class for options contract information"""
    symbol: str
    strike: float
    expiration: str
    contract_type: str  # 'call' or 'put'
    delta: float
    bid: float
    ask: float
    mid_price: float
    spread: float
    spread_percent: float
    open_interest: int
    volume: int
    implied_volatility: float
    days_to_expiry: int
    liquidity_score: float  # 0-100 score based on spread and OI
    
    def is_liquid(self) -> bool:
        """Check if contract meets liquidity requirements"""
        return (self.open_interest >= config.OPTIONS_MIN_OPEN_INTEREST and 
                self.spread_percent <= config.OPTIONS_MAX_SPREAD_PERCENT)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for display"""
        return {
            'strike': self.strike,
            'expiration': self.expiration,
            'type': self.contract_type.upper(),
            'delta': self.delta,
            'bid': self.bid,
            'ask': self.ask,
            'mid': self.mid_price,
            'spread%': self.spread_percent * 100,
            'OI': self.open_interest,
            'volume': self.volume,
            'IV': self.implied_volatility,
            'DTE': self.days_to_expiry,
            'liquidity': self.liquidity_score
        }


class OptionsChainAnalyzer:
    """
    Analyzes options chains to find optimal contracts based on:
    - Target delta (0.70 for calls, -0.70 for puts)
    - Expiration window (30-60 days)
    - Liquidity requirements (OI > 100, spread < 10%)
    - Monthly expirations preferred
    """
    
    def __init__(self, data_provider):
        """
        Initialize with data provider
        
        Args:
            data_provider: AlpacaDataProvider instance with options methods
        """
        self.data_provider = data_provider
        self.cache = {}  # Cache for options chains
        self.cache_expiry = {}  # Cache expiry times
        
    def get_optimal_contracts(self, symbol: str, signal_type: str, 
                             stock_price: float) -> Optional[List[OptionsContract]]:
        """
        Get optimal options contracts for a stock based on signal type
        
        Args:
            symbol: Stock symbol
            signal_type: 'BULLISH' or 'BEARISH'
            stock_price: Current stock price
            
        Returns:
            List of top 3 optimal contracts or None if no suitable contracts
        """
        # Define signal categories
        actionable_bullish = ['STRONG_BUY', 'BUY', 'BULLISH']
        actionable_bearish = ['STRONG_SELL', 'SELL', 'BEARISH']
        neutral_signals = ['NEUTRAL', 'NEUTRAL_BULL', 'NEUTRAL_BEAR', 'HOLD', 
                          'NEUTRAL+', 'NEUTRAL-', 'NO ACTION', 'WEAK_BUY', 'WEAK_SELL']
        
        try:
            # Determine contract type and delta based on signal
            if signal_type in actionable_bullish:
                contract_type = 'call'
                target_delta = config.OPTIONS_CALL_DELTA
                logger.info(f"Fetching CALL options for {symbol} (bullish signal)")
            elif signal_type in actionable_bearish:
                contract_type = 'put'
                target_delta = config.OPTIONS_PUT_DELTA
                logger.info(f"Fetching PUT options for {symbol} (bearish signal)")
            elif signal_type in neutral_signals or 'NEUTRAL' in signal_type.upper():
                # For neutral signals, recommend ATM options (both calls and puts useful)
                contract_type = 'both'  # We'll look at both calls and puts
                target_delta = config.OPTIONS_NEUTRAL_DELTA  # 0.50 for ATM
                logger.info(f"Fetching ATM options for {symbol} (neutral signal)")
            else:
                # Unknown signal type
                logger.warning(f"Unknown signal type '{signal_type}' for {symbol}, treating as neutral")
                contract_type = 'both'
                target_delta = config.OPTIONS_NEUTRAL_DELTA
            
            # Fetch options chain (with caching)
            options_chain = self._fetch_options_chain(symbol)
            if not options_chain:
                logger.warning(f"No options chain available for {symbol}")
                return None
            
            # Filter by contract type
            if contract_type == 'both':
                # For neutral signals, consider both calls and puts
                contracts = options_chain
            else:
                contracts = [c for c in options_chain if c.contract_type == contract_type]
            
            # Filter by expiration window and get the SINGLE best expiration
            best_expiration_contracts = self._get_best_expiration_contracts(contracts)
            
            if not best_expiration_contracts:
                logger.info(f"No contracts in target expiration window for {symbol}")
                return None
            
            # Filter by liquidity
            liquid_contracts = [c for c in best_expiration_contracts if c.is_liquid()]
            
            if not liquid_contracts:
                logger.info(f"No liquid {contract_type} contracts for {symbol}")
                # Fallback to less liquid contracts if needed
                liquid_contracts = self._relax_liquidity_requirements(best_expiration_contracts)
                if not liquid_contracts:
                    return None
            
            # Find the SINGLE contract closest to target delta
            optimal_contract = self._find_single_best_contract(
                liquid_contracts, target_delta, stock_price, contract_type
            )
            
            # Return as a list with single contract for compatibility
            return [optimal_contract] if optimal_contract else None
            
        except Exception as e:
            logger.error(f"Error getting optimal contracts for {symbol}: {e}")
            return None
    
    def _fetch_options_chain(self, symbol: str) -> Optional[List[OptionsContract]]:
        """
        Fetch options chain with caching
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of OptionsContract objects or None
        """
        # Check cache
        cache_key = f"{symbol}_chain"
        if cache_key in self.cache:
            expiry_time = self.cache_expiry.get(cache_key, datetime.min)
            if datetime.now() < expiry_time:
                logger.debug(f"Using cached options chain for {symbol}")
                return self.cache[cache_key]
        
        # Fetch from API
        logger.info(f"Fetching options chain for {symbol}")
        
        # Add delay to respect rate limits
        time.sleep(config.OPTIONS_API_DELAY)
        
        # Call data provider's options method
        chain_data = self.data_provider.fetch_options_chain(symbol)
        
        if not chain_data:
            return None
        
        # Parse into OptionsContract objects
        contracts = self._parse_options_data(chain_data, symbol)
        
        # Cache the results
        self.cache[cache_key] = contracts
        self.cache_expiry[cache_key] = datetime.now() + timedelta(
            minutes=config.OPTIONS_CACHE_MINUTES
        )
        
        return contracts
    
    def _parse_options_data(self, chain_data: Dict, symbol: str) -> List[OptionsContract]:
        """
        Parse raw options data into OptionsContract objects
        
        Args:
            chain_data: Raw options chain data from API
            symbol: Stock symbol
            
        Returns:
            List of OptionsContract objects
        """
        contracts = []
        
        for expiration_date, strikes in chain_data.items():
            # Calculate days to expiry
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            days_to_expiry = (exp_date - datetime.now()).days
            
            # Skip if outside our window
            if days_to_expiry < 0 or days_to_expiry > 90:
                continue
            
            for strike_price, contract_data in strikes.items():
                # Process calls
                if 'call' in contract_data:
                    call = contract_data['call']
                    contracts.append(self._create_contract(
                        symbol, float(strike_price), expiration_date, 
                        'call', call, days_to_expiry
                    ))
                
                # Process puts
                if 'put' in contract_data:
                    put = contract_data['put']
                    contracts.append(self._create_contract(
                        symbol, float(strike_price), expiration_date,
                        'put', put, days_to_expiry
                    ))
        
        return contracts
    
    def _create_contract(self, symbol: str, strike: float, expiration: str,
                        contract_type: str, data: Dict, days_to_expiry: int) -> OptionsContract:
        """
        Create OptionsContract object from raw data
        
        Args:
            symbol: Stock symbol
            strike: Strike price
            expiration: Expiration date
            contract_type: 'call' or 'put'
            data: Raw contract data
            days_to_expiry: Days until expiration
            
        Returns:
            OptionsContract object
        """
        bid = data.get('bid', 0)
        ask = data.get('ask', 0)
        
        # Calculate mid price and spread
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
            spread = ask - bid
            spread_percent = spread / mid_price if mid_price > 0 else 1.0
        else:
            mid_price = data.get('last', 0)
            spread = 0
            spread_percent = 1.0  # Max spread for illiquid
        
        # Calculate liquidity score (0-100)
        liquidity_score = self._calculate_liquidity_score(
            data.get('open_interest', 0),
            spread_percent,
            data.get('volume', 0)
        )
        
        return OptionsContract(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            contract_type=contract_type,
            delta=data.get('delta', 0),
            bid=bid,
            ask=ask,
            mid_price=mid_price,
            spread=spread,
            spread_percent=spread_percent,
            open_interest=data.get('open_interest', 0),
            volume=data.get('volume', 0),
            implied_volatility=data.get('implied_volatility', 0),
            days_to_expiry=days_to_expiry,
            liquidity_score=liquidity_score
        )
    
    def _calculate_liquidity_score(self, open_interest: int, 
                                  spread_percent: float, volume: int) -> float:
        """
        Calculate liquidity score from 0-100
        
        Args:
            open_interest: Open interest
            spread_percent: Bid-ask spread as percentage
            volume: Daily volume
            
        Returns:
            Liquidity score 0-100
        """
        # Open interest score (40% weight)
        oi_score = min(100, (open_interest / 500) * 100) * 0.4
        
        # Spread score (40% weight) - lower is better
        spread_score = max(0, 100 - (spread_percent * 1000)) * 0.4
        
        # Volume score (20% weight)
        vol_score = min(100, (volume / 100) * 100) * 0.2
        
        return oi_score + spread_score + vol_score
    
    def _get_best_expiration_contracts(self, contracts: List[OptionsContract]) -> List[OptionsContract]:
        """
        Get contracts from the SINGLE best expiration date closest to 45 days
        
        Args:
            contracts: List of all contracts
            
        Returns:
            Contracts from the single best expiration date
        """
        # Filter to target window
        filtered = []
        for contract in contracts:
            if (config.OPTIONS_MIN_DAYS <= contract.days_to_expiry <= 
                config.OPTIONS_MAX_DAYS):
                filtered.append(contract)
        
        if not filtered:
            return []
        
        # Group by expiration date
        expirations = {}
        for contract in filtered:
            if contract.expiration not in expirations:
                expirations[contract.expiration] = []
            expirations[contract.expiration].append(contract)
        
        # Find expiration closest to target (45 days)
        target_days = config.OPTIONS_TARGET_DAYS
        best_expiration = None
        min_distance = float('inf')
        
        for exp_date, exp_contracts in expirations.items():
            # Use the first contract to get days to expiry
            days = exp_contracts[0].days_to_expiry
            distance = abs(days - target_days)
            
            # Prefer monthly expirations if distance is similar
            is_monthly = self._is_monthly_expiration(exp_date)
            
            if distance < min_distance or (distance == min_distance and is_monthly):
                min_distance = distance
                best_expiration = exp_date
        
        # Return contracts from the single best expiration
        return expirations.get(best_expiration, [])
    
    def _is_monthly_expiration(self, expiration: str) -> bool:
        """
        Check if expiration is a monthly (3rd Friday of month)
        
        Args:
            expiration: Expiration date string
            
        Returns:
            True if monthly expiration
        """
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        
        # Check if it's a Friday
        if exp_date.weekday() == 4:  # Friday is 4
            # Check if it's the 3rd Friday
            day = exp_date.day
            if 15 <= day <= 21:  # 3rd Friday falls in this range
                return True
        
        return False
    
    def _find_single_best_contract(self, contracts: List[OptionsContract], 
                                  target_delta: float, stock_price: float,
                                  contract_type: str) -> Optional[OptionsContract]:
        """
        Find the SINGLE best contract closest to target delta
        
        Args:
            contracts: List of liquid contracts from single expiration
            target_delta: Target delta value
            stock_price: Current stock price
            contract_type: 'call', 'put', or 'both'
            
        Returns:
            Single best contract or None
        """
        if not contracts:
            return None
        
        # For neutral signals (ATM), find the single best ATM option
        if abs(target_delta) == 0.50:
            # Find contract with strike closest to current price
            best_contract = None
            min_strike_distance = float('inf')
            
            for contract in contracts:
                strike_distance = abs(contract.strike - stock_price)
                # Prefer calls for ATM if contract_type is 'both'
                if strike_distance < min_strike_distance:
                    min_strike_distance = strike_distance
                    best_contract = contract
                elif strike_distance == min_strike_distance:
                    # If same distance, prefer better liquidity
                    if contract.liquidity_score > best_contract.liquidity_score:
                        best_contract = contract
            
            return best_contract
        
        # For directional trades, find contract closest to target delta
        best_contract = None
        min_delta_distance = float('inf')
        
        for contract in contracts:
            delta_distance = abs(contract.delta - target_delta)
            
            if delta_distance < min_delta_distance:
                min_delta_distance = delta_distance
                best_contract = contract
            elif delta_distance == min_delta_distance:
                # If same delta distance, prefer better liquidity
                if contract.liquidity_score > best_contract.liquidity_score:
                    best_contract = contract
        
        return best_contract
    
    def _relax_liquidity_requirements(self, contracts: List[OptionsContract]) -> List[OptionsContract]:
        """
        Relax liquidity requirements if no contracts meet strict criteria
        
        Args:
            contracts: List of all contracts
            
        Returns:
            Contracts meeting relaxed criteria
        """
        # Relax to 50 OI and 15% spread
        relaxed = []
        
        for contract in contracts:
            if (contract.open_interest >= 50 and 
                contract.spread_percent <= 0.15):
                relaxed.append(contract)
        
        if relaxed:
            logger.info(f"Using relaxed liquidity criteria: {len(relaxed)} contracts")
        
        return relaxed
    
    def format_recommendation(self, symbol: str, contracts: List[OptionsContract],
                             signal_type: str, stock_price: float) -> str:
        """
        Format options recommendations for display
        
        Args:
            symbol: Stock symbol
            contracts: List of recommended contracts
            signal_type: Signal type
            stock_price: Current stock price
            
        Returns:
            Formatted string for display
        """
        if not contracts:
            return f"No liquid options available for {symbol}"
        
        lines = []
        lines.append(f"\n📊 Options Recommendations for {symbol} @ ${stock_price:.2f}")
        lines.append(f"Signal: {signal_type}")
        lines.append("-" * 60)
        
        for i, contract in enumerate(contracts, 1):
            lines.append(f"\n{i}. {contract.contract_type.upper()} ${contract.strike:.2f} "
                        f"exp {contract.expiration} ({contract.days_to_expiry}d)")
            lines.append(f"   Delta: {contract.delta:.2f} | "
                        f"Bid/Ask: ${contract.bid:.2f}/${contract.ask:.2f} | "
                        f"Spread: {contract.spread_percent*100:.1f}%")
            lines.append(f"   OI: {contract.open_interest:,} | "
                        f"Volume: {contract.volume:,} | "
                        f"IV: {contract.implied_volatility:.1%} | "
                        f"Liquidity: {contract.liquidity_score:.0f}/100")
        
        return "\n".join(lines)