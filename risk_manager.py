"""
Risk Management Module for Options Trading
Implements daily stop-loss limits and portfolio-based position sizing
"""
import json
import os
from datetime import datetime, date
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages trading risk with:
    - Daily stop-loss limits
    - Portfolio-based position sizing
    - Trade validation
    - P&L tracking
    """
    
    def __init__(self, portfolio_value: float, risk_per_trade: float, 
                 daily_loss_limit: float, data_file: str = 'risk_data/daily_pnl.json'):
        """
        Initialize risk manager
        
        Args:
            portfolio_value: Total portfolio value in USD
            risk_per_trade: Percentage of portfolio to risk per trade (e.g., 0.015 for 1.5%)
            daily_loss_limit: Maximum daily loss in USD
            data_file: Path to persist daily P&L data
        """
        self.portfolio_value = portfolio_value
        self.risk_per_trade = risk_per_trade
        self.daily_loss_limit = daily_loss_limit
        self.max_dollar_risk = portfolio_value * risk_per_trade
        self.data_file = data_file
        
        # Ensure risk_data directory exists
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        
        # Initialize or load daily P&L
        self.daily_pnl = self._load_daily_pnl()
        self.trades_blocked = False
        self.block_reason = ""
        
        # Check if trading should be blocked
        self._check_trading_status()
        
        logger.info(f"Risk Manager initialized: Portfolio=${portfolio_value:,.0f}, "
                   f"Max Risk/Trade=${self.max_dollar_risk:.0f}, "
                   f"Daily Limit=${daily_loss_limit:,.0f}")
    
    def _load_daily_pnl(self) -> float:
        """
        Load daily P&L from file or reset if new day
        
        Returns:
            Current day's P&L
        """
        today = date.today().isoformat()
        
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                # Check if it's a new day
                if data.get('date') == today:
                    pnl = data.get('pnl', 0.0)
                    logger.info(f"Loaded existing daily P&L: ${pnl:.2f}")
                    return pnl
                else:
                    # New day - reset P&L
                    logger.info("New trading day - resetting P&L to 0")
                    return 0.0
            except Exception as e:
                logger.error(f"Error loading P&L data: {e}")
                return 0.0
        else:
            logger.info("No existing P&L data - starting fresh")
            return 0.0
    
    def _save_daily_pnl(self):
        """Save current P&L to file"""
        today = date.today().isoformat()
        data = {
            'date': today,
            'pnl': self.daily_pnl,
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value
        }
        
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved P&L data: ${self.daily_pnl:.2f}")
        except Exception as e:
            logger.error(f"Error saving P&L data: {e}")
    
    def _check_trading_status(self):
        """Check if trading should be blocked based on daily P&L"""
        if self.daily_pnl <= -self.daily_loss_limit:
            self.trades_blocked = True
            self.block_reason = f"Daily loss limit reached: ${-self.daily_pnl:.2f} >= ${self.daily_loss_limit:.2f}"
            logger.warning(f"TRADING BLOCKED: {self.block_reason}")
        else:
            self.trades_blocked = False
            self.block_reason = ""
    
    def can_place_trade(self) -> Tuple[bool, str]:
        """
        Check if new trades are allowed
        
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        self._check_trading_status()
        
        if self.trades_blocked:
            return False, self.block_reason
        
        # Check if we're close to the limit (warning zone)
        remaining = self.daily_loss_limit + self.daily_pnl  # daily_pnl is negative for losses
        if remaining < self.max_dollar_risk:
            warning = f"⚠️ WARNING: Close to daily limit. Remaining: ${remaining:.2f}"
            logger.warning(warning)
            return True, warning
        
        return True, "Trading allowed"
    
    def validate_contract_risk(self, contract_price: float, contracts: int = 1) -> Tuple[bool, str, int]:
        """
        Validate if a contract meets risk requirements
        
        Args:
            contract_price: Price per contract (premium * 100)
            contracts: Number of contracts (default 1)
            
        Returns:
            Tuple of (valid: bool, message: str, recommended_contracts: int)
        """
        total_risk = contract_price * contracts
        
        # Check against max risk per trade
        if total_risk > self.max_dollar_risk:
            # Calculate how many contracts we CAN buy
            max_contracts = int(self.max_dollar_risk / contract_price) if contract_price > 0 else 0
            
            if max_contracts == 0:
                return False, f"Premium ${contract_price:.2f} exceeds max risk ${self.max_dollar_risk:.2f}", 0
            else:
                return False, f"Reduce to {max_contracts} contract(s) to stay within risk limit", max_contracts
        
        # Check against remaining daily capacity
        remaining_capacity = self.daily_loss_limit + self.daily_pnl
        if total_risk > remaining_capacity:
            return False, f"Insufficient daily capacity. Remaining: ${remaining_capacity:.2f}", 0
        
        return True, f"Risk approved: ${total_risk:.2f} / ${self.max_dollar_risk:.2f}", contracts
    
    def calculate_position_size(self, contract_price: float) -> int:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            contract_price: Price per contract (premium * 100)
            
        Returns:
            Number of contracts to trade
        """
        if contract_price <= 0:
            return 0
        
        # Calculate based on max risk per trade
        max_contracts = int(self.max_dollar_risk / contract_price)
        
        # Also check against remaining daily capacity
        remaining_capacity = self.daily_loss_limit + self.daily_pnl
        max_by_capacity = int(remaining_capacity / contract_price) if remaining_capacity > 0 else 0
        
        # Return the minimum of the two
        return min(max_contracts, max_by_capacity)
    
    def update_pnl(self, profit_loss: float, trade_details: Dict = None):
        """
        Update daily P&L after a trade closes
        
        Args:
            profit_loss: Profit (positive) or loss (negative) from the trade
            trade_details: Optional details about the trade for logging
        """
        self.daily_pnl += profit_loss
        self._save_daily_pnl()
        
        # Log trade
        log_msg = f"Trade closed: P&L ${profit_loss:+.2f} | Daily Total: ${self.daily_pnl:+.2f}"
        if trade_details:
            log_msg += f" | Details: {trade_details}"
        logger.info(log_msg)
        
        # Check if we should block further trading
        self._check_trading_status()
    
    def get_risk_status(self) -> Dict:
        """
        Get current risk management status
        
        Returns:
            Dictionary with risk metrics
        """
        remaining_capacity = self.daily_loss_limit + self.daily_pnl
        capacity_percent = (remaining_capacity / self.daily_loss_limit) * 100 if self.daily_loss_limit > 0 else 0
        
        # Determine status color/level
        if self.trades_blocked:
            status = "BLOCKED"
            status_color = "red"
        elif capacity_percent < 20:
            status = "WARNING"
            status_color = "yellow"
        else:
            status = "ACTIVE"
            status_color = "green"
        
        return {
            'status': status,
            'status_color': status_color,
            'daily_pnl': self.daily_pnl,
            'daily_limit': self.daily_loss_limit,
            'remaining_capacity': remaining_capacity,
            'capacity_percent': capacity_percent,
            'portfolio_value': self.portfolio_value,
            'max_risk_per_trade': self.max_dollar_risk,
            'trades_blocked': self.trades_blocked,
            'block_reason': self.block_reason
        }
    
    def reset_daily_pnl(self):
        """Reset daily P&L (use with caution - mainly for testing)"""
        self.daily_pnl = 0.0
        self.trades_blocked = False
        self.block_reason = ""
        self._save_daily_pnl()
        logger.info("Daily P&L reset to 0")
    
    def get_risk_summary(self) -> str:
        """
        Get formatted risk summary for display
        
        Returns:
            Formatted string with risk information
        """
        status = self.get_risk_status()
        
        if status['trades_blocked']:
            return f"❌ TRADING BLOCKED - Daily Loss Limit Reached\n" \
                   f"Current Loss: ${-status['daily_pnl']:.2f} | Limit: ${status['daily_limit']:.2f}\n" \
                   f"Trading will resume tomorrow"
        
        emoji = "✅" if status['status'] == "ACTIVE" else "⚠️" if status['status'] == "WARNING" else "❌"
        
        return f"{emoji} Risk Status: {status['status']} | " \
               f"Daily P&L: ${status['daily_pnl']:+.2f} | " \
               f"Remaining: ${status['remaining_capacity']:.2f} ({status['capacity_percent']:.0f}%)"