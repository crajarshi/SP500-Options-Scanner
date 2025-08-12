"""
Backtesting Engine for ML Trading Strategies
Simulates realistic trading with transaction costs, slippage, and risk management
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    ticker: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 100
    ml_probability: float = 0.5
    ml_confidence: float = 0.5
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    

class MLBacktester:
    """Backtests ML-based trading strategies with realistic constraints"""
    
    def __init__(self,
                 initial_capital: float = 30000,
                 risk_per_trade: float = 0.015,  # 1.5% risk per trade
                 max_positions: int = 10,
                 commission: float = 1.0,  # $1 per trade
                 slippage_pct: float = 0.001,  # 0.1% slippage
                 stop_loss_pct: float = 0.02,  # 2% stop loss
                 take_profit_pct: float = 0.05,  # 5% take profit
                 confidence_threshold: float = 0.6):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            risk_per_trade: Percentage of capital to risk per trade
            max_positions: Maximum concurrent positions
            commission: Commission per trade in dollars
            slippage_pct: Slippage as percentage of price
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            confidence_threshold: Minimum ML confidence to take trade
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.confidence_threshold = confidence_threshold
        
        # Track positions and history
        self.open_positions = {}
        self.closed_trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        # Performance metrics
        self.metrics = {}
        
    def calculate_position_size(self, price: float, stop_loss: float, 
                              confidence: float = 1.0) -> int:
        """
        Calculate position size based on risk management
        
        Args:
            price: Entry price
            stop_loss: Stop loss price
            confidence: ML model confidence (scales position)
            
        Returns:
            Number of shares to trade
        """
        # Risk amount based on confidence
        risk_amount = self.capital * self.risk_per_trade * confidence
        
        # Risk per share
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        # Position size
        shares = int(risk_amount / risk_per_share)
        
        # Ensure we don't exceed capital
        max_shares = int((self.capital * 0.95) / price)  # Use max 95% of capital
        
        return min(shares, max_shares)
    
    def execute_entry(self, ticker: str, date: datetime, price: float,
                     ml_probability: float, ml_confidence: float) -> Optional[Trade]:
        """
        Execute trade entry with slippage and commission
        
        Args:
            ticker: Stock ticker
            date: Entry date
            price: Entry price
            ml_probability: ML model probability
            ml_confidence: ML model confidence
            
        Returns:
            Trade object or None if trade not executed
        """
        # Check if we can take the trade
        if len(self.open_positions) >= self.max_positions:
            return None
        
        if ml_confidence < self.confidence_threshold:
            return None
        
        # Calculate slippage
        slippage = price * self.slippage_pct
        entry_price = price + slippage  # Buy at higher price due to slippage
        
        # Calculate stop loss and take profit
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)
        
        # Calculate position size
        shares = self.calculate_position_size(entry_price, stop_loss, ml_confidence)
        
        if shares == 0:
            return None
        
        # Calculate total cost
        total_cost = (shares * entry_price) + self.commission
        
        if total_cost > self.capital:
            return None
        
        # Create trade
        trade = Trade(
            ticker=ticker,
            entry_date=date,
            entry_price=entry_price,
            shares=shares,
            ml_probability=ml_probability,
            ml_confidence=ml_confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission=self.commission,
            slippage=slippage
        )
        
        # Update capital
        self.capital -= total_cost
        
        # Add to open positions
        self.open_positions[ticker] = trade
        
        return trade
    
    def execute_exit(self, trade: Trade, date: datetime, price: float, 
                    reason: str = "signal") -> Trade:
        """
        Execute trade exit
        
        Args:
            trade: Trade to exit
            date: Exit date
            price: Exit price
            reason: Reason for exit
            
        Returns:
            Updated trade object
        """
        # Apply slippage
        slippage = price * self.slippage_pct
        exit_price = price - slippage  # Sell at lower price due to slippage
        
        # Update trade
        trade.exit_date = date
        trade.exit_price = exit_price
        trade.exit_reason = reason
        
        # Calculate P&L
        gross_pnl = (exit_price - trade.entry_price) * trade.shares
        trade.pnl = gross_pnl - self.commission  # Subtract exit commission
        trade.pnl_pct = (exit_price / trade.entry_price - 1) * 100
        
        # Update capital
        self.capital += (trade.shares * exit_price) - self.commission
        
        # Remove from open positions
        if trade.ticker in self.open_positions:
            del self.open_positions[trade.ticker]
        
        # Add to closed trades
        self.closed_trades.append(trade)
        
        return trade
    
    def check_stops(self, date: datetime, price_data: Dict[str, float]):
        """
        Check and execute stop losses and take profits
        
        Args:
            date: Current date
            price_data: Dictionary of ticker -> current price
        """
        positions_to_close = []
        
        for ticker, trade in self.open_positions.items():
            if ticker not in price_data:
                continue
                
            current_price = price_data[ticker]
            
            # Check stop loss
            if current_price <= trade.stop_loss:
                positions_to_close.append((trade, trade.stop_loss, "stop_loss"))
            
            # Check take profit
            elif current_price >= trade.take_profit:
                positions_to_close.append((trade, trade.take_profit, "take_profit"))
        
        # Execute exits
        for trade, exit_price, reason in positions_to_close:
            self.execute_exit(trade, date, exit_price, reason)
    
    def run_backtest(self, data: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data with ML predictions
        
        Args:
            data: Historical price data with columns: date, ticker, open, high, low, close, volume
            predictions: ML predictions with columns: date, ticker, probability, confidence
            
        Returns:
            Dictionary of backtest results
        """
        # Merge data and predictions
        df = pd.merge(data, predictions, on=['date', 'ticker'], how='inner')
        df = df.sort_values('date')
        
        # Group by date for day-by-day simulation
        for date, daily_data in df.groupby('date'):
            # Update equity curve
            current_equity = self.capital + sum(
                trade.shares * daily_data[daily_data['ticker'] == trade.ticker]['close'].iloc[0]
                for trade in self.open_positions.values()
                if trade.ticker in daily_data['ticker'].values
            )
            self.equity_curve.append({
                'date': date,
                'equity': current_equity,
                'cash': self.capital,
                'n_positions': len(self.open_positions)
            })
            
            # Check stops for existing positions
            price_data = dict(zip(daily_data['ticker'], daily_data['close']))
            self.check_stops(date, price_data)
            
            # Look for new entry signals
            for _, row in daily_data.iterrows():
                # Only enter on high probability signals
                if row['probability'] > 0.6 and row['ticker'] not in self.open_positions:
                    self.execute_entry(
                        ticker=row['ticker'],
                        date=date,
                        price=row['close'],
                        ml_probability=row['probability'],
                        ml_confidence=row['confidence']
                    )
            
            # Exit signals based on ML predictions
            for ticker, trade in list(self.open_positions.items()):
                ticker_data = daily_data[daily_data['ticker'] == ticker]
                if not ticker_data.empty:
                    current_prob = ticker_data['probability'].iloc[0]
                    # Exit if probability drops below threshold
                    if current_prob < 0.4:
                        self.execute_exit(trade, date, ticker_data['close'].iloc[0], "ml_signal")
        
        # Close any remaining positions at last price
        if len(self.open_positions) > 0:
            last_date = df['date'].max()
            last_prices = df[df['date'] == last_date].set_index('ticker')['close'].to_dict()
            
            for ticker, trade in list(self.open_positions.items()):
                if ticker in last_prices:
                    self.execute_exit(trade, last_date, last_prices[ticker], "end_of_backtest")
        
        # Calculate metrics
        self.calculate_metrics()
        
        return self.get_results()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.closed_trades:
            logger.warning("No closed trades to calculate metrics")
            return
        
        # Basic metrics
        total_trades = len(self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl <= 0]
        
        # P&L metrics
        total_pnl = sum(t.pnl for t in self.closed_trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Win rate
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Average metrics
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        avg_win_pct = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Maximum drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
            max_drawdown = equity_df['drawdown'].min()
            
            # Calculate daily returns
            equity_df['returns'] = equity_df['equity'].pct_change()
            
            # Sharpe ratio (assuming 252 trading days, risk-free rate = 0)
            if len(equity_df) > 1:
                sharpe_ratio = np.sqrt(252) * equity_df['returns'].mean() / equity_df['returns'].std()
            else:
                sharpe_ratio = 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0
        
        # Store metrics
        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': self.capital,
            'total_commission': sum(t.commission * 2 for t in self.closed_trades),  # Entry + exit
            'total_slippage': sum(t.slippage * t.shares for t in self.closed_trades)
        }
    
    def get_results(self) -> Dict:
        """
        Get complete backtest results
        
        Returns:
            Dictionary with results and metrics
        """
        return {
            'metrics': self.metrics,
            'trades': [self._trade_to_dict(t) for t in self.closed_trades],
            'equity_curve': self.equity_curve,
            'config': {
                'initial_capital': self.initial_capital,
                'risk_per_trade': self.risk_per_trade,
                'max_positions': self.max_positions,
                'commission': self.commission,
                'slippage_pct': self.slippage_pct,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'confidence_threshold': self.confidence_threshold
            }
        }
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade object to dictionary"""
        return {
            'ticker': trade.ticker,
            'entry_date': trade.entry_date.isoformat() if trade.entry_date else None,
            'entry_price': trade.entry_price,
            'exit_date': trade.exit_date.isoformat() if trade.exit_date else None,
            'exit_price': trade.exit_price,
            'shares': trade.shares,
            'ml_probability': trade.ml_probability,
            'ml_confidence': trade.ml_confidence,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'exit_reason': trade.exit_reason
        }
    
    def print_summary(self):
        """Print backtest summary"""
        if not self.metrics:
            print("No metrics available. Run backtest first.")
            return
        
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.metrics['final_capital']:,.2f}")
        print(f"Total Return: {self.metrics['total_return_pct']:.2f}%")
        print(f"Total P&L: ${self.metrics['total_pnl']:,.2f}")
        print("\n" + "-"*60)
        print("TRADE STATISTICS")
        print("-"*60)
        print(f"Total Trades: {self.metrics['total_trades']}")
        print(f"Winning Trades: {self.metrics['winning_trades']}")
        print(f"Losing Trades: {self.metrics['losing_trades']}")
        print(f"Win Rate: {self.metrics['win_rate']:.1f}%")
        print(f"Avg Win: ${self.metrics['avg_win']:,.2f} ({self.metrics['avg_win_pct']:.2f}%)")
        print(f"Avg Loss: ${self.metrics['avg_loss']:,.2f} ({self.metrics['avg_loss_pct']:.2f}%)")
        print(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        print("\n" + "-"*60)
        print("RISK METRICS")
        print("-"*60)
        print(f"Max Drawdown: {self.metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print("\n" + "-"*60)
        print("COSTS")
        print("-"*60)
        print(f"Total Commission: ${self.metrics['total_commission']:,.2f}")
        print(f"Total Slippage: ${self.metrics['total_slippage']:,.2f}")
        print("="*60)
    
    def save_results(self, filepath: str):
        """Save backtest results to file"""
        results = self.get_results()
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Backtest results saved to {filepath}")