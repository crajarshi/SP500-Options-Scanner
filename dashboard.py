"""
Interactive console dashboard for S&P 500 Options Scanner
"""
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich import box
import pytz

import config


class OptionsScannnerDashboard:
    """Interactive dashboard for displaying stock analysis results"""
    
    def __init__(self):
        self.console = Console()
        self.eastern = pytz.timezone('US/Eastern')
        
    def get_market_status(self) -> Tuple[str, str]:
        """
        Get current market status
        
        Returns:
            Tuple of (status, time_string)
        """
        now_et = datetime.now(self.eastern)
        current_time = now_et.strftime("%I:%M %p ET")
        
        # Check if it's a weekday
        if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return "CLOSED", current_time
        
        # Check market hours
        market_open = now_et.replace(
            hour=config.MARKET_OPEN_HOUR,
            minute=config.MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0
        )
        market_close = now_et.replace(
            hour=config.MARKET_CLOSE_HOUR,
            minute=config.MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0
        )
        
        if market_open <= now_et <= market_close:
            return "OPEN", current_time
        elif now_et < market_open:
            return "PRE-MARKET", current_time
        else:
            return "AFTER-HOURS", current_time
    
    def format_price(self, price: float) -> str:
        """Format price with dollar sign"""
        return f"${price:.2f}"
    
    def format_change(self, change_pct: float) -> Text:
        """Format percentage change with color"""
        if change_pct > 0:
            return Text(f"+{change_pct:.1f}%", style="green")
        elif change_pct < 0:
            return Text(f"{change_pct:.1f}%", style="red")
        else:
            return Text(f"{change_pct:.1f}%", style="white")
    
    def format_indicator_status(self, indicator: str, value: any, positive: bool) -> str:
        """Format indicator with checkmark or X"""
        if indicator == "RSI":
            return f"RSI:{value:.0f} {'✓' if positive else '✗'}"
        elif indicator == "MACD":
            return f"MACD:{'✓' if positive else '✗'}"
        elif indicator == "BB":
            return f"BB:{value:.0f} {'✓' if value > 50 else '✗'}"
        elif indicator == "OBV":
            return f"OBV:{'✓' if positive else '✗'}"
    
    def create_header(self, scan_time: datetime, next_scan: datetime) -> Panel:
        """Create dashboard header"""
        market_status, current_time = self.get_market_status()
        status_color = "green" if market_status == "OPEN" else "yellow"
        
        # Calculate time until next scan
        time_until_next = next_scan - datetime.now()
        minutes_until = int(time_until_next.total_seconds() / 60)
        seconds_until = int(time_until_next.total_seconds() % 60)
        
        header_text = (
            f"[bold white]S&P 500 Intraday Options Scanner[/bold white]\n\n"
            f"Market Status: [{status_color}]{market_status}[/{status_color}]    "
            f"Time: {current_time}    "
            f"Next Scan: {minutes_until:02d}:{seconds_until:02d}"
        )
        
        return Panel(
            Align.center(header_text),
            box=box.DOUBLE,
            border_style="bright_blue"
        )
    
    def create_results_table(self, analyses: List[Dict], limit: int = None) -> Table:
        """Create results table"""
        if limit is None:
            limit = config.TOP_STOCKS_DISPLAY
        
        table = Table(
            title=f"TOP OPTIONS TRADING SIGNALS (Last {config.RSI_PERIOD * 15 / 60:.1f} hours analysis)",
            box=box.HEAVY_HEAD,
            show_lines=True,
            title_style="bold white"
        )
        
        # Add columns
        table.add_column("Rank", style="cyan", width=6, justify="center")
        table.add_column("Ticker", style="bold white", width=8)
        table.add_column("Price", style="white", width=10, justify="right")
        table.add_column("Chg%", width=8, justify="right")
        table.add_column("Score", style="bold", width=7, justify="center")
        table.add_column("Signal", width=25)
        
        if config.SHOW_DETAILED_INDICATORS:
            table.add_column("Indicators", style="dim white", width=35)
        
        # Add rows
        for i, analysis in enumerate(analyses[:limit]):
            rank = i + 1
            ticker = analysis['ticker']
            price = self.format_price(analysis['current_price'])
            change = self.format_change(analysis['price_change_pct'])
            score = analysis['scores']['composite_score']
            
            # Format score with color
            if score > config.SIGNAL_STRONG_BUY:
                score_text = Text(f"{score:.1f}", style="bright_green bold")
            elif score > config.SIGNAL_BUY:
                score_text = Text(f"{score:.1f}", style="green")
            else:
                score_text = Text(f"{score:.1f}", style="white")
            
            # Signal with emoji
            signal = f"{analysis['signal']['emoji']} {analysis['signal']['text']}"
            
            # Color signal text based on type
            signal_colors = {
                'STRONG_BUY': config.COLOR_STRONG_BUY,
                'BUY': config.COLOR_BUY,
                'HOLD': config.COLOR_HOLD,
                'AVOID': config.COLOR_AVOID
            }
            signal_color = signal_colors.get(analysis['signal']['type'], 'white')
            
            row = [str(rank), ticker, price, change, score_text, 
                   Text(signal, style=signal_color)]
            
            # Add indicators if enabled
            if config.SHOW_DETAILED_INDICATORS:
                indicators_text = " ".join([
                    self.format_indicator_status("RSI", 
                        analysis['indicators']['rsi'],
                        analysis['scores']['rsi_score'] > 50),
                    self.format_indicator_status("MACD", 
                        None,
                        analysis['indicators']['macd_bullish']),
                    self.format_indicator_status("BB", 
                        analysis['scores']['bollinger_score'],
                        analysis['scores']['bollinger_score'] > 50),
                    self.format_indicator_status("OBV", 
                        None,
                        analysis['indicators']['obv_above_sma'])
                ])
                row.append(indicators_text)
            
            table.add_row(*row)
        
        return table
    
    def create_footer(self) -> Panel:
        """Create dashboard footer with instructions"""
        footer_text = (
            "[F1] Full List  [F2] Filter  [R] Refresh Now  "
            "[E] Export CSV  [Q] Quit"
        )
        return Panel(
            Align.center(footer_text),
            box=box.ROUNDED,
            border_style="dim white"
        )
    
    def display_results(self, analyses: List[Dict], 
                       scan_time: datetime,
                       errors: List[Dict] = None):
        """Display full dashboard"""
        self.console.clear()
        
        # Calculate next scan time
        next_scan = scan_time + timedelta(minutes=config.REFRESH_INTERVAL_MINUTES)
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(self.create_header(scan_time, next_scan), size=5),
            Layout(self.create_results_table(analyses), size=20),
            Layout(self.create_footer(), size=3)
        )
        
        self.console.print(layout)
        
        # Show error summary if any
        if errors:
            error_text = f"\n[yellow]⚠ Skipped {len(errors)} stocks due to errors. "
            error_text += "See logs/error_log.txt for details.[/yellow]"
            self.console.print(error_text)
    
    def display_progress(self, current: int, total: int, ticker: str = ""):
        """Display progress bar"""
        progress_pct = (current / total) * 100
        bar_length = 40
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        status = f"[Processing {current}/{total}] {bar} {progress_pct:.1f}%"
        if ticker:
            status += f" - {ticker}"
        
        return status
    
    def display_error(self, message: str):
        """Display error message"""
        self.console.print(f"[red]✗ Error:[/red] {message}")
    
    def display_success(self, message: str):
        """Display success message"""
        self.console.print(f"[green]✓[/green] {message}")