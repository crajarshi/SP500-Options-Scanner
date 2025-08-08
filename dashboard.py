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
from signals import get_actionable_header


class OptionsScannnerDashboard:
    """Interactive dashboard for displaying stock analysis results"""
    
    def __init__(self, risk_manager=None):
        self.console = Console()
        self.risk_manager = risk_manager
        self.eastern = pytz.timezone('US/Eastern')
        
    def get_market_status(self, scan_context: Dict = None) -> Tuple[str, str]:
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
    
    def create_header(self, scan_time: datetime, next_scan: datetime, market_regime_bullish: bool = True,
                     scan_type: str = 'sp500', watchlist_file: str = None, scan_context: Dict = None) -> Panel:
        """Create dashboard header with watchlist support"""
        market_status, current_time = self.get_market_status(scan_context)
        status_color = "green" if market_status == "OPEN" else "yellow"
        
        # Calculate time until next scan
        time_until_next = next_scan - datetime.now()
        minutes_until = int(time_until_next.total_seconds() / 60)
        seconds_until = int(time_until_next.total_seconds() % 60)
        
        # Add regime warning if not bullish
        regime_text = ""
        if not market_regime_bullish:
            regime_text = "\n[yellow]⚠️  MARKET REGIME: NOT BULLISH - Exercise Caution[/yellow]"
        
        # Determine title based on scan type
        if scan_type == 'watchlist' and watchlist_file:
            import os
            watchlist_name = os.path.basename(watchlist_file)
            title = f"[bold white]WATCHLIST SCAN - {watchlist_name}[/bold white]"
        else:
            title = "[bold white]S&P 500 Intraday Options Scanner[/bold white]"
        
        # Add data freshness indicator if market is closed
        data_info = ""
        if scan_context and not scan_context.get('is_market_open'):
            if scan_context.get('reference_date'):
                ref_date = scan_context['reference_date']
                data_info = f"\n[cyan]📅 Options Analysis for: {ref_date.strftime('%B %d, %Y')}[/cyan]"
        
        header_text = (
            f"{title}\n\n"
            f"Market Status: [{status_color}]{market_status}[/{status_color}]    "
            f"Time: {current_time}    "
            f"Next Scan: {minutes_until:02d}:{seconds_until:02d}"
            f"{regime_text}"
            f"{data_info}"
        )
        
        return Panel(
            Align.center(header_text),
            box=box.DOUBLE,
            border_style="bright_blue"
        )
    
    def create_compact_scoring_info(self) -> Panel:
        """Create a compact scoring info panel"""
        info_text = (
            "[bold]Scoring:[/bold] MACD 30% + RSI 30% + BB 20% + OBV 10% + VOL 10% | "
            "[bold]Signals:[/bold] >85 STRONG BUY | 70-85 BUY | 50-70 NEUTRAL+ | 30-50 NEUTRAL- | <30 SELL"
        )
        return Panel(
            info_text,
            box=box.MINIMAL,
            border_style="dim",
            padding=(0, 1)
        )
    
    def create_scoring_explanation(self) -> Panel:
        """Create a panel explaining the scoring system"""
        explanation_text = (
            "[bold]Scoring System Breakdown:[/bold]\n\n"
            "📊 [cyan]Composite Score = [/cyan]\n"
            f"   MACD ({config.WEIGHT_MACD*100:.0f}%) + "
            f"RSI ({config.WEIGHT_RSI*100:.0f}%) + "
            f"Bollinger ({config.WEIGHT_BOLLINGER*100:.0f}%) + \n"
            f"   OBV ({config.WEIGHT_OBV*100:.0f}%) + "
            f"ATR ({config.WEIGHT_ATR*100:.0f}%)\n\n"
            "[bold]Individual Indicators:[/bold]\n"
            "• [yellow]RSI[/yellow]: 100pts if oversold (<30), 0pts if overbought (>70)\n"
            "• [yellow]MACD[/yellow]: 100pts if bullish crossover, 0pts otherwise\n"
            "• [yellow]Bollinger[/yellow]: 100pts at lower band, 0pts at upper band\n"
            "• [yellow]OBV[/yellow]: 100pts if above 20-SMA, 0pts if below\n"
            "• [yellow]ATR[/yellow]: 100pts if expanding (>30-SMA), 0pts if contracting\n\n"
            "[bold]Signal Thresholds:[/bold]\n"
            f"• [bright_green]STRONG BUY[/bright_green]: Score > {config.SIGNAL_STRONG_BUY}\n"
            f"• [green]BUY[/green]: Score {config.SIGNAL_BUY}-{config.SIGNAL_STRONG_BUY}\n"
            f"• [white]HOLD[/white]: Score {config.SIGNAL_HOLD_MIN}-{config.SIGNAL_HOLD_MAX}\n"
            f"• [red]AVOID[/red]: Score < {config.SIGNAL_HOLD_MIN}"
        )
        
        return Panel(
            explanation_text,
            title="📈 How Scoring Works",
            box=box.ROUNDED,
            border_style="blue",
            padding=(1, 2)
        )
    
    def create_results_table(self, analyses: List[Dict], limit: int = None, 
                           scan_type: str = 'sp500', watchlist_file: str = None) -> Table:
        """Create results table with watchlist support"""
        if limit is None:
            limit = config.TOP_STOCKS_DISPLAY
        
        # Determine table title
        if scan_type == 'watchlist' and watchlist_file:
            import os
            watchlist_name = os.path.splitext(os.path.basename(watchlist_file))[0]
            title = f"TOP OPPORTUNITIES FROM WATCHLIST ({watchlist_name}.txt)"
        else:
            title = f"TOP {limit} OPTIONS TRADING OPPORTUNITIES (Last {config.RSI_PERIOD * 15 / 60:.1f} hours analysis)"
        
        table = Table(
            title=title,
            box=box.HEAVY_HEAD,
            show_lines=True,
            title_style="bold white"
        )
        
        # Add columns - Enhanced with individual scores
        table.add_column("#", style="cyan", width=3, justify="center")
        table.add_column("Ticker", style="bold white", width=6)
        table.add_column("Price", style="white", width=8, justify="right")
        table.add_column("Chg%", width=6, justify="right")
        table.add_column("Score", style="bold", width=5, justify="center")
        
        # Individual indicator scores (more compact)
        table.add_column("RSI", width=4, justify="center")
        table.add_column("MCD", width=4, justify="center")
        table.add_column("BB", width=4, justify="center")
        table.add_column("OBV", width=4, justify="center")
        table.add_column("VOL", width=4, justify="center")
        
        table.add_column("ATR$", width=6, justify="right")
        table.add_column("Strategy", width=22)
        
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
            
            # Get individual scores
            scores = analysis['scores']
            
            # Format individual indicator scores with colors
            def format_score(score, threshold=50):
                if score >= 80:
                    return Text(f"{score:.0f}", style="bright_green")
                elif score >= threshold:
                    return Text(f"{score:.0f}", style="green")
                elif score > 0:
                    return Text(f"{score:.0f}", style="yellow")
                else:
                    return Text(f"{score:.0f}", style="red")
            
            rsi_score_text = format_score(scores.get('rsi_score', 0))
            macd_score_text = format_score(scores.get('macd_score', 0))
            bb_score_text = format_score(scores.get('bollinger_score', 0))
            obv_score_text = format_score(scores.get('obv_score', 0))
            vol_score_text = format_score(scores.get('volume_score', 0))
            atr_score_text = format_score(scores.get('atr_score', 0))
            
            # ATR value with trend
            atr_value = analysis['indicators'].get('atr_value', 0)
            atr_trend = analysis['indicators'].get('atr_trend', 'Unknown')
            
            if atr_trend == 'Rising':
                vol_text = Text(f"${atr_value:.1f}↑", style="green")
            else:
                vol_text = Text(f"${atr_value:.1f}↓", style="yellow")
            
            # Color signal text based on type
            signal_colors = {
                'STRONG_BUY': 'bright_green',
                'BUY': 'green',
                'NEUTRAL_BULL': 'yellow',
                'NEUTRAL_BEAR': 'orange1',
                'STRONG_SELL': 'red',
                # Legacy
                'HOLD': 'white',
                'AVOID': 'red'
            }
            signal_color = signal_colors.get(analysis['signal']['type'], 'white')
            signal_text = Text(analysis['signal']['text'], style=signal_color)
            
            row = [
                str(rank), 
                ticker, 
                price, 
                change, 
                score_text,
                rsi_score_text,
                macd_score_text,
                bb_score_text,
                obv_score_text,
                vol_score_text,
                vol_text,
                signal_text
            ]
            
            table.add_row(*row)
        
        return table
    
    def create_footer(self) -> Panel:
        """Create dashboard footer with instructions"""
        footer_text = (
            "Use command-line options: --top N for more results | --filter SIGNAL_TYPE | "
            "--export for CSV | --continuous for auto-refresh"
        )
        return Panel(
            Align.center(footer_text),
            box=box.ROUNDED,
            border_style="dim white"
        )
    
    def display_results(self, analyses: List[Dict], 
                       scan_time: datetime,
                       errors: List[Dict] = None,
                       market_regime_bullish: bool = True,
                       mode: str = 'adaptive',
                       scan_type: str = 'sp500',
                       watchlist_file: str = None,
                       scan_context: Dict = None):
        """Display full dashboard with mode and watchlist awareness"""
        self.console.clear()
        
        # Calculate next scan time
        next_scan = scan_time + timedelta(minutes=config.REFRESH_INTERVAL_MINUTES)
        
        # Display header directly with watchlist info
        self.console.print(self.create_header(scan_time, next_scan, market_regime_bullish, 
                                              scan_type, watchlist_file, scan_context))
        
        # Display risk status if risk manager is available
        if self.risk_manager:
            self.display_risk_status()
        
        # Display compact scoring info
        self.console.print(self.create_compact_scoring_info())
        
        # Handle mixed mode display
        if mode == 'mixed':
            # Separate bullish and bearish stocks
            bullish_stocks = [a for a in analyses if a.get('score_type') == 'bullish']
            bearish_stocks = [a for a in analyses if a.get('score_type') == 'bearish']
            
            # Display bullish opportunities
            if bullish_stocks:
                self.console.print("\n🟢 [bold green]TOP BULLISH OPPORTUNITIES (Uptrending Stocks)[/bold green]")
                self.console.print(self.create_results_table(bullish_stocks[:10], scan_type=scan_type, 
                                                            watchlist_file=watchlist_file))
            
            # Display bearish opportunities
            if bearish_stocks:
                self.console.print("\n🔴 [bold red]TOP BEARISH OPPORTUNITIES (Downtrending Stocks)[/bold red]")
                self.console.print(self.create_results_table(bearish_stocks[:10], scan_type=scan_type,
                                                            watchlist_file=watchlist_file))
            
            if not bullish_stocks and not bearish_stocks:
                self.console.print("\n[yellow]No trend-confirmed opportunities found in mixed market[/yellow]")
        else:
            # Single mode display
            self.console.print(self.create_results_table(analyses, scan_type=scan_type,
                                                        watchlist_file=watchlist_file))
        
        # Display options recommendations if available
        self.display_options_recommendations(analyses)
        
        # Display footer
        self.console.print(self.create_footer())
        
        # Show error summary if any
        if errors:
            error_text = f"\n[yellow]⚠ Skipped {len(errors)} stocks due to errors. "
            error_text += "See logs/error_log.txt for details.[/yellow]"
            self.console.print(error_text)
    
    def display_risk_status(self):
        """Display risk management status panel"""
        if not self.risk_manager:
            return
        
        status = self.risk_manager.get_risk_status()
        
        # Determine emoji and style based on status
        if status['status'] == 'BLOCKED':
            emoji = "❌"
            status_style = "bold red"
            panel_style = "red"
        elif status['status'] == 'WARNING':
            emoji = "⚠️"
            status_style = "bold yellow"
            panel_style = "yellow"
        else:
            emoji = "✅"
            status_style = "bold green"
            panel_style = "green"
        
        # Create risk status content
        if status['trades_blocked']:
            content = Text(
                f"{emoji} TRADING BLOCKED - Daily Loss Limit Reached\n"
                f"Current Loss: ${-status['daily_pnl']:.2f} | Limit: ${status['daily_limit']:.2f}\n"
                f"Trading will resume tomorrow",
                style="bold red"
            )
        else:
            content = Text()
            content.append(f"💰 RISK STATUS: {emoji} {status['status']}", style=status_style)
            content.append(f" | Daily P&L: ${status['daily_pnl']:+.2f}")
            content.append(f" | Limit: ${status['daily_limit']:.2f}\n")
            content.append(f"Remaining Capacity: ${status['remaining_capacity']:.2f}")
            content.append(f" ({status['capacity_percent']:.0f}%)")
            content.append(f" | Portfolio: ${status['portfolio_value']:,.0f}")
        
        # Display panel
        self.console.print(Panel(
            content,
            box=box.DOUBLE,
            border_style=panel_style,
            padding=(0, 1)
        ))
    
    def display_options_recommendations(self, analyses: List[Dict]):
        """Display options contract recommendations if available"""
        # Check if any analyses have options data
        stocks_with_options = [a for a in analyses if a.get('options_contracts')]
        
        if not stocks_with_options:
            return
        
        # Create options recommendations panel
        self.console.print("\n")
        self.console.print(Panel(
            "[bold cyan]📊 OPTIONS CONTRACT RECOMMENDATIONS[/bold cyan]",
            box=box.DOUBLE,
            border_style="cyan"
        ))
        
        for analysis in stocks_with_options[:5]:  # Show top 5 only for space
            ticker = analysis['ticker']
            signal = analysis['signal']['type']  # Use signal TYPE not text
            contracts = analysis['options_contracts']
            
            if not contracts:
                continue
            
            # Get actionable header for clear instructions
            actionable_header = get_actionable_header(signal)
            
            # Create a table for this stock's options
            table = Table(
                title=f"[bold cyan]{ticker}[/bold cyan] @ ${analysis['current_price']:.2f} - {actionable_header}",
                box=box.ROUNDED,
                show_lines=True,
                title_style="bold white",
                show_header=True,
                header_style="bold"
            )
            
            # Add columns - INCLUDING SYMBOL
            table.add_column("Symbol", style="bold cyan", width=6)
            table.add_column("Strike", style="white", width=8)
            table.add_column("Exp", style="white", width=10)
            table.add_column("Type", style="white", width=5)
            table.add_column("Delta", style="yellow", width=6)
            table.add_column("Bid/Ask", style="white", width=12)
            table.add_column("Spread", style="white", width=7)
            table.add_column("OI", style="green", width=8)
            table.add_column("Liq", style="magenta", width=5)
            
            # Add rows for each contract
            for contract in contracts[:3]:  # Top 3 contracts
                spread_color = "green" if contract['spread%'] < 5 else "yellow" if contract['spread%'] < 10 else "red"
                liquidity_color = "green" if contract['liquidity'] > 70 else "yellow" if contract['liquidity'] > 50 else "red"
                
                # Check for risk information
                if contract.get('risk_blocked', False):
                    # Contract is blocked by risk manager
                    table.add_row(
                        ticker,
                        f"${contract['strike']:.2f}",
                        contract['expiration'],
                        contract['type'],
                        f"{contract['delta']:.2f}",
                        f"${contract['bid']:.2f}/${contract['ask']:.2f}",
                        Text(f"{contract['spread%']:.1f}%", style=spread_color),
                        f"{contract['OI']:,}",
                        Text("BLOCKED", style="bold red")
                    )
                    # Add risk message in next row
                    self.console.print(f"   [red]❌ {contract.get('risk_message', 'Risk limit exceeded')}[/red]")
                elif not contract.get('risk_valid', True):
                    # Contract exceeds risk but can be adjusted
                    table.add_row(
                        ticker,
                        f"${contract['strike']:.2f}",
                        contract['expiration'],
                        contract['type'],
                        f"{contract['delta']:.2f}",
                        f"${contract['bid']:.2f}/${contract['ask']:.2f}",
                        Text(f"{contract['spread%']:.1f}%", style=spread_color),
                        f"{contract['OI']:,}",
                        Text(f"{contract['liquidity']:.0f}", style=liquidity_color)
                    )
                    # Show risk adjustment message
                    position_size = contract.get('position_size', 0)
                    risk_msg = contract.get('risk_message', '')
                    if position_size > 0:
                        self.console.print(f"   [yellow]⚠️ {risk_msg} | Buy {position_size} contract(s)[/yellow]")
                    else:
                        self.console.print(f"   [red]❌ {risk_msg}[/red]")
                else:
                    # Contract passes risk checks
                    position_size = contract.get('position_size', 1)
                    total_risk = contract.get('total_risk', contract['mid'] * 100)
                    
                    table.add_row(
                        ticker,
                        f"${contract['strike']:.2f}",
                        contract['expiration'],
                        contract['type'],
                        f"{contract['delta']:.2f}",
                        f"${contract['bid']:.2f}/${contract['ask']:.2f}",
                        Text(f"{contract['spread%']:.1f}%", style=spread_color),
                        f"{contract['OI']:,}",
                        Text(f"{contract['liquidity']:.0f}", style=liquidity_color)
                    )
                    # Show approved risk info
                    if self.risk_manager:
                        self.console.print(f"   [green]✅ Buy {position_size} contract(s) | Risk: ${total_risk:.2f}[/green]")
            
            self.console.print(table)
        
        # Add explanation
        self.console.print("\n[dim]💡 Recommendations based on: Delta targeting (0.70/-0.70), "
                          "30-60 day expiration, minimum 100 OI, max 10% spread[/dim]")
    
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