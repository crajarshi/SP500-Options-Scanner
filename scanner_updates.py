"""
Updates for SP500OptionsScanner to support adaptive mode
These methods should be added to the SP500OptionsScanner class
"""

def __init__(self, demo_mode=False, use_alpaca=True, quick_mode=False, scanner_mode='adaptive'):
    """Enhanced init with scanner mode"""
    # ... existing init code ...
    self.scanner_mode = scanner_mode
    self.market_regime = None
    self.effective_mode = None

def determine_effective_mode(self) -> str:
    """Determine the effective scanning mode based on market conditions"""
    if self.scanner_mode != 'adaptive':
        return self.scanner_mode
    
    # Use market regime to determine mode
    if not self.market_regime:
        return 'bullish'  # Safe default
    
    breadth = self.market_regime.get('breadth_pct', 50)
    vix = self.market_regime.get('vix_level', 20)
    
    if breadth < config.BEARISH_BREADTH_THRESHOLD or vix > config.BEARISH_VIX_THRESHOLD:
        logger.info("📉 Market regime is BEARISH - scanning for bearish setups")
        return 'bearish'
    elif breadth > config.BULLISH_BREADTH_THRESHOLD and vix < config.BULLISH_VIX_THRESHOLD:
        logger.info("📈 Market regime is BULLISH - scanning for bullish setups")
        return 'bullish'
    else:
        logger.info("🔄 Market regime is MIXED - showing trend-confirmed setups")
        return 'mixed'

def process_stock_adaptive(self, ticker: str) -> Optional[Dict]:
    """Process a single stock with mode-aware scoring"""
    try:
        # Fetch intraday data
        df = self.fetch_intraday_data(ticker)
        if df is None or len(df) < config.MIN_REQUIRED_BARS:
            self.errors.append({
                'ticker': ticker,
                'error': 'Insufficient data',
                'timestamp': datetime.now()
            })
            return None
        
        # Calculate indicators
        indicators = calculate_all_indicators(df)
        if indicators is None:
            self.errors.append({
                'ticker': ticker,
                'error': 'Failed to calculate indicators',
                'timestamp': datetime.now()
            })
            return None
        
        # Analyze with mode awareness
        analysis = analyze_stock(
            ticker, 
            indicators, 
            mode=self.effective_mode,
            market_regime=self.market_regime
        )
        return analysis
        
    except Exception as e:
        self.errors.append({
            'ticker': ticker,
            'error': str(e),
            'timestamp': datetime.now()
        })
        logger.error(f"Error processing {ticker}: {e}")
        return None

def run_scan_adaptive(self, skip_breadth=False, top_count=None, filter_signals=None, auto_export=False) -> List[Dict]:
    """Enhanced scan with adaptive mode support"""
    scan_time = datetime.now()
    self.errors = []
    
    # Check market regime first
    if not self.demo_mode:
        self.market_regime_bullish = self.check_market_regime(skip_breadth=skip_breadth)
    else:
        self.market_regime = {'breadth_pct': 50, 'vix_level': 20, 'spy_above_ma': True, 'is_bullish': True}
        self.market_regime_bullish = True
    
    # Determine effective mode
    self.effective_mode = self.determine_effective_mode()
    
    # Display mode information
    mode_display = {
        'bullish': '[green]📈 BULLISH MODE[/green] - Scanning for call/put-sell opportunities',
        'bearish': '[red]📉 BEARISH MODE[/red] - Scanning for put/call-sell opportunities',
        'mixed': '[yellow]🔄 MIXED MODE[/yellow] - Showing trend-confirmed opportunities'
    }
    self.dashboard.console.print(f"\n{mode_display.get(self.effective_mode, 'ADAPTIVE MODE')}\n")
    
    # Get S&P 500 tickers
    self.dashboard.display_success("Fetching S&P 500 constituents...")
    tickers = self.fetch_sp500_tickers()
    
    # Process stocks with progress bar
    analyses = []
    self.dashboard.console.print(f"\nProcessing {len(tickers)} stocks...")
    
    with tqdm(total=len(tickers), desc="Scanning", ncols=100) as pbar:
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            
            for ticker in batch:
                pbar.set_description(f"Processing {ticker}")
                # Use adaptive processing
                analysis = self.process_stock_adaptive(ticker)
                if analysis:
                    analyses.append(analysis)
                pbar.update(1)
    
    # Rank stocks
    ranked_analyses = rank_stocks(analyses)
    
    # Save results
    self.save_results(ranked_analyses, scan_time)
    self.save_error_log()
    
    # Filter and prepare display
    display_analyses = ranked_analyses
    if filter_signals:
        display_analyses = [a for a in ranked_analyses if a['signal']['type'] in filter_signals]
    
    # In mixed mode, separate bullish and bearish
    if self.effective_mode == 'mixed':
        bullish_analyses = [a for a in display_analyses if a.get('score_type') == 'bullish']
        bearish_analyses = [a for a in display_analyses if a.get('score_type') == 'bearish']
        
        # Apply top count to each category
        if top_count:
            display_analyses = bullish_analyses[:top_count//2] + bearish_analyses[:top_count//2]
        else:
            display_analyses = bullish_analyses[:10] + bearish_analyses[:10]
    else:
        # Single mode - apply top count normally
        if top_count:
            display_analyses = display_analyses[:top_count]
    
    # Auto-export if requested
    if auto_export:
        self.export_to_csv(display_analyses, scan_time)
    
    # Display results with mode awareness
    self.dashboard.display_results(
        display_analyses, 
        scan_time, 
        self.errors,
        market_regime_bullish=getattr(self, 'market_regime_bullish', True),
        mode=self.effective_mode
    )
    
    return ranked_analyses