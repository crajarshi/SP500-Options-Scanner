#!/usr/bin/env python
"""Test options functionality"""
from alpaca_data_provider import AlpacaDataProvider
import logging

logging.basicConfig(level=logging.INFO)

provider = AlpacaDataProvider()
# Test with a stock that should have options
chain = provider.fetch_options_chain('AAPL')
if chain:
    print('✓ Options chain fetched (using simulated data)')
    print(f'  Expirations: {len(chain)}')
    for exp_date in list(chain.keys())[:1]:
        print(f'  Example expiration: {exp_date}')
        print(f'  Strikes available: {len(chain[exp_date])}')
        # Show one contract
        for strike in list(chain[exp_date].keys())[:1]:
            if 'call' in chain[exp_date][strike]:
                call = chain[exp_date][strike]['call']
                print(f'  Call @ ${strike}: Delta={call.get("delta", 0):.2f}, Bid/Ask=${call.get("bid", 0):.2f}/${call.get("ask", 0):.2f}')
else:
    print('✗ No options chain returned')