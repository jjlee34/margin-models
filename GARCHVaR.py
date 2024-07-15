#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:58:45 2024

@author: jordanlee
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bond_pricing as bp
from arch import arch_model
import yfinance as yf
from fredapi import Fred

fred = Fred(api_key='ab681e773883416226591f77779d3692')

securityType = input('Equity or Treasury: ')

if (securityType == 'Equity'): 
    symbol = input('Symbol: ')
    start_date = '2005-12-01'
    end_date = '2020-12-01'
    data = yf.download(symbol, start=start_date, end=end_date)
    
    cp = data['Adj Close']
    returns = 100*cp.pct_change(1).dropna()
    
elif (securityType == 'Treasury'):
    time = input("Bond Maturity: ")
    coupon = input("Coupon: ")
    fr = input("Frequency: ")
    coupon_rate = float(coupon)/100
    symbol = 'U.S. ' + time
    if 'MO' in symbol:
        bond = symbol[:-2]
        maturity = float(int(bond[5:])/12)
    else:
        maturity = int(symbol[5:])
    frequency = float(fr)

    data = fred.get_series(f'DGS{time}').dropna()
    data = pd.DataFrame(data)
    data = data.tail(617)
    bondPrice = []
    temp = ""

    for y in data[data.columns[0]]:
        if y == ".":
            y = temp
        bondPrice.append(bp.simple_bonds.bond_price(mat=maturity, cpn=coupon_rate, yld=float(y)/100, freq=frequency))
        temp = y
    data["BondPrice"] = pd.Series(bondPrice, index=data.index)
    data = data.dropna()
    returns = 100 * data["BondPrice"].pct_change(1).dropna()

am = arch_model(returns)
res = am.fit(disp='off')
print(res.summary())
quit()

window_size = 252
n_simulations = 10000
confidence_level = 0.99

var_series = []
sample_series = []

for i in range(window_size, len(returns)):
    train_data = returns.iloc[i-window_size:i]
    
    
    model = arch_model(train_data, vol='Garch', p=1, o=0, q=1, dist='normal')
    res = model.fit(disp="off")
    sample_series.append(res.conditional_volatility)
    
    forecasts = res.forecast(horizon=3, start=train_data.index[-1], method='simulation', simulations=n_simulations)
    simulated_returns = forecasts.simulations.values[-1, :, 0]
    VaR = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    var_series.append((train_data.index[-1], VaR))

var_df = pd.DataFrame(var_series, columns=['Date', 'VaR']).set_index('Date')
mean_var = var_df['VaR'].mean()

aligned_returns = returns.loc[var_df.index]
exceedances = aligned_returns[aligned_returns < var_df['VaR']]
expected_exceedances = (1 - confidence_level) * len(returns)

plt.figure(figsize=(12, 6))
plt.plot(var_df, color='blue', label='VaR (99%)')
plt.plot(aligned_returns, color='grey', label='Returns')
plt.scatter(exceedances.index, exceedances, color='red', marker='x', label='Exceedances', zorder=5)
plt.axhline(y=0.0, color='black', linestyle='-')
plt.title(f'GARCH VaR Time Series for {symbol} vs Returns')
plt.xlabel('Date')
plt.ylabel('VaR / Returns (%)')
plt.legend()
plt.show()

print(f"Mean VaR (99%): {str(mean_var*-1)}")
print(f"Mean VaR (99%) in bps: {str(mean_var*-100)}")
print(f"Number of VaR exceedances vs expected exceedances: {len(exceedances)}, {expected_exceedances}")
print(f"Latest VaR (99%): {var_df.iloc[-1, 0]*-1}")
print(f"Latest VaR in bps (99%): {var_df.iloc[-1, 0]*-100}")
