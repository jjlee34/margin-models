#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:18:15 2024

@author: jordanlee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import bond_pricing as bp
from fredapi import Fred


fred = Fred(api_key='ab681e773883416226591f77779d3692')
securityType = input('Equity or Treasury: ')

if (securityType == 'Equity'): 
    symbol = input('Symbol: ')
    start_date = '2005-12-01'
    end_date = '2020-12-01'
    data = yf.download(symbol, start=start_date, end=end_date)
    
    cp = data['Adj Close']
    returns = 100*cp.pct_change(3).dropna()
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
    returns = 100 * data["BondPrice"].pct_change(3).dropna()
else:
    quit()

def ewma_volatility(returns, lam=0.97):
    vol = np.zeros_like(returns)
    vol[0] = np.std(returns)
    for t in range(1, len(returns)):
        vol[t] = np.sqrt(((1 - lam) * returns[t-1]**2 + lam * vol[t-1]**2))
    return vol

def calculate_fhs_var(returns, vol, confidence_level=0.99):

    standardized_returns = returns / vol
    var_threshold = np.percentile(standardized_returns, (1 - confidence_level)*100)
    var = var_threshold * vol[-1]
    return var

def calculate_hs_var(returns, confidence_level=0.99):

    var_threshold = np.percentile(returns, (1 - confidence_level)*100)
    return var_threshold


lam = 0.97
ewma_vol = ewma_volatility(returns, lam)

window_size = 252
confidence_level = 0.99

fhs_var_series = []
hs_var_series = []
dates = []
pnl = []

for i in range(window_size, len(returns)):
    window_returns = returns.iloc[i-window_size:i]
    window_vol = ewma_vol[i-window_size:i]
    
    fhs_var = calculate_fhs_var(window_returns, window_vol, confidence_level)
    hs_var = calculate_hs_var(window_returns, confidence_level)
    
    fhs_var_series.append(fhs_var)
    hs_var_series.append(hs_var)
    dates.append(returns.index[i])
    pnl.append(returns.iloc[i])

fhs_var_df = pd.DataFrame({'Date': dates, 'FHS VaR': fhs_var_series}).set_index('Date')
hs_var_df = pd.DataFrame({'Date': dates, 'HS VaR': hs_var_series}).set_index('Date')
returns_df = pd.DataFrame({'Date': dates, 'Returns': pnl}).set_index('Date')

mean_fhs_var = fhs_var_df['FHS VaR'].mean()
mean_hs_var = hs_var_df['HS VaR'].mean()

plt.figure(figsize=(12, 6))
plt.plot(fhs_var_df, color='blue', label='FHS VaR (99%)', alpha=0.7)
plt.plot(hs_var_df, color='red', label='HS VaR (99%)', alpha=0.7)
plt.plot(returns_df, color='green', label='Returns', alpha=0.5)
plt.axhline(y=0.0, color='black', linestyle='-')
plt.fill_between(returns_df.index, returns_df['Returns'], where=returns_df['Returns'] < fhs_var_df['FHS VaR'], color='blue', alpha=0.3)
plt.fill_between(returns_df.index, returns_df['Returns'], where=returns_df['Returns'] < hs_var_df['HS VaR'], color='red', alpha=0.3)
plt.title(f'FHS/HS VaR Time Series vs P&L for {symbol}')
plt.xlabel('Date')
plt.ylabel('VaR / Returns (%)')
plt.legend()

expected_exceedances = (1 - confidence_level) * len(returns)

exceedances_fhs = returns_df[returns_df['Returns'] < fhs_var_df['FHS VaR']]
exceedances_hs = returns_df[returns_df['Returns'] < hs_var_df['HS VaR']]
plt.scatter(exceedances_fhs.index, exceedances_fhs['Returns'], color='black', marker='x', label='FHS Exceedances')
plt.scatter(exceedances_hs.index, exceedances_hs['Returns'], color='orange', marker='x', label='HS Exceedances')

plt.legend()
plt.show()

print(f"Latest FHS VaR in bps (99%): {fhs_var_df.iloc[-1, 0]*-100}")
print(f"Latest HS VaR in bps (99%): {hs_var_df.iloc[-1, 0]*-100}")
print(f"Mean FHS VaR (99%): {mean_fhs_var*-100}")
print(f"Mean HS VaR (99%): {mean_hs_var*-100}")
print(f"Number of FHS VaR exceedances vs expected exceedances: {len(exceedances_fhs)}, {expected_exceedances}")
print(f"Number of HS VaR exceedances vs expected exceedances: {len(exceedances_hs)}, {expected_exceedances}")