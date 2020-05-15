# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:34:42 2020

@author: tobja
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
plt.style.use('seaborn-bright')
# np.random.seed(5)

os.chdir(os.path.expanduser('~/Documents/projects/optimization'))

# load in PortOptimizer class
from optimization import PortOptimizer

# list of ETFs to assemble into an optimal portfolio
tickers = ['IWV','EFA','EEM','LQD','TIP','IAU','IYR']

# instantiate a portfolio optimization
po = PortOptimizer()

# gather returns
po.get_returns(tickers, refresh=False)

# calculate vcv matrix and correlation matrix
po.get_covar()

# get weights
po.optimize()

# turn weights into numpy array
wts = po.optimal_weights
wts = np.array(wts[0])

# get mean returns for each asset
mean_rets = np.array(np.mean(po.returns, axis=0) * 250)

# compute portfolio risk/return from the optimal weights
port_ret = np.dot(mean_rets, wts)
port_vol = np.sqrt(np.dot(wts, np.dot(po.covar, wts.T))) * np.sqrt(250)

def simulate_portfolios(po, trials, mean_rets, tickers):
    '''
    Produces two arrays of shape (10000, 1), one for returns, one for stdevs 
    for 10,000 simulated portfolios. 
    '''
    
    # simulated portfolio weights
    sim_wts = np.random.uniform(size=(trials, len(tickers)))
    col_sums = np.repeat(np.sum(sim_wts, axis=1), len(tickers)).reshape(trials, len(tickers))
    sim_wts = np.divide(sim_wts, col_sums)
    
    # simulated portfolio returns
    asset_rets = np.repeat(mean_rets, trials).reshape(len(tickers), trials).T
    sim_rets = np.sum(sim_wts * asset_rets, axis=1)
    
    # simulated portfolio volatilities
    f = lambda x: np.sqrt(np.dot(x, np.dot(po.covar, x))) * np.sqrt(250)
    sim_vols = np.apply_along_axis(func1d=f, axis=1, arr=sim_wts)
    
    return sim_rets, sim_vols

sim_rets, sim_vols = simulate_portfolios(po, 10000, mean_rets, tickers)

def generate_plot(sim_rets, sim_vols, port_ret, port_vol):
    '''
    Produces a .png file of a plot of the results of the simulated portfolios. 
    Adds the optimal portfolio and annotates accordingly.
    '''
    
    # adjust data for percent formatting
    sim_rets *= 100
    sim_vols *= 100
    port_ret *= 100
    port_vol *= 100    
    
    # generate plot
    fig, ax = plt.subplots(1, figsize=(12,9))
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.10, left=0.08, right=0.95)
    
    # plot simulated portfolios and such
    ax.scatter(x=sim_vols, y=sim_rets)
    
    # plot the optimal portfolio with annotation
    ax.scatter(x=[port_vol], y=[port_ret], color='black')
    ax.annotate(
        'Optimal Pf', 
        xy=(port_vol, port_ret), xycoords='data',
        xytext=(port_vol * 0.9, port_ret * 1.1), textcoords='data',
        arrowprops=dict(arrowstyle='-', connectionstyle='arc3')
        )
    
    # labels etc
    fig.suptitle('Optimized Portfolio vs. 10,000 Simulated Portfolios')
    ax.set_ylabel('Portfolio Expected Return')
    ax.set_xlabel('Portfolio Standard Deviation')
    
    # percent formatting
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    
    # save figure
    plt.gcf().savefig('optimal_port.png')
    
generate_plot(sim_rets, sim_vols, port_ret, port_vol)