#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   portfolios.py
@Time    :   2022/07/09 15:31:50
@Author  :   Jack Tobin
@Version :   1.0
"""

from optimization import PortOptimizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
plt.style.use('seaborn-bright')

# load in PortOptimizer class

# list of ETFs to assemble into an optimal portfolio
tickers = ['IWV', 'EFA', 'EEM', 'LQD', 'TIP', 'IAU', 'IYR']

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
    Produces two arrays of shape ({trials}, 1), one for returns, one for stdevs
    for {trials} simulated portfolios.
    '''

    # simulated portfolio weights
    alphas = np.tile(1, len(tickers))
    sim_wts = np.random.dirichlet(alpha=alphas, size=trials)

    # simulated portfolio returns
    asset_rets = np.repeat(mean_rets, trials).reshape(len(tickers), trials).T
    sim_rets = np.sum(sim_wts * asset_rets, axis=1)

    # simulated portfolio volatilities
    sim_vols = np.apply_along_axis(
        func1d=lambda x: np.sqrt(np.dot(x.T, np.dot(po.covar, x))) * np.sqrt(250),
        axis=1, arr=sim_wts)

    return sim_rets, sim_vols


# run simulation
sim_rets, sim_vols = simulate_portfolios(po, 20000, mean_rets, tickers)
sim_sharpe = sim_rets / sim_vols


def generate_plot(sim_rets, sim_vols, sim_sharpe, port_ret, port_vol):
    '''
    Produces a .png file of a plot of the results of the simulated portfolios.
    Adds the optimal portfolio and annotates accordingly.
    '''

    # adjust data for percent formatting
    sim_rets *= 100
    sim_vols *= 100
    port_ret *= 100
    port_vol *= 100

    # number of trials
    trials = len(sim_rets)

    # generate plot
    fig, ax = plt.subplots(1, figsize=(12, 9))
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.10, left=0.08, right=0.95)

    # plot simulated portfolios and such
    cm = sns.color_palette("viridis", as_cmap=True)
    ax.scatter(x=sim_vols, y=sim_rets, c=sim_sharpe, cmap=cm)

    # plot the optimal portfolio with annotation
    ax.scatter(x=[port_vol], y=[port_ret], color='black')
    ax.annotate(
        'Optimal Pf',
        xy=(port_vol, port_ret), xycoords='data',
        xytext=(port_vol * 0.9, port_ret * 1.1), textcoords='data',
        arrowprops=dict(arrowstyle='-', connectionstyle='arc3')
    )

    # labels etc
    fig.suptitle(
        'Optimized Portfolio vs. {x:,.0f} Simulated Portfolios'.format(x=trials))
    ax.set_ylabel('Portfolio Expected Return')
    ax.set_xlabel('Portfolio Standard Deviation')

    # percent formatting
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    # save figure
    plt.gcf().savefig('optimal_port.png')


generate_plot(sim_rets, sim_vols, sim_sharpe, port_ret, port_vol)
