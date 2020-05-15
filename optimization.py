# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:37:58 2020

@author: tobja
"""

import pandas as pd
import yfinance as yf
import numpy as np
import pickle
from scipy.optimize import minimize
from statsmodels.stats.moment_helpers import cov2corr

class PortOptimizer:
    '''
    A portfolio optimization object. Allows the user to specify any number of
    publicly traded securities and creates a Markowitz-efficient portfolio 
    from those constituent assets. Objective function is to maximize the 
    portfolio's risk-adjusted return, as defined by the Sharpe ratio.
    '''
    
    def __init__(self):
        self.objective = None
        self.assets = None
        self.returns = None
        self.covar = None
        self.corrs = None
        self.optimal_weights = None
        
    def get_returns(self, assets, refresh=True):
        '''
        Gathers daily price data for each asset in the 'assets' list. Converts
        those prices into daily returns. 'refresh' determines whether the
        data will be pulled fresh from Yahoo! or whether it will be read from 
        an existing .pickle file.
        
        returns: 
            self.assets, the list of assets specified, in the order of 
                the Yahoo! query results
            self.returns, the dataframe of daily returns using adjusted prices
                for each asset in the list of assets.
        '''
        
        # convert list to joined string for yfinance
        assets_str = ' '.join(assets)
        
        # load in data - either preexisting or new
        if refresh:
            # yfinance query - daily data for as long as possible
            prices = yf.download(
                assets_str,
                period='max',
                interval='1d',
                auto_adjust=True,
                progress=False
                )
            
            # process query results
            prices = prices['Close']
            prices.dropna(axis=0, inplace=True)
            
            # convert to returns
            self.returns = prices.pct_change()
            self.returns.dropna(axis=0, inplace=True)
            
            # write to pickle file for later
            with open('returns.pickle', 'wb') as f:
                pickle.dump(self.returns, f)
        else:
            # read from pickle file
            with open('returns.pickle', 'rb') as f:
                self.returns = pickle.load(f)
                
        # save assets for later
        self.assets = list(self.returns.columns)
        
        return self
    
    def get_covar(self):
        '''
        Computes the sample variance-covariance and correlation matrices
        for the returns of the portfolio assets. 
        
        returns: 
            self.covar, a dataframe of pairwise covariance coefficients
                between each of the portfolio assets.
            self.corrs, a dataframe of pairwise correlation coefficients
                between each of the portfolio assets.
        '''
        
        # compute covariances
        cov = np.cov(self.returns, rowvar=False)
        
        # assign to self as neat dataframe
        self.covar = pd.DataFrame(cov, columns=self.assets, index=self.assets)
        
        # compute correlation matrix from covar matrix
        corrs = cov2corr(self.covar)
        self.corrs = pd.DataFrame(corrs, columns=self.assets, index=self.assets)
        
        return self
    
    def optimize(self):
        '''
        Identifies the set of weights that minimizes the negative of the Sharpe
        ratio (-S) (that maximizes the sharpe ratio) of the portfolio.
        
            weights = (min -S) s.t. (Sigma(weights) == 100%, weights >= 0.0%)
        
        returns: self.optimal_weights, a dataframe listing the optimal weight
            for each asset in the portfolio.
        '''
        
        def objective(weights, args):           
            '''
            This is the objective function that the optimization will seek
            to minimize. Intakes the weights, returns and covariance matrix
            and computes the Sharpe ratio of the portfolio, where Sharpe ratio
            is defined as per the below:
                
                Sharpe = (R_p - R_f) / S_p 
            
            where R_p represents portfolio expected return, R_f represents the
            risk-free rate, and S_p represents the portfolio's sample standard
            deviaton.
            
            returns: sharpe_negative, the negative of the sharpe ratio (we return
                negative because the optimization is technicaly a minimization
                function).
            '''
            
            # unpack args
            returns, cov = args
        
            # calculate mean returns as simple average of daily returns
            # for each asset
            mean_rets = np.mean(returns, axis=0) * 250
            
            # get portfolio exp return - w-avg of asset returns and asset weights
            port_ret = np.dot(mean_rets, weights)
            
            # get portfolio volatility - using stdev of multi-asset pf formula
            port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights.T))) * np.sqrt(250)
            
            # calculate sharpe ratio - assumes rf of 1.5%
            rf = 0.015
            sharpe = (port_ret - rf) / port_vol
            
            # set as negative since the optimization is a minimization problem
            sharpe_negative = sharpe * -1
        
            return sharpe_negative
            
        # starting weights - equal
        w0 = np.tile(1/len(self.assets), len(self.assets))
        
        # set constraints - long only, fully invested
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq', 'fun': lambda x: x}]
        
        # minimzation - obtain the minimum inverse of sharpe ratio subject to 
        # long only and fully invested constraints.
        optimal_weights = minimize(
            fun=objective,
            x0=w0,
            args=[self.returns, self.covar],
            constraints=constraints,
            options={'disp': False}
            )
        
        # assign to self
        self.optimal_weights = pd.DataFrame(optimal_weights.x, index=self.assets)
        
        return self