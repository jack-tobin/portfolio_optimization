# portfolio_optimization
Portfolio Optimization in Python

In this project I designed a simple portfolio optimization model. The user provides the model with a list of publicly traded securities, such as stocks or ETFs. The model then constructs a Markowitz-efficient portfolio of the constituent assets by identifying the portfolio weights that produce the highest risk-adjusted expected return as defined by the Sharpe Ratio.

More on Markowitz Portfolios:
https://en.wikipedia.org/wiki/Markowitz_model

More on the Shape ratio:
https://en.wikipedia.org/wiki/Sharpe_ratio

This project contains two modules, "optimization.py" and "portfolios.py." The "optimization" module contains the class architecture for the optimizer, and the "portfolios" module makes use of the Optimizer class by comparing the "optimal" portfolio with 10,000 randomly-weighted simulated portfolios. The results are that the "optimal" portfolio produces the highest risk-adjusted return of any combination of the constituent assets.

![Alt text](/optimal_port.png?raw=true "Optimal Portfolio vs. 10,000 Randomly-Weighted Simulated Portfolios")

Next in line for new features: allow the user to specify the objective function, i.e. either maximize Sharpe, minimize portfolio variance, or maximize some portfolio "utility function".
