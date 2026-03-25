"""
VIX Decomposition Research Framework
=====================================
A modular, research-grade pipeline for reconstructing VIX from the SPX
option surface, decomposing daily VIX changes, and building/backtesting
volatility-informed equity strategies.

Module map
----------
data_ingestion      Step 1 – raw data loading & cleaning
iv_surface          Step 2 – continuous IV surface construction
vix_reconstruction  Step 3 – CBOE variance formula → synthetic VIX
vix_decomposition   Step 4 – parallel / skew / convexity decomposition
signals             Step 5 – strategy signal generation
portfolio           Step 6 – daily-rebalanced portfolio simulation
metrics             Step 7 – performance statistics
walk_forward        Step 8 – walk-forward validation
output              Step 9 – charts and summary tables
"""
