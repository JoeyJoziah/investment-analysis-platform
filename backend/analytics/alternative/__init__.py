"""
Alternative Data Analytics Package

This package provides comprehensive alternative data analysis including:
- Social media sentiment analysis (Twitter, Reddit, Discord)
- Insider trading analysis from SEC filings
- Earnings whispers and estimate analysis
- Options flow analysis from free CBOE data
- Macro economic indicators
- Supply chain intelligence

All data sources are free or have generous free tiers to maintain budget constraints.
"""

__version__ = "1.0.0"
__all__ = [
    "SocialSentimentAnalyzer",
    "InsiderTradingAnalyzer", 
    "EarningsWhisperAnalyzer",
    "OptionsFlowAnalyzer",
    "MacroEconomicAnalyzer",
    "SupplyChainAnalyzer",
    "AlternativeDataOrchestrator"
]