"""
Investment Thesis SQLAlchemy Model
Provides structured documentation for investment decisions and rationale.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index, DECIMAL
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.models.unified_models import Base


class InvestmentThesis(Base):
    """
    Investment thesis documentation for stocks.
    Captures comprehensive investment rationale, analysis, and decision-making process.
    """
    __tablename__ = "investment_thesis"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False, index=True)

    # Core thesis fields
    investment_objective = Column(Text, nullable=False, comment="Primary investment goal (growth, income, preservation)")
    time_horizon = Column(String(50), nullable=False, comment="Expected holding period (short/medium/long-term)")
    target_price = Column(DECIMAL(10, 2), nullable=True, comment="Target price based on valuation")

    # Business analysis
    business_model = Column(Text, nullable=True, comment="Description of how the company makes money")
    competitive_advantages = Column(Text, nullable=True, comment="Moats and competitive positioning")

    # Financial analysis
    financial_health = Column(Text, nullable=True, comment="Analysis of balance sheet, cash flow, profitability")

    # Growth and risk analysis
    growth_drivers = Column(Text, nullable=True, comment="Key factors driving future growth")
    risks = Column(Text, nullable=True, comment="Risk assessment and mitigation strategies")

    # Valuation
    valuation_rationale = Column(Text, nullable=True, comment="Valuation methodology and price targets")

    # Exit strategy
    exit_strategy = Column(Text, nullable=True, comment="Conditions for selling or rebalancing")

    # Full thesis content (markdown)
    content = Column(Text, nullable=True, comment="Complete thesis in markdown format")

    # Versioning
    version = Column(Integer, default=1, nullable=False, comment="Version number for tracking updates")

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", backref="investment_theses")
    stock = relationship("Stock", backref="investment_theses")

    # Indexes for performance
    __table_args__ = (
        Index('idx_thesis_user_stock', 'user_id', 'stock_id'),
        Index('idx_thesis_updated_at', 'updated_at'),
        Index('idx_thesis_user_updated', 'user_id', 'updated_at'),
    )

    def __repr__(self):
        return f"<InvestmentThesis(id={self.id}, user_id={self.user_id}, stock_id={self.stock_id}, version={self.version})>"
