#!/usr/bin/env python3
"""
Demo Data Seed Script for Investment Analysis Platform

Populates the database with realistic mock data so the frontend has something
to display. Safe to run multiple times - fully idempotent.

Usage:
    cd /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform
    python scripts/seed_demo_data.py

    # Dry-run to see what would be inserted without touching the DB:
    python scripts/seed_demo_data.py --dry-run

    # Reset all demo data and re-seed from scratch:
    python scripts/seed_demo_data.py --reset
"""

import sys
import os
import argparse
import logging
import random
import math
from datetime import datetime, timedelta, timezone, date
from decimal import Decimal
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap - must run before any backend imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so settings are satisfied before importing backend modules
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.config.settings import settings
from backend.models.unified_models import (
    Base,
    Exchange,
    Sector,
    Industry,
    Stock,
    PriceHistory,
    Fundamentals,
    Recommendation,
    Portfolio,
    Position,
    Watchlist,
    WatchlistItem,
    User,
)
from backend.auth.oauth2 import get_password_hash

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_demo_data")

# ---------------------------------------------------------------------------
# Reproducible randomness so results look the same on every run
# ---------------------------------------------------------------------------
random.seed(42)

# ---------------------------------------------------------------------------
# Static demo data definitions
# ---------------------------------------------------------------------------

DEMO_USER = {
    "username": "demo_user",
    "email": "demo@investmentplatform.local",
    "password": "DemoPass123!",
    "full_name": "Demo Investor",
    "risk_tolerance": "moderate",
    "investment_style": "growth",
}

EXCHANGE_DATA = {
    "code": "NASDAQ",
    "name": "NASDAQ Stock Market",
    "timezone": "America/New_York",
    "country": "US",
    "currency": "USD",
    "market_open": "09:30",
    "market_close": "16:00",
}

NYSE_EXCHANGE_DATA = {
    "code": "NYSE",
    "name": "New York Stock Exchange",
    "timezone": "America/New_York",
    "country": "US",
    "currency": "USD",
    "market_open": "09:30",
    "market_close": "16:00",
}

# sector_name -> [industry_name, ...]
SECTOR_INDUSTRY_MAP = {
    "Technology": ["Semiconductors", "Software", "Internet Services", "Hardware"],
    "Financial Services": ["Banking", "Payments & Credit Services", "Insurance"],
    "Healthcare": ["Pharmaceuticals", "Managed Care", "Medical Devices"],
    "Consumer Discretionary": ["E-Commerce", "Specialty Retail", "Entertainment & Media"],
    "Consumer Staples": ["Household Products", "Beverages", "Personal Care"],
    "Industrials": ["Aerospace & Defense", "Industrial Machinery"],
}

# fmt: off
STOCKS = [
    # symbol, name, sector, industry, exchange, price, market_cap_B, pe_ratio, description
    ("AAPL",  "Apple Inc.",                    "Technology",             "Hardware",                    "NASDAQ", 189.30, 2950.0, 30.2, "Designs and sells consumer electronics, software, and services."),
    ("MSFT",  "Microsoft Corporation",         "Technology",             "Software",                    "NASDAQ", 415.20, 3090.0, 35.8, "Cloud computing, productivity software, and gaming."),
    ("GOOGL", "Alphabet Inc.",                 "Technology",             "Internet Services",           "NASDAQ", 175.50, 2200.0, 26.4, "Search, advertising, cloud, and AI services."),
    ("AMZN",  "Amazon.com Inc.",               "Consumer Discretionary", "E-Commerce",                  "NASDAQ", 185.80, 1940.0, 42.1, "E-commerce, cloud computing (AWS), and digital advertising."),
    ("TSLA",  "Tesla Inc.",                    "Consumer Discretionary", "Specialty Retail",            "NASDAQ", 248.50, 792.0,  55.3, "Electric vehicles, energy storage, and solar products."),
    ("NVDA",  "NVIDIA Corporation",            "Technology",             "Semiconductors",              "NASDAQ", 875.40, 2160.0, 65.7, "Graphics processing units, AI chips, and data-center computing."),
    ("META",  "Meta Platforms Inc.",           "Technology",             "Internet Services",           "NASDAQ", 505.10, 1290.0, 27.9, "Social media (Facebook, Instagram, WhatsApp) and VR/AR."),
    ("JPM",   "JPMorgan Chase & Co.",          "Financial Services",     "Banking",                     "NYSE",   202.30, 585.0,  12.3, "Global financial services, investment banking, and asset management."),
    ("V",     "Visa Inc.",                     "Financial Services",     "Payments & Credit Services",  "NYSE",   276.40, 563.0,  30.1, "Global payments technology connecting consumers and businesses."),
    ("JNJ",   "Johnson & Johnson",             "Healthcare",             "Pharmaceuticals",             "NYSE",   147.80, 354.0,  14.8, "Pharmaceuticals, medical devices, and consumer health products."),
    ("UNH",   "UnitedHealth Group Inc.",       "Healthcare",             "Managed Care",                "NYSE",   524.60, 481.0,  22.6, "Diversified health care and well-being company."),
    ("HD",    "The Home Depot Inc.",           "Consumer Discretionary", "Specialty Retail",            "NYSE",   362.10, 356.0,  23.4, "Home improvement retail stores and professional services."),
    ("PG",    "Procter & Gamble Co.",          "Consumer Staples",       "Household Products",          "NYSE",   164.20, 388.0,  25.1, "Consumer goods including cleaning, grooming, and health products."),
    ("KO",    "The Coca-Cola Company",         "Consumer Staples",       "Beverages",                   "NYSE",    61.50, 265.0,  24.3, "Beverages including sodas, juices, teas, and water brands."),
    ("DIS",   "The Walt Disney Company",       "Consumer Discretionary", "Entertainment & Media",       "NYSE",    92.40, 169.0,  35.7, "Entertainment, theme parks, streaming (Disney+), and media networks."),
]
# fmt: on

# Stocks to include in the demo portfolio (symbol, quantity, cost_basis_per_share)
PORTFOLIO_POSITIONS = [
    ("AAPL",  50,   172.50),
    ("MSFT",  20,   385.00),
    ("NVDA",  15,   620.00),
    ("GOOGL", 30,   155.00),
    ("JNJ",   40,   158.00),
    ("V",     25,   245.00),
]

# Stocks to watch (just the symbols)
WATCHLIST_SYMBOLS = ["TSLA", "META", "AMZN", "KO"]

# Recommendation overrides: symbol -> (action, confidence, reasoning snippet)
RECOMMENDATION_OVERRIDES = {
    "AAPL":  ("buy",         0.82, "Strong iPhone 15 cycle and Services segment growth support continued upside."),
    "MSFT":  ("strong_buy",  0.91, "Azure cloud acceleration and Copilot AI monetisation create a durable growth runway."),
    "GOOGL": ("buy",         0.78, "Search moat remains intact; Gemini integration strengthens advertising pricing power."),
    "AMZN":  ("buy",         0.74, "AWS re-acceleration and margin expansion in retail drive near-term upside."),
    "TSLA":  ("hold",        0.60, "Volume growth story intact but margin pressure from price cuts limits near-term upside."),
    "NVDA":  ("strong_buy",  0.93, "Dominant H100/H200 share in AI training; data-centre backlog extends into 2025."),
    "META":  ("buy",         0.81, "Reality Labs losses declining; ad-revenue recovery and Llama ecosystem differentiate."),
    "JPM":   ("hold",        0.65, "Net-interest income likely to peak; credit normalisation creates caution at current multiple."),
    "V":     ("buy",         0.79, "Resilient cross-border volumes and pricing power; digital shift widens the competitive moat."),
    "JNJ":   ("hold",        0.62, "MedTech growth offsets pharma LOE; litigation tail risk limits re-rating potential."),
    "UNH":   ("buy",         0.76, "Optum growth and Medicare Advantage penetration support double-digit EPS growth."),
    "HD":    ("hold",        0.63, "Housing market softness weighing on big-ticket demand; watching rate-cut catalysts."),
    "PG":    ("hold",        0.61, "Pricing power demonstrated; organic volume recovery needed to justify premium valuation."),
    "KO":    ("hold",        0.67, "Defensive characteristics attractive; emerging-market FX headwinds cap upside near-term."),
    "DIS":   ("sell",        0.55, "Streaming losses and linear TV decline; execution risk on parks capacity expansion."),
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _days_ago(n: int) -> datetime:
    return _now_utc() - timedelta(days=n)


def _generate_price_series(
    base_price: float,
    days: int = 30,
    volatility: float = 0.015,
) -> list[dict]:
    """
    Produce a list of daily OHLCV dicts using a geometric Brownian motion walk.
    The walk runs backwards from today so day 0 = today.
    """
    prices = []
    current = base_price
    for i in range(days, -1, -1):
        dt = _days_ago(i)
        # Daily return drawn from N(0, volatility)
        daily_ret = random.gauss(0.0003, volatility)
        current = current * math.exp(daily_ret)

        intra_vol = current * volatility * random.uniform(0.5, 1.5)
        open_p = round(current * random.uniform(0.998, 1.002), 4)
        close_p = round(current, 4)
        high_p = round(max(open_p, close_p) + intra_vol * random.uniform(0.1, 0.6), 4)
        low_p = round(min(open_p, close_p) - intra_vol * random.uniform(0.1, 0.6), 4)
        volume = int(random.uniform(8_000_000, 60_000_000))

        prices.append({
            "date": dt,
            "open": Decimal(str(open_p)),
            "high": Decimal(str(high_p)),
            "low": Decimal(str(low_p)),
            "close": Decimal(str(close_p)),
            "adjusted_close": Decimal(str(close_p)),
            "volume": volume,
            "typical_price": Decimal(str(round((high_p + low_p + close_p) / 3, 4))),
        })
    return prices


def _target_price(current: float, action: str) -> float:
    multipliers = {
        "strong_buy": random.uniform(1.20, 1.35),
        "buy":        random.uniform(1.10, 1.20),
        "hold":       random.uniform(0.98, 1.08),
        "sell":       random.uniform(0.85, 0.95),
        "strong_sell": random.uniform(0.70, 0.85),
    }
    return round(current * multipliers.get(action, 1.05), 2)


def _stop_loss(current: float, action: str) -> float:
    cuts = {
        "strong_buy": 0.88,
        "buy":        0.90,
        "hold":       0.92,
        "sell":       0.97,
        "strong_sell": 1.05,
    }
    return round(current * cuts.get(action, 0.90), 2)


# ---------------------------------------------------------------------------
# Main seeding logic
# ---------------------------------------------------------------------------

class DemoSeeder:
    """Idempotent demo data seeder."""

    def __init__(self, session: Session, dry_run: bool = False):
        self.session = session
        self.dry_run = dry_run
        self._exchange_cache: dict[str, Exchange] = {}
        self._sector_cache: dict[str, Sector] = {}
        self._industry_cache: dict[str, Industry] = {}
        self._stock_cache: dict[str, Stock] = {}

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, model_cls, filters: dict, defaults: dict):
        """Return existing row or create a new one. Returns (obj, created)."""
        obj = self.session.query(model_cls).filter_by(**filters).first()
        if obj is not None:
            return obj, False
        if self.dry_run:
            logger.info("[DRY-RUN] Would create %s %s", model_cls.__name__, filters)
            return None, True
        obj = model_cls(**filters, **defaults)
        self.session.add(obj)
        self.session.flush()  # assign PK without committing
        return obj, True

    # ------------------------------------------------------------------
    # Exchanges
    # ------------------------------------------------------------------

    def seed_exchange(self, data: dict) -> Exchange:
        code = data["code"]
        if code in self._exchange_cache:
            return self._exchange_cache[code]
        filters = {"code": code}
        defaults = {k: v for k, v in data.items() if k != "code"}
        obj, created = self._get_or_create(Exchange, filters, defaults)
        if created:
            logger.info("  + Exchange: %s", code)
        if obj:
            self._exchange_cache[code] = obj
        return obj

    # ------------------------------------------------------------------
    # Sectors & Industries
    # ------------------------------------------------------------------

    def seed_sectors_and_industries(self):
        for sector_name, industries in SECTOR_INDUSTRY_MAP.items():
            sector, created = self._get_or_create(
                Sector, {"name": sector_name}, {"description": f"{sector_name} sector"}
            )
            if created:
                logger.info("  + Sector: %s", sector_name)
            if sector:
                self._sector_cache[sector_name] = sector

            for ind_name in industries:
                if sector is None:
                    continue
                industry, created = self._get_or_create(
                    Industry,
                    {"name": ind_name},
                    {"sector_id": sector.id, "description": f"{ind_name} industry"},
                )
                if created:
                    logger.info("    + Industry: %s", ind_name)
                if industry:
                    self._industry_cache[ind_name] = industry

    # ------------------------------------------------------------------
    # Stocks
    # ------------------------------------------------------------------

    def seed_stocks(self, exchanges: dict[str, Exchange]):
        for row in STOCKS:
            symbol, name, sector_name, industry_name, exch_code, price, mktcap_b, pe, desc = row

            exch = exchanges.get(exch_code)
            sector = self._sector_cache.get(sector_name)
            industry = self._industry_cache.get(industry_name)

            filters = {"symbol": symbol}
            defaults = {
                "name": name,
                "exchange_id": exch.id if exch else None,
                "sector_id": sector.id if sector else None,
                "industry_id": industry.id if industry else None,
                "asset_type": "stock",
                "market_cap": mktcap_b * 1e9,
                "country": "US",
                "currency": "USD",
                "description": desc,
                "is_active": True,
                "is_tradable": True,
                "is_delisted": False,
                "last_updated": _now_utc(),
                "last_price_update": _now_utc(),
            }
            stock, created = self._get_or_create(Stock, filters, defaults)
            if created:
                logger.info("  + Stock: %s - %s", symbol, name)
            if stock:
                self._stock_cache[symbol] = stock

    # ------------------------------------------------------------------
    # Price History
    # ------------------------------------------------------------------

    def seed_price_history(self, days: int = 30):
        for row in STOCKS:
            symbol = row[0]
            base_price = row[5]
            stock = self._stock_cache.get(symbol)
            if stock is None:
                continue

            # Check how many existing rows we have for this stock
            existing_count = (
                self.session.query(PriceHistory)
                .filter_by(stock_id=stock.id)
                .count()
            )
            if existing_count >= days:
                logger.info("  ~ Price history already present for %s (%d rows)", symbol, existing_count)
                continue

            if self.dry_run:
                logger.info("[DRY-RUN] Would insert %d price rows for %s", days + 1, symbol)
                continue

            series = _generate_price_series(base_price, days=days)
            inserted = 0
            for entry in series:
                exists = (
                    self.session.query(PriceHistory)
                    .filter_by(stock_id=stock.id, date=entry["date"])
                    .first()
                )
                if exists:
                    continue
                ph = PriceHistory(stock_id=stock.id, **entry)
                self.session.add(ph)
                inserted += 1
            logger.info("  + Price history: %s (%d new rows)", symbol, inserted)

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def seed_fundamentals(self):
        for row in STOCKS:
            symbol, name, _, _, _, price, mktcap_b, pe, _ = row
            stock = self._stock_cache.get(symbol)
            if stock is None:
                continue

            period = date(2024, 12, 31)
            exists = (
                self.session.query(Fundamentals)
                .filter_by(stock_id=stock.id, period_date=period, period_type="annual")
                .first()
            )
            if exists:
                logger.info("  ~ Fundamentals already present for %s", symbol)
                continue

            if self.dry_run:
                logger.info("[DRY-RUN] Would insert fundamentals for %s", symbol)
                continue

            shares = int((mktcap_b * 1e9) / price)
            net_income = round(mktcap_b * 1e9 / pe / 1e6, 2) if pe else 0  # in millions
            revenue = round(net_income * random.uniform(4, 8), 2)
            eps = round(price / pe, 2) if pe else 0

            fund = Fundamentals(
                stock_id=stock.id,
                period_date=period,
                period_type="annual",
                filing_date=date(2025, 2, 1),
                revenue=Decimal(str(revenue * 1e6)),
                gross_profit=Decimal(str(revenue * 1e6 * random.uniform(0.35, 0.65))),
                net_income=Decimal(str(net_income * 1e6)),
                eps=Decimal(str(eps)),
                diluted_eps=Decimal(str(round(eps * 0.97, 2))),
                total_assets=Decimal(str(revenue * 1e6 * random.uniform(1.0, 2.5))),
                total_liabilities=Decimal(str(revenue * 1e6 * random.uniform(0.3, 0.9))),
                total_equity=Decimal(str(revenue * 1e6 * random.uniform(0.4, 1.2))),
                cash=Decimal(str(revenue * 1e6 * random.uniform(0.05, 0.25))),
                operating_cash_flow=Decimal(str(net_income * 1e6 * random.uniform(1.1, 1.4))),
                free_cash_flow=Decimal(str(net_income * 1e6 * random.uniform(0.8, 1.1))),
                pe_ratio=pe,
                pb_ratio=round(random.uniform(2.5, 12.0), 2),
                ps_ratio=round(mktcap_b * 1e9 / (revenue * 1e6), 2),
                roe=round(random.uniform(0.10, 0.45), 4),
                roa=round(random.uniform(0.04, 0.18), 4),
                gross_margin=round(random.uniform(0.35, 0.72), 4),
                operating_margin=round(random.uniform(0.12, 0.38), 4),
                net_margin=round(random.uniform(0.08, 0.28), 4),
                debt_to_equity=round(random.uniform(0.2, 1.8), 4),
            )
            self.session.add(fund)
            logger.info("  + Fundamentals: %s", symbol)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def seed_recommendations(self):
        for row in STOCKS:
            symbol = row[0]
            price = row[5]
            stock = self._stock_cache.get(symbol)
            if stock is None:
                continue

            action, confidence, reasoning = RECOMMENDATION_OVERRIDES.get(
                symbol,
                ("hold", 0.60, "Neutral outlook based on current valuation and market conditions.")
            )

            exists = (
                self.session.query(Recommendation)
                .filter_by(stock_id=stock.id, is_active=True)
                .first()
            )
            if exists:
                logger.info("  ~ Recommendation already present for %s", symbol)
                continue

            if self.dry_run:
                logger.info("[DRY-RUN] Would insert recommendation for %s (%s)", symbol, action)
                continue

            target = _target_price(price, action)
            sl = _stop_loss(price, action)
            exp_return = round((target - price) / price, 4)

            risk_map = {
                "strong_buy": "medium",
                "buy": "medium",
                "hold": "low",
                "sell": "high",
                "strong_sell": "high",
            }
            risk_score_map = {
                "strong_buy": round(random.uniform(0.35, 0.55), 2),
                "buy": round(random.uniform(0.30, 0.50), 2),
                "hold": round(random.uniform(0.20, 0.40), 2),
                "sell": round(random.uniform(0.55, 0.75), 2),
                "strong_sell": round(random.uniform(0.65, 0.85), 2),
            }

            rec = Recommendation(
                stock_id=stock.id,
                action=action,
                confidence=confidence,
                priority=min(10, max(1, int(confidence * 10))),
                entry_price=Decimal(str(round(price, 4))),
                target_price=Decimal(str(target)),
                stop_loss=Decimal(str(sl)),
                expected_return=exp_return,
                time_horizon_days=random.choice([30, 60, 90, 180]),
                risk_score=risk_score_map[action],
                risk_level=risk_map[action],
                technical_score=round(random.uniform(0.45, 0.90), 4),
                fundamental_score=round(random.uniform(0.50, 0.92), 4),
                sentiment_score=round(random.uniform(0.40, 0.88), 4),
                overall_score=round(confidence * random.uniform(0.95, 1.05), 4),
                reasoning=reasoning,
                key_factors=[
                    "Revenue growth trajectory",
                    "Margin expansion potential",
                    "Competitive positioning",
                ],
                risks=["Macro slowdown", "Valuation multiple compression"],
                opportunities=["Market share gains", "New product cycle"],
                is_active=True,
                created_at=_days_ago(random.randint(1, 7)),
                valid_until=_now_utc() + timedelta(days=30),
            )
            self.session.add(rec)
            logger.info("  + Recommendation: %s -> %s (%.0f%%)", symbol, action.upper(), confidence * 100)

    # ------------------------------------------------------------------
    # Demo user
    # ------------------------------------------------------------------

    def seed_demo_user(self) -> User | None:
        existing = (
            self.session.query(User)
            .filter_by(username=DEMO_USER["username"])
            .first()
        )
        if existing:
            logger.info("  ~ Demo user already exists: %s", DEMO_USER["username"])
            return existing

        if self.dry_run:
            logger.info("[DRY-RUN] Would create demo user %s", DEMO_USER["username"])
            return None

        user = User(
            username=DEMO_USER["username"],
            email=DEMO_USER["email"],
            hashed_password=get_password_hash(DEMO_USER["password"]),
            full_name=DEMO_USER["full_name"],
            is_active=True,
            is_verified=True,
            is_admin=False,
            is_premium=True,
            risk_tolerance=DEMO_USER["risk_tolerance"],
            investment_style=DEMO_USER["investment_style"],
            role="premium_user",
            preferences={},
            created_at=_days_ago(60),
        )
        self.session.add(user)
        self.session.flush()
        logger.info("  + Demo user created: %s  (password: %s)", DEMO_USER["username"], DEMO_USER["password"])
        return user

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def seed_portfolio(self, user: User) -> Portfolio | None:
        if user is None:
            return None

        existing = (
            self.session.query(Portfolio)
            .filter_by(user_id=user.id, name="Demo Growth Portfolio")
            .first()
        )
        if existing:
            logger.info("  ~ Portfolio already exists for demo user")
            return existing

        if self.dry_run:
            logger.info("[DRY-RUN] Would create demo portfolio")
            return None

        # Calculate totals from position data
        total_cost = sum(qty * cost for _, qty, cost in PORTFOLIO_POSITIONS)
        total_market = sum(
            qty * next(r[5] for r in STOCKS if r[0] == sym)
            for sym, qty, _ in PORTFOLIO_POSITIONS
        )
        total_return = total_market - total_cost
        total_return_pct = (total_return / total_cost) if total_cost else 0.0

        portfolio = Portfolio(
            user_id=user.id,
            name="Demo Growth Portfolio",
            description="A diversified growth portfolio seeded with demo data.",
            is_public=False,
            is_default=True,
            benchmark="SPY",
            total_value=Decimal(str(round(total_market, 2))),
            cash_balance=Decimal("5000.00"),
            invested_value=Decimal(str(round(total_market, 2))),
            total_cost_basis=Decimal(str(round(total_cost, 2))),
            total_return=Decimal(str(round(total_return, 2))),
            total_return_pct=round(total_return_pct, 6),
            daily_return=round(random.uniform(-0.01, 0.02), 6),
            monthly_return=round(random.uniform(0.01, 0.06), 6),
            yearly_return=round(random.uniform(0.08, 0.22), 6),
            volatility=round(random.uniform(0.12, 0.20), 6),
            sharpe_ratio=round(random.uniform(0.8, 1.8), 4),
            created_at=_days_ago(45),
        )
        self.session.add(portfolio)
        self.session.flush()
        logger.info("  + Portfolio: 'Demo Growth Portfolio' (user=%s)", user.username)
        return portfolio

    # ------------------------------------------------------------------
    # Portfolio Positions
    # ------------------------------------------------------------------

    def seed_positions(self, portfolio: Portfolio):
        if portfolio is None:
            return

        for symbol, quantity, cost_basis in PORTFOLIO_POSITIONS:
            stock = self._stock_cache.get(symbol)
            if stock is None:
                logger.warning("  ! Stock %s not found, skipping position", symbol)
                continue

            exists = (
                self.session.query(Position)
                .filter_by(portfolio_id=portfolio.id, stock_id=stock.id)
                .first()
            )
            if exists:
                logger.info("  ~ Position already exists: %s", symbol)
                continue

            if self.dry_run:
                logger.info("[DRY-RUN] Would create position: %s x%d @ $%.2f", symbol, quantity, cost_basis)
                continue

            current_price = next(r[5] for r in STOCKS if r[0] == symbol)
            total_cost = quantity * cost_basis
            market_value = quantity * current_price
            unrealized_gl = market_value - total_cost
            unrealized_gl_pct = unrealized_gl / total_cost if total_cost else 0.0

            position = Position(
                portfolio_id=portfolio.id,
                stock_id=stock.id,
                quantity=Decimal(str(quantity)),
                avg_cost_basis=Decimal(str(round(cost_basis, 4))),
                total_cost_basis=Decimal(str(round(total_cost, 2))),
                current_price=Decimal(str(round(current_price, 4))),
                market_value=Decimal(str(round(market_value, 2))),
                unrealized_gain_loss=Decimal(str(round(unrealized_gl, 2))),
                unrealized_gain_loss_pct=round(unrealized_gl_pct, 6),
                realized_gain_loss=Decimal("0.00"),
                first_purchase_date=_days_ago(random.randint(30, 180)),
                last_transaction_date=_days_ago(random.randint(1, 30)),
            )
            self.session.add(position)
            logger.info(
                "  + Position: %s x%d @ $%.2f (now $%.2f, %+.1f%%)",
                symbol, quantity, cost_basis, current_price,
                unrealized_gl_pct * 100,
            )

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------

    def seed_watchlist(self, user: User):
        if user is None:
            return

        watchlist, created = self._get_or_create(
            Watchlist,
            {"user_id": user.id, "name": "Tech Watchlist"},
            {
                "description": "Stocks on my radar for potential future purchases.",
                "is_public": False,
                "created_at": _days_ago(30),
            },
        )
        if created:
            logger.info("  + Watchlist: 'Tech Watchlist'")

        if watchlist is None:
            return

        for symbol in WATCHLIST_SYMBOLS:
            stock = self._stock_cache.get(symbol)
            if stock is None:
                continue

            exists = (
                self.session.query(WatchlistItem)
                .filter_by(watchlist_id=watchlist.id, stock_id=stock.id)
                .first()
            )
            if exists:
                logger.info("  ~ Watchlist item already present: %s", symbol)
                continue

            if self.dry_run:
                logger.info("[DRY-RUN] Would add %s to watchlist", symbol)
                continue

            current_price = next(r[5] for r in STOCKS if r[0] == symbol)
            item = WatchlistItem(
                watchlist_id=watchlist.id,
                stock_id=stock.id,
                target_price=Decimal(str(round(current_price * random.uniform(0.95, 1.15), 2))),
                notes=f"Monitoring {symbol} for entry opportunity.",
                alert_enabled=True,
                added_at=_days_ago(random.randint(1, 20)),
            )
            self.session.add(item)
            logger.info("  + Watchlist item: %s", symbol)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self):
        logger.info("=" * 60)
        logger.info("Investment Analysis Platform - Demo Data Seeder")
        if self.dry_run:
            logger.info("MODE: DRY RUN (no changes will be saved)")
        logger.info("=" * 60)

        # 1. Exchanges
        logger.info("\n[1/8] Seeding exchanges...")
        nasdaq = self.seed_exchange(EXCHANGE_DATA)
        nyse = self.seed_exchange(NYSE_EXCHANGE_DATA)
        exchanges = {"NASDAQ": nasdaq, "NYSE": nyse}

        # 2. Sectors & Industries
        logger.info("\n[2/8] Seeding sectors and industries...")
        self.seed_sectors_and_industries()

        # 3. Stocks
        logger.info("\n[3/8] Seeding stocks...")
        self.seed_stocks(exchanges)

        # 4. Price history
        logger.info("\n[4/8] Seeding price history (30 days)...")
        self.seed_price_history(days=30)

        # 5. Fundamentals
        logger.info("\n[5/8] Seeding fundamentals...")
        self.seed_fundamentals()

        # 6. Recommendations
        logger.info("\n[6/8] Seeding recommendations...")
        self.seed_recommendations()

        # 7. Demo user & portfolio
        logger.info("\n[7/8] Seeding demo user, portfolio, and positions...")
        user = self.seed_demo_user()
        portfolio = self.seed_portfolio(user)
        self.seed_positions(portfolio)

        # 8. Watchlist
        logger.info("\n[8/8] Seeding watchlist...")
        self.seed_watchlist(user)

        # Commit everything
        if not self.dry_run:
            self.session.commit()
            logger.info("\n[OK] All demo data committed successfully.")
        else:
            self.session.rollback()
            logger.info("\n[OK] Dry-run complete. No changes were saved.")

        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Reset helper
# ---------------------------------------------------------------------------

def reset_demo_data(session: Session):
    """
    Remove all demo-specific rows so the seeder can start fresh.
    Deletes in reverse dependency order to satisfy FK constraints.
    """
    logger.info("Resetting demo data...")

    # Find the demo user
    user = session.query(User).filter_by(username=DEMO_USER["username"]).first()
    if user:
        # Cascade deletes will remove portfolios -> positions, watchlists -> items
        session.delete(user)
        session.flush()
        logger.info("  - Demo user and associated data deleted")

    # Remove recommendations for all demo stocks
    symbols = [r[0] for r in STOCKS]
    stocks = session.query(Stock).filter(Stock.symbol.in_(symbols)).all()
    stock_ids = [s.id for s in stocks]

    if stock_ids:
        deleted_recs = (
            session.query(Recommendation)
            .filter(Recommendation.stock_id.in_(stock_ids))
            .delete(synchronize_session="fetch")
        )
        logger.info("  - %d recommendations deleted", deleted_recs)

        deleted_ph = (
            session.query(PriceHistory)
            .filter(PriceHistory.stock_id.in_(stock_ids))
            .delete(synchronize_session="fetch")
        )
        logger.info("  - %d price history rows deleted", deleted_ph)

        deleted_fund = (
            session.query(Fundamentals)
            .filter(Fundamentals.stock_id.in_(stock_ids))
            .delete(synchronize_session="fetch")
        )
        logger.info("  - %d fundamentals rows deleted", deleted_fund)

    # Remove stocks, industries, sectors, exchanges
    for stock in stocks:
        session.delete(stock)
    session.flush()

    for ind_name in {ind for inds in SECTOR_INDUSTRY_MAP.values() for ind in inds}:
        ind = session.query(Industry).filter_by(name=ind_name).first()
        if ind:
            session.delete(ind)
    session.flush()

    for sector_name in SECTOR_INDUSTRY_MAP:
        s = session.query(Sector).filter_by(name=sector_name).first()
        if s:
            session.delete(s)
    session.flush()

    for exch_code in ["NASDAQ", "NYSE"]:
        e = session.query(Exchange).filter_by(code=exch_code).first()
        if e:
            session.delete(e)
    session.flush()

    session.commit()
    logger.info("Reset complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Seed the Investment Analysis Platform database with demo data."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be inserted without writing to the database.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing demo data first, then re-seed from scratch.",
    )
    args = parser.parse_args()

    # Use the sync engine (same pattern as create_admin_user.py and db_init.py)
    engine = create_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )

    # Ensure all tables exist before we try to insert
    logger.info("Ensuring all database tables exist...")
    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        if args.reset:
            reset_demo_data(session)

        seeder = DemoSeeder(session=session, dry_run=args.dry_run)
        seeder.run()

    engine.dispose()
    logger.info("Done.")


if __name__ == "__main__":
    main()
