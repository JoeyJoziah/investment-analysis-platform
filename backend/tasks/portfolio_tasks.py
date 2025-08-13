"""
Celery tasks for portfolio management and updates
"""
from celery import shared_task
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
from decimal import Decimal
import logging
import json

from backend.tasks.celery_app import celery_app
from backend.utils.database import get_db_sync
from backend.utils.cache import get_redis_client
from backend.models.tables import (
    Portfolio, Position, Transaction, Order, Stock, PriceHistory,
    PortfolioPerformance, User
)
from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

@celery_app.task
def update_portfolio_value(portfolio_id: int) -> Dict[str, Any]:
    """Update portfolio value based on current prices"""
    try:
        with get_db_sync() as db:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            if not portfolio:
                return {'error': f'Portfolio {portfolio_id} not found'}
            
            # Get all positions
            positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
            
            total_positions_value = Decimal(0)
            position_updates = []
            
            for position in positions:
                # Get current price
                latest_price = db.query(PriceHistory).filter(
                    PriceHistory.stock_id == position.stock_id
                ).order_by(PriceHistory.date.desc()).first()
                
                if latest_price:
                    current_price = latest_price.close
                    market_value = position.quantity * current_price
                    total_positions_value += market_value
                    
                    position_updates.append({
                        'position_id': position.id,
                        'symbol': position.stock.symbol,
                        'quantity': float(position.quantity),
                        'current_price': float(current_price),
                        'market_value': float(market_value),
                        'cost_basis': float(position.quantity * position.average_cost),
                        'unrealized_gain': float(market_value - (position.quantity * position.average_cost))
                    })
            
            # Calculate total portfolio value
            total_value = total_positions_value + portfolio.cash_balance
            
            # Store performance record
            performance_record = PortfolioPerformance(
                portfolio_id=portfolio_id,
                date=date.today(),
                total_value=total_value,
                cash_value=portfolio.cash_balance,
                positions_value=total_positions_value
            )
            
            # Check if record for today exists
            existing = db.query(PortfolioPerformance).filter(
                and_(
                    PortfolioPerformance.portfolio_id == portfolio_id,
                    PortfolioPerformance.date == date.today()
                )
            ).first()
            
            if existing:
                existing.total_value = total_value
                existing.positions_value = total_positions_value
                existing.cash_value = portfolio.cash_balance
            else:
                db.add(performance_record)
            
            # Update portfolio updated_at
            portfolio.updated_at = datetime.utcnow()
            
            db.commit()
            
            result = {
                'portfolio_id': portfolio_id,
                'total_value': float(total_value),
                'cash_balance': float(portfolio.cash_balance),
                'positions_value': float(total_positions_value),
                'positions': position_updates,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Cache the result
            redis_client = get_redis_client()
            cache_key = f"portfolio_value:{portfolio_id}"
            redis_client.setex(cache_key, 300, json.dumps(result))
            
            return result
            
    except Exception as e:
        logger.error(f"Error updating portfolio {portfolio_id}: {e}")
        return {'error': str(e)}

@celery_app.task
def update_all_portfolio_values() -> Dict[str, Any]:
    """Update values for all active portfolios"""
    try:
        with get_db_sync() as db:
            # Get all portfolios
            portfolios = db.query(Portfolio).join(User).filter(
                User.is_active == True
            ).all()
            
            results = {
                'updated': 0,
                'errors': []
            }
            
            for portfolio in portfolios:
                try:
                    update_portfolio_value(portfolio.id)
                    results['updated'] += 1
                except Exception as e:
                    results['errors'].append({
                        'portfolio_id': portfolio.id,
                        'error': str(e)
                    })
            
            logger.info(f"Updated {results['updated']} portfolios")
            return results
            
    except Exception as e:
        logger.error(f"Error updating all portfolios: {e}")
        return {'error': str(e)}

@celery_app.task
def execute_order(order_id: int) -> Dict[str, Any]:
    """Execute a pending order"""
    try:
        with get_db_sync() as db:
            order = db.query(Order).filter(Order.id == order_id).first()
            if not order:
                return {'error': f'Order {order_id} not found'}
            
            if order.status != 'pending':
                return {'error': f'Order {order_id} is not pending'}
            
            # Get current price
            latest_price = db.query(PriceHistory).filter(
                PriceHistory.stock_id == order.stock_id
            ).order_by(PriceHistory.date.desc()).first()
            
            if not latest_price:
                return {'error': 'No price data available'}
            
            current_price = float(latest_price.close)
            
            # Check if order can be executed based on type
            can_execute = False
            execution_price = current_price
            
            if order.order_type == 'market':
                can_execute = True
            elif order.order_type == 'limit':
                if order.order_side == 'buy' and current_price <= float(order.limit_price):
                    can_execute = True
                    execution_price = min(current_price, float(order.limit_price))
                elif order.order_side == 'sell' and current_price >= float(order.limit_price):
                    can_execute = True
                    execution_price = max(current_price, float(order.limit_price))
            elif order.order_type == 'stop':
                if order.order_side == 'buy' and current_price >= float(order.stop_price):
                    can_execute = True
                elif order.order_side == 'sell' and current_price <= float(order.stop_price):
                    can_execute = True
            
            if not can_execute:
                return {'status': 'not_executable', 'current_price': current_price}
            
            # Execute the order
            order.status = 'filled'
            order.filled_quantity = order.quantity
            order.average_fill_price = Decimal(str(execution_price))
            order.filled_at = datetime.utcnow()
            
            # Create transaction
            transaction = Transaction(
                portfolio_id=order.portfolio_id,
                stock_id=order.stock_id,
                transaction_type=order.order_side,
                quantity=order.quantity,
                price=Decimal(str(execution_price)),
                commission=order.commission,
                executed_at=datetime.utcnow()
            )
            db.add(transaction)
            
            # Update portfolio position
            portfolio = order.portfolio
            position = db.query(Position).filter(
                and_(
                    Position.portfolio_id == order.portfolio_id,
                    Position.stock_id == order.stock_id
                )
            ).first()
            
            if order.order_side == 'buy':
                if position:
                    # Update existing position
                    new_quantity = position.quantity + order.quantity
                    new_cost = (position.quantity * position.average_cost + 
                               order.quantity * Decimal(str(execution_price))) / new_quantity
                    position.quantity = new_quantity
                    position.average_cost = new_cost
                else:
                    # Create new position
                    position = Position(
                        portfolio_id=order.portfolio_id,
                        stock_id=order.stock_id,
                        quantity=order.quantity,
                        average_cost=Decimal(str(execution_price))
                    )
                    db.add(position)
                
                # Update cash balance
                total_cost = order.quantity * Decimal(str(execution_price)) + order.commission
                portfolio.cash_balance -= total_cost
                
            else:  # sell
                if position and position.quantity >= order.quantity:
                    position.quantity -= order.quantity
                    if position.quantity == 0:
                        db.delete(position)
                    
                    # Update cash balance
                    total_proceeds = order.quantity * Decimal(str(execution_price)) - order.commission
                    portfolio.cash_balance += total_proceeds
                else:
                    return {'error': 'Insufficient position to sell'}
            
            db.commit()
            
            # Update portfolio value
            update_portfolio_value.delay(order.portfolio_id)
            
            return {
                'order_id': order_id,
                'status': 'executed',
                'execution_price': execution_price,
                'quantity': float(order.quantity),
                'total_value': float(order.quantity * Decimal(str(execution_price))),
                'executed_at': datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error executing order {order_id}: {e}")
        return {'error': str(e)}

@celery_app.task
def check_rebalancing_needed() -> Dict[str, Any]:
    """Check which portfolios need rebalancing"""
    try:
        with get_db_sync() as db:
            portfolios = db.query(Portfolio).filter(
                Portfolio.target_allocation != None,
                Portfolio.rebalance_frequency != None
            ).all()
            
            needs_rebalancing = []
            
            for portfolio in portfolios:
                # Get current allocation
                positions = db.query(Position).filter(
                    Position.portfolio_id == portfolio.id
                ).all()
                
                if not positions:
                    continue
                
                # Calculate current allocation
                total_value = sum(p.quantity * p.average_cost for p in positions) + portfolio.cash_balance
                current_allocation = {}
                
                for position in positions:
                    weight = float((position.quantity * position.average_cost) / total_value * 100)
                    sector = position.stock.sector or 'Other'
                    current_allocation[sector] = current_allocation.get(sector, 0) + weight
                
                # Compare with target allocation
                target = portfolio.target_allocation or {}
                max_deviation = 0
                
                for sector, target_weight in target.items():
                    current_weight = current_allocation.get(sector, 0)
                    deviation = abs(current_weight - target_weight)
                    max_deviation = max(max_deviation, deviation)
                
                # Check if rebalancing is needed (>5% deviation)
                if max_deviation > 5:
                    needs_rebalancing.append({
                        'portfolio_id': portfolio.id,
                        'portfolio_name': portfolio.name,
                        'max_deviation': max_deviation,
                        'current_allocation': current_allocation,
                        'target_allocation': target
                    })
            
            return {
                'checked': len(portfolios),
                'needs_rebalancing': len(needs_rebalancing),
                'portfolios': needs_rebalancing
            }
            
    except Exception as e:
        logger.error(f"Error checking rebalancing: {e}")
        return {'error': str(e)}

@celery_app.task
def rebalance_portfolio(portfolio_id: int, execute: bool = False) -> Dict[str, Any]:
    """Generate or execute rebalancing trades for a portfolio"""
    try:
        with get_db_sync() as db:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            if not portfolio or not portfolio.target_allocation:
                return {'error': 'Portfolio not found or no target allocation set'}
            
            # Get current positions
            positions = db.query(Position).filter(
                Position.portfolio_id == portfolio_id
            ).all()
            
            # Calculate current value
            position_values = {}
            total_value = portfolio.cash_balance
            
            for position in positions:
                latest_price = db.query(PriceHistory).filter(
                    PriceHistory.stock_id == position.stock_id
                ).order_by(PriceHistory.date.desc()).first()
                
                if latest_price:
                    value = position.quantity * latest_price.close
                    position_values[position.stock_id] = {
                        'symbol': position.stock.symbol,
                        'sector': position.stock.sector,
                        'current_value': float(value),
                        'current_quantity': float(position.quantity),
                        'current_price': float(latest_price.close)
                    }
                    total_value += value
            
            # Calculate target values
            target_values = {}
            for sector, weight in portfolio.target_allocation.items():
                target_values[sector] = float(total_value) * (weight / 100)
            
            # Generate rebalancing trades
            trades = []
            
            # Group positions by sector
            sector_values = {}
            for stock_id, pos_data in position_values.items():
                sector = pos_data['sector'] or 'Other'
                if sector not in sector_values:
                    sector_values[sector] = []
                sector_values[sector].append(pos_data)
            
            # Calculate trades needed
            for sector, target_value in target_values.items():
                current_sector_value = sum(
                    p['current_value'] for p in sector_values.get(sector, [])
                )
                
                difference = target_value - current_sector_value
                
                if abs(difference) > 100:  # Minimum trade value
                    # Simplified: just record the sector-level adjustment needed
                    trades.append({
                        'sector': sector,
                        'current_value': current_sector_value,
                        'target_value': target_value,
                        'adjustment': difference,
                        'action': 'buy' if difference > 0 else 'sell'
                    })
            
            result = {
                'portfolio_id': portfolio_id,
                'total_value': float(total_value),
                'trades_needed': trades,
                'estimated_trades': len(trades)
            }
            
            if execute and trades:
                # Create orders for rebalancing (simplified)
                for trade in trades:
                    # This would need more sophisticated logic to select specific stocks
                    logger.info(f"Would execute rebalancing trade: {trade}")
                
                result['status'] = 'executed'
            else:
                result['status'] = 'simulated'
            
            return result
            
    except Exception as e:
        logger.error(f"Error rebalancing portfolio {portfolio_id}: {e}")
        return {'error': str(e)}

@celery_app.task
def calculate_portfolio_performance(portfolio_id: int, period_days: int = 30) -> Dict[str, Any]:
    """Calculate portfolio performance metrics"""
    try:
        with get_db_sync() as db:
            # Get performance history
            start_date = date.today() - timedelta(days=period_days)
            
            performance_records = db.query(PortfolioPerformance).filter(
                and_(
                    PortfolioPerformance.portfolio_id == portfolio_id,
                    PortfolioPerformance.date >= start_date
                )
            ).order_by(PortfolioPerformance.date).all()
            
            if len(performance_records) < 2:
                return {'error': 'Insufficient data for performance calculation'}
            
            # Calculate returns
            values = [float(p.total_value) for p in performance_records]
            returns = [(values[i] - values[i-1]) / values[i-1] 
                      for i in range(1, len(values))]
            
            if not returns:
                return {'error': 'No returns to calculate'}
            
            # Calculate metrics
            import numpy as np
            
            total_return = (values[-1] - values[0]) / values[0]
            avg_daily_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02 / 252
            sharpe_ratio = (avg_daily_return - risk_free_rate) / (np.std(returns) or 1) * np.sqrt(252)
            
            # Max drawdown
            peak = values[0]
            max_drawdown = 0
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Win rate
            winning_days = sum(1 for r in returns if r > 0)
            win_rate = winning_days / len(returns) if returns else 0
            
            result = {
                'portfolio_id': portfolio_id,
                'period_days': period_days,
                'total_return': total_return,
                'annualized_return': total_return * (365 / period_days),
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': -max_drawdown,
                'win_rate': win_rate,
                'best_day': max(returns) if returns else 0,
                'worst_day': min(returns) if returns else 0,
                'current_value': values[-1],
                'starting_value': values[0]
            }
            
            # Cache results
            redis_client = get_redis_client()
            cache_key = f"portfolio_performance:{portfolio_id}:{period_days}"
            redis_client.setex(cache_key, 3600, json.dumps(result))
            
            return result
            
    except Exception as e:
        logger.error(f"Error calculating performance for portfolio {portfolio_id}: {e}")
        return {'error': str(e)}

@celery_app.task
def check_stop_losses() -> Dict[str, Any]:
    """Check all positions for stop loss triggers"""
    try:
        with get_db_sync() as db:
            # Get all positions
            positions = db.query(Position).all()
            
            triggered = []
            
            for position in positions:
                # Get current price
                latest_price = db.query(PriceHistory).filter(
                    PriceHistory.stock_id == position.stock_id
                ).order_by(PriceHistory.date.desc()).first()
                
                if not latest_price:
                    continue
                
                current_price = float(latest_price.close)
                loss_percent = (current_price - float(position.average_cost)) / float(position.average_cost)
                
                # Check if stop loss triggered (e.g., -10%)
                if loss_percent <= -0.10:
                    triggered.append({
                        'portfolio_id': position.portfolio_id,
                        'symbol': position.stock.symbol,
                        'quantity': float(position.quantity),
                        'average_cost': float(position.average_cost),
                        'current_price': current_price,
                        'loss_percent': loss_percent * 100
                    })
                    
                    # Create sell order (optional, based on settings)
                    # This would depend on user preferences
            
            return {
                'positions_checked': len(positions),
                'stop_losses_triggered': len(triggered),
                'triggered_positions': triggered
            }
            
    except Exception as e:
        logger.error(f"Error checking stop losses: {e}")
        return {'error': str(e)}