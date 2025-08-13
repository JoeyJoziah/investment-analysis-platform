"""
Celery tasks for notifications and alerts
"""
from celery import shared_task
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from jinja2 import Template

from backend.tasks.celery_app import celery_app
from backend.utils.database import get_db_sync
from backend.utils.cache import get_redis_client
from backend.models.tables import (
    User, Alert, Portfolio, Position, Stock, PriceHistory,
    Recommendation, News, Order, PortfolioPerformance, Watchlist, WatchlistItem
)
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Email configuration from environment
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
FROM_EMAIL = os.getenv('FROM_EMAIL', 'noreply@investmentapp.com')
ENABLE_EMAIL = os.getenv('ENABLE_EMAIL', 'false').lower() == 'true'

# Email templates
DAILY_SUMMARY_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; }
        .portfolio-card { background: #f4f4f4; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .recommendation { background: #3498db; color: white; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .footer { background: #34495e; color: white; padding: 10px; text-align: center; margin-top: 20px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #34495e; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Your Daily Investment Summary</h1>
        <p>{{ date }}</p>
    </div>
    <div class="content">
        <h2>Portfolio Performance</h2>
        {% for portfolio in portfolios %}
        <div class="portfolio-card">
            <h3>{{ portfolio.name }}</h3>
            <p>Total Value: ${{ "%.2f"|format(portfolio.total_value) }}</p>
            <p>Daily Change: <span class="{{ 'positive' if portfolio.daily_change > 0 else 'negative' }}">
                {{ "%.2f"|format(portfolio.daily_change) }} ({{ "%.2f"|format(portfolio.daily_change_percent) }}%)
            </span></p>
        </div>
        {% endfor %}
        
        <h2>Top Recommendations</h2>
        {% for rec in recommendations %}
        <div class="recommendation">
            <strong>{{ rec.symbol }}</strong> - {{ rec.recommendation_type }}
            <br>Target: ${{ "%.2f"|format(rec.target_price) }}
            <br>Confidence: {{ "%.0f"|format(rec.confidence * 100) }}%
        </div>
        {% endfor %}
        
        <h2>Market Overview</h2>
        <table>
            <tr>
                <th>Index</th>
                <th>Price</th>
                <th>Change</th>
                <th>% Change</th>
            </tr>
            {% for index in market_overview %}
            <tr>
                <td>{{ index.symbol }}</td>
                <td>${{ "%.2f"|format(index.price) }}</td>
                <td class="{{ 'positive' if index.change > 0 else 'negative' }}">{{ "%.2f"|format(index.change) }}</td>
                <td class="{{ 'positive' if index.change_percent > 0 else 'negative' }}">{{ "%.2f"|format(index.change_percent) }}%</td>
            </tr>
            {% endfor %}
        </table>
        
        <h2>Recent News</h2>
        {% for article in news %}
        <div style="margin: 10px 0;">
            <strong>{{ article.headline }}</strong><br>
            <small>{{ article.source }} - {{ article.published_at }}</small>
        </div>
        {% endfor %}
    </div>
    <div class="footer">
        <p>© 2024 Investment Analysis Platform | <a href="#" style="color: #3498db;">Unsubscribe</a></p>
    </div>
</body>
</html>
"""

ALERT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .alert-box { border: 2px solid #e74c3c; padding: 20px; margin: 20px; border-radius: 5px; }
        .alert-header { color: #e74c3c; font-size: 24px; font-weight: bold; }
        .details { margin-top: 15px; }
        .action-button { 
            display: inline-block; 
            padding: 10px 20px; 
            background: #3498db; 
            color: white; 
            text-decoration: none; 
            border-radius: 5px; 
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="alert-box">
        <div class="alert-header">⚠️ {{ alert_type }}</div>
        <div class="details">
            <p><strong>{{ symbol }}</strong></p>
            <p>{{ message }}</p>
            <p>Current Price: ${{ "%.2f"|format(current_price) }}</p>
            <p>Alert Condition: {{ condition }}</p>
            <p>Time: {{ timestamp }}</p>
        </div>
        <a href="{{ action_url }}" class="action-button">View Details</a>
    </div>
</body>
</html>
"""

@celery_app.task
def send_email(to_email: str, subject: str, html_content: str, attachments: List[Dict] = None) -> bool:
    """Send an email with HTML content and optional attachments"""
    if not ENABLE_EMAIL:
        logger.info(f"Email sending disabled. Would send to {to_email}: {subject}")
        return True
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Attach HTML content
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Add attachments if provided
        if attachments:
            for attachment in attachments:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment['content'])
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {attachment["filename"]}'
                )
                msg.attach(part)
        
        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            if SMTP_USER and SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending email to {to_email}: {e}")
        return False

@celery_app.task
def send_daily_summaries() -> Dict[str, Any]:
    """Send daily summary emails to all active users"""
    try:
        with get_db_sync() as db:
            # Get users with email notifications enabled
            users = db.query(User).filter(
                User.is_active == True,
                User.is_verified == True
            ).all()
            
            sent_count = 0
            failed_count = 0
            
            for user in users:
                try:
                    # Check if user wants daily summaries
                    notifications = user.notification_settings or {}
                    if not notifications.get('email_daily_summary', True):
                        continue
                    
                    # Gather user's data
                    summary_data = gather_daily_summary_data(db, user.id)
                    
                    if not summary_data:
                        continue
                    
                    # Render email template
                    template = Template(DAILY_SUMMARY_TEMPLATE)
                    html_content = template.render(**summary_data)
                    
                    # Send email
                    subject = f"Daily Investment Summary - {date.today().strftime('%B %d, %Y')}"
                    
                    send_email.delay(user.email, subject, html_content)
                    sent_count += 1
                    
                except Exception as e:
                    logger.error(f"Error sending summary to user {user.id}: {e}")
                    failed_count += 1
            
            return {
                'sent': sent_count,
                'failed': failed_count,
                'total_users': len(users)
            }
            
    except Exception as e:
        logger.error(f"Error sending daily summaries: {e}")
        return {'error': str(e)}

@celery_app.task
def check_price_alerts() -> Dict[str, Any]:
    """Check and trigger price alerts"""
    try:
        with get_db_sync() as db:
            # Get active alerts
            alerts = db.query(Alert).filter(
                Alert.is_active == True
            ).all()
            
            triggered = []
            
            for alert in alerts:
                try:
                    # Get current price
                    if alert.stock_id:
                        latest_price = db.query(PriceHistory).filter(
                            PriceHistory.stock_id == alert.stock_id
                        ).order_by(PriceHistory.date.desc()).first()
                        
                        if not latest_price:
                            continue
                        
                        current_price = float(latest_price.close)
                        
                        # Check alert conditions
                        condition = alert.condition
                        should_trigger = False
                        message = ""
                        
                        if condition.get('type') == 'price_above':
                            if current_price > condition.get('value', 0):
                                should_trigger = True
                                message = f"Price rose above ${condition['value']:.2f}"
                        
                        elif condition.get('type') == 'price_below':
                            if current_price < condition.get('value', float('inf')):
                                should_trigger = True
                                message = f"Price fell below ${condition['value']:.2f}"
                        
                        elif condition.get('type') == 'percent_change':
                            # Get previous close
                            yesterday = db.query(PriceHistory).filter(
                                PriceHistory.stock_id == alert.stock_id,
                                PriceHistory.date < latest_price.date
                            ).order_by(PriceHistory.date.desc()).first()
                            
                            if yesterday:
                                change_percent = (current_price - float(yesterday.close)) / float(yesterday.close) * 100
                                
                                if abs(change_percent) >= abs(condition.get('value', 0)):
                                    should_trigger = True
                                    message = f"Price changed by {change_percent:.2f}%"
                        
                        elif condition.get('type') == 'volume_spike':
                            avg_volume = db.query(func.avg(PriceHistory.volume)).filter(
                                PriceHistory.stock_id == alert.stock_id,
                                PriceHistory.date >= date.today() - timedelta(days=20)
                            ).scalar()
                            
                            if avg_volume and latest_price.volume > avg_volume * condition.get('multiplier', 2):
                                should_trigger = True
                                message = f"Volume spike detected: {latest_price.volume:,} vs avg {int(avg_volume):,}"
                        
                        if should_trigger:
                            # Send alert notification
                            stock = db.query(Stock).filter(Stock.id == alert.stock_id).first()
                            
                            triggered.append({
                                'alert_id': alert.id,
                                'user_id': alert.user_id,
                                'symbol': stock.symbol if stock else 'Unknown',
                                'message': message,
                                'current_price': current_price
                            })
                            
                            # Update alert
                            alert.triggered_count += 1
                            alert.last_triggered = datetime.utcnow()
                            
                            # Send notification based on user preferences
                            user = db.query(User).filter(User.id == alert.user_id).first()
                            if user:
                                send_alert_notification.delay(
                                    user.email,
                                    alert.alert_type,
                                    stock.symbol if stock else 'Unknown',
                                    message,
                                    current_price
                                )
                            
                            # Disable one-time alerts
                            if condition.get('one_time', False):
                                alert.is_active = False
                
                except Exception as e:
                    logger.error(f"Error checking alert {alert.id}: {e}")
            
            db.commit()
            
            return {
                'alerts_checked': len(alerts),
                'alerts_triggered': len(triggered),
                'triggered': triggered
            }
            
    except Exception as e:
        logger.error(f"Error checking price alerts: {e}")
        return {'error': str(e)}

@celery_app.task
def send_alert_notification(
    email: str,
    alert_type: str,
    symbol: str,
    message: str,
    current_price: float
) -> bool:
    """Send an alert notification email"""
    try:
        template = Template(ALERT_TEMPLATE)
        html_content = template.render(
            alert_type=alert_type.replace('_', ' ').title(),
            symbol=symbol,
            message=message,
            current_price=current_price,
            condition=alert_type,
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            action_url=f"https://app.example.com/stocks/{symbol}"
        )
        
        subject = f"🚨 Alert: {symbol} - {alert_type.replace('_', ' ').title()}"
        
        return send_email(email, subject, html_content)
        
    except Exception as e:
        logger.error(f"Error sending alert notification: {e}")
        return False

@celery_app.task
def send_order_confirmation(order_id: int) -> bool:
    """Send order confirmation email"""
    try:
        with get_db_sync() as db:
            order = db.query(Order).filter(Order.id == order_id).first()
            if not order:
                return False
            
            user = order.user
            stock = order.stock
            
            # Create confirmation email
            html_content = f"""
            <html>
            <body>
                <h2>Order Confirmation</h2>
                <p>Your order has been {order.status}.</p>
                <table>
                    <tr><td>Order ID:</td><td>{order.id}</td></tr>
                    <tr><td>Symbol:</td><td>{stock.symbol}</td></tr>
                    <tr><td>Type:</td><td>{order.order_type}</td></tr>
                    <tr><td>Side:</td><td>{order.order_side}</td></tr>
                    <tr><td>Quantity:</td><td>{order.quantity}</td></tr>
                    <tr><td>Price:</td><td>${order.average_fill_price or order.limit_price or 'Market'}</td></tr>
                    <tr><td>Status:</td><td>{order.status}</td></tr>
                    <tr><td>Time:</td><td>{order.filled_at or order.created_at}</td></tr>
                </table>
            </body>
            </html>
            """
            
            subject = f"Order {order.status.title()}: {stock.symbol}"
            
            return send_email(user.email, subject, html_content)
            
    except Exception as e:
        logger.error(f"Error sending order confirmation: {e}")
        return False

@celery_app.task
def send_recommendation_alert(recommendation_id: int) -> Dict[str, Any]:
    """Send alerts for new recommendations"""
    try:
        with get_db_sync() as db:
            recommendation = db.query(Recommendation).filter(
                Recommendation.id == recommendation_id
            ).first()
            
            if not recommendation:
                return {'error': 'Recommendation not found'}
            
            stock = recommendation.stock
            
            # Get users interested in this stock (watchlist or position holders)
            interested_users = set()
            
            # Users with positions
            positions = db.query(Position).filter(
                Position.stock_id == stock.id
            ).all()
            
            for position in positions:
                portfolio = position.portfolio
                interested_users.add(portfolio.user_id)
            
            # Users with stock in watchlist
            watchlist_items = db.query(WatchlistItem).filter(
                WatchlistItem.stock_id == stock.id
            ).all()
            
            for item in watchlist_items:
                interested_users.add(item.watchlist.user_id)
            
            # Send notifications
            sent = 0
            for user_id in interested_users:
                user = db.query(User).filter(User.id == user_id).first()
                if user and user.is_active:
                    html_content = f"""
                    <html>
                    <body>
                        <h2>New Recommendation: {stock.symbol}</h2>
                        <p><strong>{recommendation.recommendation_type.replace('_', ' ').title()}</strong></p>
                        <p>Confidence: {recommendation.confidence_score * 100:.0f}%</p>
                        <p>Current Price: ${recommendation.current_price:.2f}</p>
                        <p>Target Price: ${recommendation.target_price:.2f}</p>
                        <p>Reasoning: {recommendation.reasoning}</p>
                        <p>Key Factors:</p>
                        <ul>
                            {''.join(f'<li>{factor}</li>' for factor in (recommendation.key_factors or []))}
                        </ul>
                    </body>
                    </html>
                    """
                    
                    subject = f"New {recommendation.recommendation_type.replace('_', ' ').title()} Recommendation: {stock.symbol}"
                    
                    send_email.delay(user.email, subject, html_content)
                    sent += 1
            
            return {
                'recommendation_id': recommendation_id,
                'symbol': stock.symbol,
                'notifications_sent': sent
            }
            
    except Exception as e:
        logger.error(f"Error sending recommendation alert: {e}")
        return {'error': str(e)}

@celery_app.task
def send_portfolio_report(portfolio_id: int, period: str = 'monthly') -> bool:
    """Send detailed portfolio performance report"""
    try:
        with get_db_sync() as db:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            if not portfolio:
                return False
            
            user = portfolio.user
            
            # Calculate performance metrics
            if period == 'monthly':
                days = 30
            elif period == 'quarterly':
                days = 90
            else:
                days = 365
            
            performance = db.query(PortfolioPerformance).filter(
                PortfolioPerformance.portfolio_id == portfolio_id,
                PortfolioPerformance.date >= date.today() - timedelta(days=days)
            ).order_by(PortfolioPerformance.date).all()
            
            if not performance:
                return False
            
            # Calculate metrics
            start_value = float(performance[0].total_value)
            end_value = float(performance[-1].total_value)
            total_return = (end_value - start_value) / start_value
            
            # Get positions
            positions = db.query(Position).filter(
                Position.portfolio_id == portfolio_id
            ).all()
            
            # Create report
            html_content = f"""
            <html>
            <body>
                <h1>Portfolio Performance Report</h1>
                <h2>{portfolio.name}</h2>
                <p>Period: {period.title()} ({days} days)</p>
                
                <h3>Performance Summary</h3>
                <table border="1">
                    <tr><td>Starting Value:</td><td>${start_value:,.2f}</td></tr>
                    <tr><td>Ending Value:</td><td>${end_value:,.2f}</td></tr>
                    <tr><td>Total Return:</td><td>{total_return * 100:.2f}%</td></tr>
                    <tr><td>Annualized Return:</td><td>{total_return * 365 / days * 100:.2f}%</td></tr>
                </table>
                
                <h3>Current Positions</h3>
                <table border="1">
                    <tr>
                        <th>Symbol</th>
                        <th>Quantity</th>
                        <th>Avg Cost</th>
                        <th>Current Value</th>
                        <th>Gain/Loss</th>
                    </tr>
                    {''.join(f'''
                    <tr>
                        <td>{p.stock.symbol}</td>
                        <td>{p.quantity:.2f}</td>
                        <td>${p.average_cost:.2f}</td>
                        <td>${p.quantity * p.average_cost:.2f}</td>
                        <td>TBD</td>
                    </tr>
                    ''' for p in positions)}
                </table>
            </body>
            </html>
            """
            
            subject = f"{period.title()} Portfolio Report - {portfolio.name}"
            
            return send_email(user.email, subject, html_content)
            
    except Exception as e:
        logger.error(f"Error sending portfolio report: {e}")
        return False

@celery_app.task
def send_watchlist_updates() -> Dict[str, Any]:
    """Send updates for watchlist items"""
    try:
        with get_db_sync() as db:
            # Get all watchlists with alerts enabled
            watchlists = db.query(Watchlist).join(User).filter(
                User.is_active == True
            ).all()
            
            notifications_sent = 0
            
            for watchlist in watchlists:
                try:
                    items_with_changes = []
                    
                    for item in watchlist.items:
                        if not item.alert_enabled:
                            continue
                        
                        # Check if target price reached
                        latest_price = db.query(PriceHistory).filter(
                            PriceHistory.stock_id == item.stock_id
                        ).order_by(PriceHistory.date.desc()).first()
                        
                        if latest_price and item.target_price:
                            current_price = float(latest_price.close)
                            
                            if current_price >= float(item.target_price):
                                items_with_changes.append({
                                    'symbol': item.stock.symbol,
                                    'current_price': current_price,
                                    'target_price': float(item.target_price),
                                    'message': 'Target price reached!'
                                })
                    
                    if items_with_changes:
                        # Send notification
                        html_content = f"""
                        <html>
                        <body>
                            <h2>Watchlist Alert - {watchlist.name}</h2>
                            {''.join(f'''
                            <div>
                                <strong>{item["symbol"]}</strong><br>
                                Current: ${item["current_price"]:.2f}<br>
                                Target: ${item["target_price"]:.2f}<br>
                                {item["message"]}
                            </div>
                            <hr>
                            ''' for item in items_with_changes)}
                        </body>
                        </html>
                        """
                        
                        subject = f"Watchlist Alert: {len(items_with_changes)} Updates"
                        
                        send_email.delay(watchlist.user.email, subject, html_content)
                        notifications_sent += 1
                
                except Exception as e:
                    logger.error(f"Error processing watchlist {watchlist.id}: {e}")
            
            return {
                'watchlists_processed': len(watchlists),
                'notifications_sent': notifications_sent
            }
            
    except Exception as e:
        logger.error(f"Error sending watchlist updates: {e}")
        return {'error': str(e)}

# Helper functions
def gather_daily_summary_data(db: Session, user_id: int) -> Dict[str, Any]:
    """Gather data for daily summary email"""
    try:
        # Get user's portfolios
        portfolios = db.query(Portfolio).filter(
            Portfolio.user_id == user_id
        ).all()
        
        portfolio_data = []
        for portfolio in portfolios:
            # Get today's performance
            today_perf = db.query(PortfolioPerformance).filter(
                PortfolioPerformance.portfolio_id == portfolio.id,
                PortfolioPerformance.date == date.today()
            ).first()
            
            yesterday_perf = db.query(PortfolioPerformance).filter(
                PortfolioPerformance.portfolio_id == portfolio.id,
                PortfolioPerformance.date == date.today() - timedelta(days=1)
            ).first()
            
            if today_perf:
                daily_change = 0
                daily_change_percent = 0
                
                if yesterday_perf:
                    daily_change = float(today_perf.total_value - yesterday_perf.total_value)
                    daily_change_percent = (daily_change / float(yesterday_perf.total_value)) * 100
                
                portfolio_data.append({
                    'name': portfolio.name,
                    'total_value': float(today_perf.total_value),
                    'daily_change': daily_change,
                    'daily_change_percent': daily_change_percent
                })
        
        # Get top recommendations
        recommendations = db.query(Recommendation).filter(
            Recommendation.is_active == True,
            Recommendation.confidence_score >= 0.7
        ).order_by(Recommendation.confidence_score.desc()).limit(5).all()
        
        rec_data = []
        for rec in recommendations:
            rec_data.append({
                'symbol': rec.stock.symbol,
                'recommendation_type': rec.recommendation_type.replace('_', ' ').title(),
                'target_price': float(rec.target_price),
                'confidence': float(rec.confidence_score)
            })
        
        # Get market overview (simplified)
        market_indices = [
            {'symbol': 'SPY', 'price': 450.25, 'change': 2.15, 'change_percent': 0.48},
            {'symbol': 'QQQ', 'price': 380.50, 'change': -1.25, 'change_percent': -0.33},
            {'symbol': 'DIA', 'price': 350.75, 'change': 0.85, 'change_percent': 0.24}
        ]
        
        # Get recent news
        recent_news = db.query(News).order_by(
            News.published_at.desc()
        ).limit(5).all()
        
        news_data = []
        for article in recent_news:
            news_data.append({
                'headline': article.headline,
                'source': article.source,
                'published_at': article.published_at.strftime('%Y-%m-%d %H:%M')
            })
        
        return {
            'date': date.today().strftime('%B %d, %Y'),
            'portfolios': portfolio_data,
            'recommendations': rec_data,
            'market_overview': market_indices,
            'news': news_data
        }
        
    except Exception as e:
        logger.error(f"Error gathering daily summary data: {e}")
        return None

@celery_app.task
def send_test_notification(email: str) -> bool:
    """Send a test notification email"""
    try:
        html_content = """
        <html>
        <body>
            <h2>Test Notification</h2>
            <p>This is a test notification from the Investment Analysis Platform.</p>
            <p>If you received this email, your notifications are working correctly!</p>
            <p>Time: """ + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC') + """</p>
        </body>
        </html>
        """
        
        subject = "Test Notification - Investment Analysis Platform"
        
        return send_email(email, subject, html_content)
        
    except Exception as e:
        logger.error(f"Error sending test notification: {e}")
        return False