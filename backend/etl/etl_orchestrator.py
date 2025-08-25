"""
ETL Orchestrator
Main orchestration module for the complete ETL pipeline
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from dotenv import load_dotenv

# Import ETL modules
from backend.etl.data_extractor import DataExtractor, DataValidator
from backend.etl.data_transformer import DataTransformer, DataAggregator
from backend.etl.data_loader import DataLoader, BatchLoader

# Import new multi-source system
from backend.etl.multi_source_extractor import MultiSourceStockExtractor, ExtractionResult
from backend.etl.distributed_batch_processor import DistributedBatchProcessor, ProcessorConfig
from backend.etl.data_validator import FinancialDataValidator, ValidationLevel, validate_extraction_results

# Import ML modules (optional)
try:
    from backend.ml.ensemble_model import EnsemblePredictor
    from backend.ml.recommendation_engine import RecommendationEngine
    HAS_ML = True
except ImportError:
    HAS_ML = False
    EnsemblePredictor = None
    RecommendationEngine = None

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ETLOrchestrator:
    """Enhanced ETL orchestrator with multi-source extraction and distributed processing"""
    
    def __init__(self, use_distributed: bool = True, cache_dir: str = "/tmp/stock_cache"):
        # Legacy components (maintained for backward compatibility)
        self.legacy_extractor = DataExtractor()
        self.transformer = DataTransformer()
        self.loader = DataLoader()
        self.batch_loader = BatchLoader(self.loader)
        self.legacy_validator = DataValidator()
        self.aggregator = DataAggregator()
        
        # Enhanced multi-source components
        self.multi_source_extractor = MultiSourceStockExtractor(cache_dir=cache_dir)
        self.data_validator = FinancialDataValidator(ValidationLevel.STANDARD)
        
        # Distributed processing
        self.use_distributed = use_distributed
        self.distributed_processor = None
        if use_distributed:
            processor_config = ProcessorConfig(
                max_concurrent_jobs=3,
                max_concurrent_per_job=8,
                batch_size=50,
                cache_dir=cache_dir,
                job_timeout_hours=12,
                retry_failed_jobs=True
            )
            self.distributed_processor = DistributedBatchProcessor(processor_config)
        
        # ML components
        self.predictor = None  # Initialize on demand
        self.recommender = None  # Initialize on demand
        
        # Pipeline configuration
        self.config = {
            'batch_size': 50 if use_distributed else 20,
            'max_workers': 8 if use_distributed else 4,
            'enable_ml': HAS_ML,
            'enable_sentiment': True,
            'enable_recommendations': HAS_ML,
            'enable_validation': True,
            'min_quality_score': 70.0,
            'use_caching': True,
            'max_tickers': None,  # Set to limit for testing
            'validation_level': 'standard'
        }
        
        # Pipeline metrics
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'stocks_processed': 0,
            'stocks_failed': 0,
            'stocks_validated': 0,
            'recommendations_generated': 0,
            'data_quality_avg': 0.0,
            'extraction_source_stats': {},
            'errors': []
        }
    
    async def run_full_pipeline(self, tickers: List[str] = None) -> Dict:
        """Run the enhanced ETL pipeline with multi-source extraction"""
        try:
            self.metrics['start_time'] = datetime.now()
            logger.info(f"Starting enhanced ETL pipeline at {self.metrics['start_time']}")
            
            # Get tickers if not provided
            if not tickers:
                tickers = await self.get_active_tickers()
            
            logger.info(f"Processing {len(tickers)} tickers with {'distributed' if self.use_distributed else 'standard'} processing")
            
            # Choose processing method based on configuration
            if self.use_distributed and len(tickers) > 100:
                return await self._run_distributed_pipeline(tickers)
            else:
                return await self._run_standard_pipeline(tickers)
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            self.metrics['errors'].append(str(e))
            return self.metrics
    
    async def _run_distributed_pipeline(self, tickers: List[str]) -> Dict:
        """Run pipeline using distributed batch processing"""
        logger.info(f"Running distributed pipeline for {len(tickers)} tickers")
        
        try:
            # Create distributed processing jobs
            job_ids = self.distributed_processor.create_jobs_from_ticker_list(
                all_tickers=tickers,
                tickers_per_job=200,
                priority=1
            )
            
            logger.info(f"Created {len(job_ids)} processing jobs")
            
            # Start distributed processing
            processing_task = asyncio.create_task(
                self.distributed_processor.start_processing()
            )
            
            # Monitor progress
            completed_jobs = 0
            while completed_jobs < len(job_ids):
                await asyncio.sleep(30)  # Check every 30 seconds
                
                completed_count = 0
                for job_id in job_ids:
                    status = self.distributed_processor.get_job_status(job_id)
                    if status and status['status'] == 'completed':
                        completed_count += 1
                
                if completed_count > completed_jobs:
                    completed_jobs = completed_count
                    logger.info(f"Progress: {completed_jobs}/{len(job_ids)} jobs completed")
            
            # Stop processing and collect results
            self.distributed_processor.stop_processing()
            
            # Update metrics from processor stats
            processor_stats = self.distributed_processor.get_processor_stats()
            self.metrics.update({
                'stocks_processed': processor_stats.get('total_tickers_processed', 0),
                'stocks_failed': processor_stats.get('total_failed_extractions', 0),
                'end_time': datetime.now()
            })
            
            duration = (self.metrics['end_time'] - self.metrics['start_time']).total_seconds()
            logger.info(f"Distributed pipeline completed in {duration:.2f} seconds")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Distributed pipeline error: {e}")
            self.metrics['errors'].append(str(e))
            return self.metrics
    
    async def _run_standard_pipeline(self, tickers: List[str]) -> Dict:
        """Run standard pipeline with enhanced multi-source extraction"""
        logger.info(f"Running standard pipeline for {len(tickers)} tickers")
        
        # Phase 1: Enhanced Multi-Source Extraction
        extraction_results = await self.enhanced_extract_phase(tickers)
        
        # Phase 2: Data Validation and Quality Assurance
        validated_data = []
        if self.config['enable_validation']:
            validated_data = await self.validation_phase(extraction_results)
        else:
            validated_data = [r.data for r in extraction_results if r.success and r.data]
        
        # Phase 3: Transform data
        transformed_data = await self.transform_phase(validated_data)
        
        # Phase 4: Load data
        await self.load_phase(transformed_data)
        
        # Phase 5: Generate predictions and recommendations
        if self.config['enable_ml']:
            recommendations = await self.ml_phase(transformed_data)
            self.metrics['recommendations_generated'] = len(recommendations)
        
        # Phase 6: Cleanup and reporting
        await self.cleanup_phase()
        
        self.metrics['end_time'] = datetime.now()
        duration = (self.metrics['end_time'] - self.metrics['start_time']).total_seconds()
        
        logger.info(f"Enhanced ETL pipeline completed in {duration:.2f} seconds")
        logger.info(f"Processed: {self.metrics['stocks_processed']} stocks")
        logger.info(f"Failed: {self.metrics['stocks_failed']} stocks") 
        logger.info(f"Data Quality Average: {self.metrics['data_quality_avg']:.1f}%")
        logger.info(f"Recommendations: {self.metrics['recommendations_generated']}")
        
        # Log extraction source statistics
        if self.metrics['extraction_source_stats']:
            logger.info("Extraction Source Statistics:")
            for source, count in self.metrics['extraction_source_stats'].items():
                logger.info(f"  {source}: {count} successful extractions")
        
        return self.metrics
    
    async def get_active_tickers(self) -> List[str]:
        """Get list of active tickers from database"""
        try:
            # Import the stock universe manager
            from backend.etl.stock_universe_manager import StockUniverseManager
            manager = StockUniverseManager()
            
            # Get all active tickers from database
            tickers = manager.get_all_active_tickers()
            
            if not tickers:
                logger.warning("No tickers in database, populating...")
                # Populate database if empty
                count = manager.populate_database_with_all_stocks()
                logger.info(f"Populated database with {count} stocks")
                # Get tickers again
                tickers = manager.get_all_active_tickers()
            
            # Apply max_tickers limit if configured
            max_tickers = self.config.get('max_tickers')
            if max_tickers and max_tickers < len(tickers):
                logger.info(f"Limiting to {max_tickers} tickers (from {len(tickers)} total)")
                return tickers[:max_tickers]
            
            logger.info(f"Processing ALL {len(tickers)} stocks from database")
            return tickers
            
        except Exception as e:
            logger.error(f"Error getting active tickers: {e}")
            # Fallback to default list if database fails
            default_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
                'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA'
            ]
            return default_tickers
    
    async def enhanced_extract_phase(self, tickers: List[str]) -> List[ExtractionResult]:
        """Enhanced extraction phase using multi-source extractor"""
        logger.info("Starting enhanced multi-source extraction phase")
        
        try:
            # Use the multi-source extractor for better rate limiting and fallback
            results = await self.multi_source_extractor.batch_extract(
                tickers=tickers,
                batch_size=self.config['batch_size'],
                max_concurrent=self.config['max_workers']
            )
            
            # Update metrics
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            self.metrics['stocks_processed'] = len(successful_results)
            self.metrics['stocks_failed'] = len(failed_results)
            
            # Track source statistics
            source_stats = {}
            for result in successful_results:
                if result.source:
                    source_stats[result.source] = source_stats.get(result.source, 0) + 1
            
            self.metrics['extraction_source_stats'] = source_stats
            
            # Log any failed extractions
            for result in failed_results:
                logger.warning(f"Failed to extract {result.ticker}: {result.error}")
                self.metrics['errors'].append(f"Extraction failed for {result.ticker}: {result.error}")
            
            logger.info(f"Enhanced extraction complete: {len(successful_results)}/{len(tickers)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Enhanced extraction phase failed: {e}")
            self.metrics['errors'].append(f"Enhanced extraction phase error: {str(e)}")
            return []
    
    async def validation_phase(self, extraction_results: List[ExtractionResult]) -> List[Dict]:
        """Validate extracted data and filter based on quality"""
        logger.info("Starting data validation phase")
        
        try:
            # Extract successful data for validation
            raw_data = []
            for result in extraction_results:
                if result.success and result.data:
                    raw_data.append(result.data)
            
            if not raw_data:
                logger.warning("No data available for validation")
                return []
            
            # Get validation level from config
            validation_level_map = {
                'basic': ValidationLevel.BASIC,
                'standard': ValidationLevel.STANDARD,
                'strict': ValidationLevel.STRICT,
                'comprehensive': ValidationLevel.COMPREHENSIVE
            }
            validation_level = validation_level_map.get(
                self.config.get('validation_level', 'standard'),
                ValidationLevel.STANDARD
            )
            
            # Validate all extraction results
            validation_summary = validate_extraction_results(
                results=raw_data,
                validation_level=validation_level,
                min_quality_score=self.config.get('min_quality_score', 70.0)
            )
            
            # Update metrics
            self.metrics['stocks_validated'] = validation_summary['valid_records']
            if validation_summary['quality_scores']:
                self.metrics['data_quality_avg'] = validation_summary['avg_quality_score']
            
            # Log validation summary
            logger.info(f"Validation complete: {validation_summary['valid_records']}/{validation_summary['total_records']} records passed")
            logger.info(f"Average quality score: {validation_summary.get('avg_quality_score', 0):.1f}%")
            
            # Log common issues
            if validation_summary.get('common_issues'):
                logger.info("Most common validation issues:")
                sorted_issues = sorted(
                    validation_summary['common_issues'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for issue, count in sorted_issues[:5]:  # Show top 5 issues
                    logger.info(f"  {issue}: {count} occurrences")
            
            # Return validated data
            validated_data = [item['data'] for item in validation_summary['filtered_results']]
            logger.info(f"Returning {len(validated_data)} validated records")
            
            return validated_data
            
        except Exception as e:
            logger.error(f"Validation phase failed: {e}")
            self.metrics['errors'].append(f"Validation phase error: {str(e)}")
            # Return original data if validation fails
            return [r.data for r in extraction_results if r.success and r.data]
    
    async def extract_phase(self, tickers: List[str]) -> List[Dict]:
        """Phase 1: Extract data from multiple sources"""
        logger.info("Starting extraction phase")
        extracted_data = []
        
        # Process in batches
        batch_size = self.config['batch_size']
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Extracting batch {i//batch_size + 1}: {batch}")
            
            # Extract data for batch
            batch_data = await self.extractor.batch_extract(batch, batch_size=10)
            
            for data in batch_data:
                if data and self.validate_extracted_data(data):
                    extracted_data.append(data)
                    self.metrics['stocks_processed'] += 1
                else:
                    self.metrics['stocks_failed'] += 1
            
            # Rate limiting delay
            await asyncio.sleep(1)
        
        logger.info(f"Extraction complete: {len(extracted_data)} stocks extracted")
        return extracted_data
    
    def validate_extracted_data(self, data: Dict) -> bool:
        """Validate extracted data"""
        try:
            # Check if data has required sources
            if 'sources' not in data or not data['sources']:
                return False
            
            # At minimum, we need price data
            has_price_data = False
            for source_name, source_data in data['sources'].items():
                if source_name == 'yfinance' and 'price_data' in source_data:
                    has_price_data = True
                    break
                elif source_name == 'polygon' and 'aggregates' in source_data:
                    has_price_data = True
                    break
            
            return has_price_data
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def transform_phase(self, extracted_data: List[Dict]) -> List[Dict]:
        """Phase 2: Transform extracted data"""
        logger.info("Starting transformation phase")
        transformed_data = []
        
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = []
            
            for data in extracted_data:
                future = executor.submit(self.transform_single_stock, data)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    if result:
                        transformed_data.append(result)
                except Exception as e:
                    logger.error(f"Transformation error: {e}")
                    self.metrics['errors'].append(str(e))
        
        logger.info(f"Transformation complete: {len(transformed_data)} stocks transformed")
        return transformed_data
    
    def transform_single_stock(self, raw_data: Dict) -> Dict:
        """Transform data for a single stock"""
        try:
            ticker = raw_data.get('ticker')
            logger.debug(f"Transforming data for {ticker}")
            
            # Transform price data
            price_df = self.transformer.transform_price_data(raw_data)
            
            if price_df.empty:
                logger.warning(f"No price data for {ticker}")
                return None
            
            # Get company info
            company_info = None
            if 'yfinance' in raw_data['sources']:
                company_info = raw_data['sources']['yfinance'].get('company_info')
            
            # Get sentiment data
            sentiment_data = None
            for source in ['newsapi', 'finnhub']:
                if source in raw_data['sources']:
                    sentiment_data = raw_data['sources'][source]
                    break
            
            # Create feature matrix
            features_df = self.transformer.create_feature_matrix(
                price_df, 
                sentiment_data, 
                company_info
            )
            
            # Transform sentiment
            transformed_sentiment = None
            if sentiment_data:
                transformed_sentiment = self.transformer.transform_sentiment_data(sentiment_data)
            
            return {
                'ticker': ticker,
                'price_df': price_df,
                'features_df': features_df,
                'company_info': company_info,
                'sentiment': transformed_sentiment,
                'extraction_time': raw_data.get('extraction_time')
            }
            
        except Exception as e:
            logger.error(f"Error transforming {raw_data.get('ticker')}: {e}")
            return None
    
    async def load_phase(self, transformed_data: List[Dict]) -> None:
        """Phase 3: Load transformed data into database"""
        logger.info("Starting loading phase")
        
        for data in transformed_data:
            if not data:
                continue
            
            ticker = data['ticker']
            
            try:
                # Load price data
                if 'price_df' in data and not data['price_df'].empty:
                    self.loader.load_price_data(data['price_df'], ticker)
                
                # Load technical indicators
                if 'features_df' in data and not data['features_df'].empty:
                    self.loader.load_technical_indicators(data['features_df'], ticker)
                
                # Load sentiment data
                if data.get('sentiment'):
                    self.loader.load_sentiment_data(data['sentiment'], ticker)
                
                logger.debug(f"Loaded data for {ticker}")
                
            except Exception as e:
                logger.error(f"Loading error for {ticker}: {e}")
                self.metrics['errors'].append(f"Load error for {ticker}: {str(e)}")
        
        logger.info("Loading phase complete")
    
    async def ml_phase(self, transformed_data: List[Dict]) -> List[Dict]:
        """Phase 4: Generate ML predictions and recommendations"""
        logger.info("Starting ML phase")
        
        if not HAS_ML:
            logger.warning("ML modules not available - skipping ML phase")
            return []
        
        try:
            # Initialize ML components if needed
            if self.predictor is None and EnsemblePredictor:
                self.predictor = EnsemblePredictor()
            if self.recommender is None and RecommendationEngine:
                self.recommender = RecommendationEngine()
            
            all_predictions = []
            
            for data in transformed_data:
                if not data or 'features_df' not in data:
                    continue
                
                ticker = data['ticker']
                features_df = data['features_df']
                
                if features_df.empty or len(features_df) < 30:
                    continue
                
                try:
                    # Generate predictions
                    predictions = self.predictor.predict(features_df)
                    
                    # Store predictions
                    self.loader.load_ml_predictions({
                        'model_name': 'ensemble',
                        'model_version': '1.0',
                        'predicted_return': predictions.get('predicted_return', 0),
                        'confidence': predictions.get('confidence', 0.5),
                        'horizon_days': 5,
                        'features': predictions.get('important_features', [])
                    }, ticker)
                    
                    all_predictions.append({
                        'ticker': ticker,
                        'prediction': predictions,
                        'features': data
                    })
                    
                except Exception as e:
                    logger.error(f"ML error for {ticker}: {e}")
            
            # Generate recommendations
            recommendations = []
            if all_predictions and self.config['enable_recommendations']:
                recommendations = self.recommender.generate_recommendations(all_predictions)
                
                # Load recommendations
                if recommendations:
                    self.loader.load_recommendations(recommendations)
            
            logger.info(f"ML phase complete: {len(recommendations)} recommendations generated")
            return recommendations
            
        except Exception as e:
            logger.error(f"ML phase error: {e}")
            return []
    
    async def cleanup_phase(self) -> None:
        """Phase 5: Cleanup old data and optimize database"""
        logger.info("Starting cleanup phase")
        
        try:
            # Clean up old data
            self.loader.cleanup_old_data()
            
            # Get final statistics
            stats = self.loader.get_loading_stats()
            logger.info(f"Final database stats: {stats}")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            self.metrics['errors'].append(f"Cleanup error: {str(e)}")
    
    async def run_incremental_update(self, tickers: List[str]) -> Dict:
        """Run incremental update for specific tickers"""
        logger.info(f"Running incremental update for {len(tickers)} tickers")
        
        # Use smaller batch size for incremental updates
        original_batch_size = self.config['batch_size']
        self.config['batch_size'] = min(5, original_batch_size)
        
        result = await self.run_full_pipeline(tickers)
        
        # Restore original batch size
        self.config['batch_size'] = original_batch_size
        
        return result
    
    async def run_realtime_update(self, ticker: str) -> Dict:
        """Run real-time update for a single ticker"""
        logger.info(f"Running real-time update for {ticker}")
        
        try:
            # Extract latest data
            data = await self.extractor.extract_all_data(ticker)
            
            if not data or not self.validate_extracted_data(data):
                return {'status': 'error', 'message': 'Failed to extract data'}
            
            # Transform
            transformed = self.transform_single_stock(data)
            
            if not transformed:
                return {'status': 'error', 'message': 'Failed to transform data'}
            
            # Load
            await self.load_phase([transformed])
            
            # Generate prediction if enabled
            prediction = None
            if self.config['enable_ml']:
                predictions = await self.ml_phase([transformed])
                if predictions:
                    prediction = predictions[0]
            
            return {
                'status': 'success',
                'ticker': ticker,
                'timestamp': datetime.now(),
                'prediction': prediction
            }
            
        except Exception as e:
            logger.error(f"Real-time update error for {ticker}: {e}")
            return {'status': 'error', 'message': str(e)}


class ETLScheduler:
    """Schedule and manage ETL pipeline runs"""
    
    def __init__(self):
        self.orchestrator = ETLOrchestrator()
        self.is_running = False
    
    async def run_daily_pipeline(self):
        """Run daily ETL pipeline"""
        if self.is_running:
            logger.warning("Pipeline already running, skipping")
            return
        
        self.is_running = True
        
        try:
            logger.info("Starting daily ETL pipeline")
            result = await self.orchestrator.run_full_pipeline()
            
            # Log results
            logger.info(f"Daily pipeline completed: {result}")
            
            # Send notifications if needed
            if result.get('errors'):
                logger.error(f"Pipeline errors: {result['errors']}")
            
        except Exception as e:
            logger.error(f"Daily pipeline failed: {e}")
        finally:
            self.is_running = False
    
    async def run_hourly_update(self):
        """Run hourly incremental updates"""
        if self.is_running:
            return
        
        try:
            # Get top movers or watchlist stocks
            watchlist = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
            
            logger.info(f"Running hourly update for {watchlist}")
            await self.orchestrator.run_incremental_update(watchlist)
            
        except Exception as e:
            logger.error(f"Hourly update failed: {e}")


async def main():
    """Main entry point for testing"""
    orchestrator = ETLOrchestrator()
    
    # Test with a few tickers
    test_tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    logger.info(f"Testing ETL pipeline with {test_tickers}")
    result = await orchestrator.run_full_pipeline(test_tickers)
    
    print("\nPipeline Results:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())