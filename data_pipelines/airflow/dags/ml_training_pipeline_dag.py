"""
Airflow DAG for ML Model Training Pipeline with Automated Retraining
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.utils.dates import days_ago
from airflow.models import Variable
import sys
import os
import asyncio
import json
import logging

# Add backend to path
sys.path.append('/app')

from backend.ml.pipeline import (
    MLOrchestrator,
    OrchestratorConfig,
    TrainingSchedule,
    ScheduleFrequency,
    RetrainingTrigger,
    TriggerType,
    ModelRegistry,
    ModelMonitor,
    DriftDetector,
    ModelDeployer,
    DeploymentConfig,
    DeploymentStrategy,
    DeploymentEnvironment,
    ABTestManager,
    ABTestConfig
)
from backend.ml.pipeline.implementations import (
    create_pipeline,
    PipelineConfig,
    ModelType
)

logger = logging.getLogger(__name__)

# Default DAG arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['ml-alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Automated ML Model Training and Retraining Pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'training', 'production'],
)


def initialize_orchestrator(**context):
    """Initialize ML Orchestrator"""
    config = OrchestratorConfig(
        name="airflow_ml_orchestrator",
        max_concurrent_pipelines=3,
        enable_scheduling=True,
        enable_auto_retraining=True,
        retraining_triggers=[
            RetrainingTrigger(
                trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                min_accuracy=0.75,
                max_error_rate=0.15
            ),
            RetrainingTrigger(
                trigger_type=TriggerType.DATA_DRIFT,
                drift_threshold=0.35
            ),
            RetrainingTrigger(
                trigger_type=TriggerType.NEW_DATA_THRESHOLD,
                min_new_samples=5000
            )
        ],
        cost_limit_daily_usd=10.0,
        models_path="/app/ml_models",
        logs_path="/app/ml_logs"
    )
    
    orchestrator = MLOrchestrator(config)
    
    # Store in XCom for other tasks
    context['task_instance'].xcom_push(key='orchestrator_config', value=config.__dict__)
    
    return "Orchestrator initialized"


def check_data_quality(**context):
    """Check data quality and determine if training should proceed"""
    from backend.utils.data_quality import DataQualityChecker
    
    checker = DataQualityChecker()
    
    # Get recent data statistics
    quality_report = checker.check_recent_data_quality(
        table="price_history",
        days_back=7
    )
    
    # Check quality thresholds
    if quality_report['missing_data_rate'] > 0.1:
        logger.warning(f"High missing data rate: {quality_report['missing_data_rate']}")
        return "skip_training"
    
    if quality_report['anomaly_rate'] > 0.05:
        logger.warning(f"High anomaly rate: {quality_report['anomaly_rate']}")
        return "needs_review"
    
    # Store quality report
    context['task_instance'].xcom_push(key='data_quality_report', value=quality_report)
    
    return "proceed_training"


def check_drift(**context):
    """Check for data and concept drift"""
    
    async def _check_drift():
        registry = ModelRegistry()
        monitor = ModelMonitor()
        drift_detector = DriftDetector()
        
        # Get deployed models
        deployed_models = await registry.get_deployed_models()
        
        drift_reports = []
        
        for model in deployed_models:
            model_name = model['model_name']
            
            # Get recent predictions and actuals
            recent_data = await monitor.get_model_metrics(model_name)
            
            if recent_data:
                # Check data drift
                data_drift = await drift_detector.detect_data_drift(
                    recent_data.get('recent_features', pd.DataFrame())
                )
                
                # Check concept drift
                if 'predictions' in recent_data and 'actuals' in recent_data:
                    concept_drift = await drift_detector.detect_concept_drift(
                        recent_data['predictions'],
                        recent_data['actuals']
                    )
                else:
                    concept_drift = None
                
                drift_reports.append({
                    'model_name': model_name,
                    'data_drift': data_drift.to_dict() if data_drift else None,
                    'concept_drift': concept_drift.to_dict() if concept_drift else None,
                    'needs_retraining': (
                        (data_drift and data_drift.is_drift_detected) or
                        (concept_drift and concept_drift.is_drift_detected)
                    )
                })
        
        return drift_reports
    
    drift_reports = asyncio.run(_check_drift())
    
    # Store drift reports
    context['task_instance'].xcom_push(key='drift_reports', value=drift_reports)
    
    # Determine if any model needs retraining
    needs_retraining = any(report['needs_retraining'] for report in drift_reports)
    
    return "trigger_retraining" if needs_retraining else "no_drift_detected"


def prepare_training_data(**context):
    """Prepare training data for all models"""
    import pandas as pd
    from sqlalchemy import create_engine
    
    # Get database connection
    db_url = Variable.get("DATABASE_URL")
    engine = create_engine(db_url)
    
    # Query for training data
    query = """
    SELECT 
        ph.*,
        ti.*,
        s.sector,
        s.industry,
        s.market_cap
    FROM price_history ph
    JOIN technical_indicators ti ON ph.stock_id = ti.stock_id AND ph.date = ti.date
    JOIN stocks s ON ph.stock_id = s.id
    WHERE ph.date >= CURRENT_DATE - INTERVAL '2 years'
    ORDER BY ph.date
    """
    
    df = pd.read_sql(query, engine)
    
    # Basic feature engineering
    df['returns'] = df.groupby('stock_id')['close'].pct_change()
    df['volatility'] = df.groupby('stock_id')['returns'].rolling(20).std().reset_index(0, drop=True)
    df['volume_ratio'] = df['volume'] / df.groupby('stock_id')['volume'].rolling(20).mean().reset_index(0, drop=True)
    
    # Create target variable (next day return)
    df['future_return'] = df.groupby('stock_id')['returns'].shift(-1)
    
    # Remove NaN values
    df = df.dropna()
    
    # Save to temporary location
    data_path = f"/tmp/training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    df.to_parquet(data_path)
    
    # Store path and statistics
    context['task_instance'].xcom_push(key='training_data_path', value=data_path)
    context['task_instance'].xcom_push(key='training_data_stats', value={
        'samples': len(df),
        'features': len(df.columns),
        'stocks': df['stock_id'].nunique()
    })
    
    return data_path


def train_models(**context):
    """Train all ML models"""
    
    async def _train_models():
        # Get training data path
        data_path = context['task_instance'].xcom_pull(key='training_data_path')
        
        # Initialize orchestrator
        orchestrator = MLOrchestrator()
        await orchestrator.start()
        
        # Define model configurations
        model_configs = [
            {
                'name': 'stock_prediction_xgboost',
                'model_type': ModelType.TIME_SERIES,
                'hyperparameters': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.01,
                    'subsample': 0.8
                }
            },
            {
                'name': 'stock_prediction_lightgbm',
                'model_type': ModelType.TIME_SERIES,
                'hyperparameters': {
                    'n_estimators': 200,
                    'num_leaves': 31,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.8
                }
            },
            {
                'name': 'stock_prediction_ensemble',
                'model_type': ModelType.ENSEMBLE,
                'hyperparameters': {
                    'models': ['xgboost', 'lightgbm', 'random_forest'],
                    'voting': 'soft'
                }
            }
        ]
        
        training_results = []
        
        for model_config in model_configs:
            # Create pipeline configuration
            pipeline_config = PipelineConfig(
                name=model_config['name'],
                version=f"1.0.{datetime.now().strftime('%Y%m%d')}",
                model_type=model_config['model_type'],
                data_source=data_path,
                feature_columns=[],  # Auto-detect
                target_column='future_return',
                hyperparameters=model_config['hyperparameters'],
                cost_limit_usd=2.0  # Per model
            )
            
            # Create and submit pipeline
            pipeline = create_pipeline(pipeline_config)
            pipeline_id = await orchestrator.submit_pipeline(
                pipeline,
                TriggerType.SCHEDULED
            )
            
            # Wait for completion (with timeout)
            max_wait = 3600  # 1 hour
            start_time = datetime.utcnow()
            
            while (datetime.utcnow() - start_time).total_seconds() < max_wait:
                status = orchestrator.get_status()
                
                # Check if our pipeline completed
                for result in orchestrator.pipeline_history:
                    if result.pipeline_id == pipeline_id:
                        if result.status in ['completed', 'failed']:
                            training_results.append({
                                'model_name': model_config['name'],
                                'pipeline_id': pipeline_id,
                                'status': result.status,
                                'metrics': result.model_artifact.metrics if result.model_artifact else {},
                                'duration': result.total_compute_time_seconds
                            })
                            break
                else:
                    await asyncio.sleep(30)
                    continue
                break
        
        await orchestrator.stop()
        return training_results
    
    results = asyncio.run(_train_models())
    
    # Store results
    context['task_instance'].xcom_push(key='training_results', value=results)
    
    # Check if all models trained successfully
    all_success = all(r['status'] == 'completed' for r in results)
    
    return "training_success" if all_success else "training_failed"


def evaluate_models(**context):
    """Evaluate and compare trained models"""
    
    async def _evaluate_models():
        registry = ModelRegistry()
        monitor = ModelMonitor()
        
        training_results = context['task_instance'].xcom_pull(key='training_results')
        
        evaluation_results = []
        
        for result in training_results:
            if result['status'] == 'completed':
                model_name = result['model_name']
                
                # Get model from registry
                model_version = await registry.get_model(model_name)
                
                if model_version:
                    # Calculate additional metrics
                    metrics = monitor.calculate_metrics(
                        y_true=np.random.randn(100),  # Would use actual test data
                        y_pred=np.random.randn(100),  # Would use actual predictions
                        model_type='regression'
                    )
                    
                    evaluation_results.append({
                        'model_name': model_name,
                        'version': model_version.version,
                        'training_metrics': result['metrics'],
                        'test_metrics': metrics.to_dict(),
                        'ranking_score': metrics.r2  # Or custom ranking metric
                    })
        
        # Rank models
        evaluation_results.sort(key=lambda x: x.get('ranking_score', 0), reverse=True)
        
        return evaluation_results
    
    evaluation_results = asyncio.run(_evaluate_models())
    
    # Store results
    context['task_instance'].xcom_push(key='evaluation_results', value=evaluation_results)
    
    # Determine best model
    if evaluation_results:
        best_model = evaluation_results[0]
        context['task_instance'].xcom_push(key='best_model', value=best_model)
        return "evaluation_complete"
    
    return "no_models_to_evaluate"


def deploy_best_model(**context):
    """Deploy the best performing model"""
    
    async def _deploy_model():
        registry = ModelRegistry()
        deployer = ModelDeployer(registry, ModelMonitor())
        
        best_model = context['task_instance'].xcom_pull(key='best_model')
        
        if not best_model:
            return None
        
        # Create deployment configuration
        deploy_config = DeploymentConfig(
            model_name=best_model['model_name'],
            model_version=best_model['version'],
            environment=DeploymentEnvironment.STAGING,  # Deploy to staging first
            strategy=DeploymentStrategy.CANARY,
            endpoint_url=f"http://ml-api/{best_model['model_name']}",
            canary_percentage=10.0,
            rollout_duration_minutes=60,
            auto_rollback=True,
            rollback_threshold_error_rate=0.1
        )
        
        # Deploy model
        deployment_status = await deployer.deploy(deploy_config)
        
        return {
            'deployment_id': deployment_status.deployment_id,
            'status': deployment_status.status,
            'endpoints': deployment_status.endpoints,
            'model_name': best_model['model_name'],
            'model_version': best_model['version']
        }
    
    deployment_result = asyncio.run(_deploy_model())
    
    # Store deployment result
    context['task_instance'].xcom_push(key='deployment_result', value=deployment_result)
    
    return "deployment_success" if deployment_result else "deployment_failed"


def run_ab_test(**context):
    """Run A/B test between new and existing model"""
    
    async def _run_ab_test():
        monitor = ModelMonitor()
        ab_manager = ABTestManager(monitor)
        
        deployment_result = context['task_instance'].xcom_pull(key='deployment_result')
        
        if not deployment_result:
            return None
        
        # Create A/B test configuration
        ab_config = ABTestConfig(
            test_name=f"model_comparison_{datetime.now().strftime('%Y%m%d')}",
            model_a_name=f"{deployment_result['model_name']}_current",
            model_a_version="production",
            model_b_name=deployment_result['model_name'],
            model_b_version=deployment_result['model_version'],
            traffic_percentage_a=50.0,
            duration_hours=24,
            primary_metric="accuracy",
            minimum_sample_size=1000
        )
        
        # Start A/B test
        test_id = await ab_manager.start_test(ab_config)
        
        # Note: In production, you would wait for test completion
        # For this example, we'll just return the test setup
        
        return {
            'test_id': test_id,
            'test_config': ab_config.__dict__,
            'status': 'running'
        }
    
    ab_test_result = asyncio.run(_run_ab_test())
    
    # Store A/B test result
    context['task_instance'].xcom_push(key='ab_test_result', value=ab_test_result)
    
    return "ab_test_started" if ab_test_result else "ab_test_skipped"


def monitor_deployment(**context):
    """Monitor deployed model performance"""
    
    async def _monitor_deployment():
        monitor = ModelMonitor()
        registry = ModelRegistry()
        
        deployment_result = context['task_instance'].xcom_pull(key='deployment_result')
        
        if not deployment_result:
            return None
        
        # Register model for monitoring
        await monitor.register_model(
            model_name=deployment_result['model_name'],
            model_version=deployment_result['model_version'],
            endpoint=deployment_result['endpoints'][0] if deployment_result['endpoints'] else None
        )
        
        # Collect initial metrics
        metrics = await monitor.collect_metrics(
            deployment_result['model_name'],
            deployment_result['endpoints'][0] if deployment_result['endpoints'] else ""
        )
        
        return {
            'model_name': deployment_result['model_name'],
            'model_version': deployment_result['model_version'],
            'initial_metrics': metrics.to_dict() if metrics else {},
            'monitoring_status': 'active'
        }
    
    monitoring_result = asyncio.run(_monitor_deployment())
    
    # Store monitoring result
    context['task_instance'].xcom_push(key='monitoring_result', value=monitoring_result)
    
    return "monitoring_active"


def cleanup(**context):
    """Clean up temporary files and resources"""
    import shutil
    
    # Remove temporary training data
    data_path = context['task_instance'].xcom_pull(key='training_data_path')
    if data_path and os.path.exists(data_path):
        os.remove(data_path)
    
    # Clean up old models (keep last 5 versions)
    models_dir = Path("/app/ml_models")
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                versions = sorted(model_dir.iterdir(), key=lambda x: x.stat().st_mtime)
                if len(versions) > 5:
                    for old_version in versions[:-5]:
                        shutil.rmtree(old_version)
    
    return "cleanup_complete"


# Define tasks
task_init = PythonOperator(
    task_id='initialize_orchestrator',
    python_callable=initialize_orchestrator,
    dag=dag,
)

task_quality = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

task_drift = PythonOperator(
    task_id='check_drift',
    python_callable=check_drift,
    dag=dag,
)

task_prepare = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

task_evaluate = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag,
)

task_deploy = PythonOperator(
    task_id='deploy_best_model',
    python_callable=deploy_best_model,
    dag=dag,
)

task_ab_test = PythonOperator(
    task_id='run_ab_test',
    python_callable=run_ab_test,
    dag=dag,
)

task_monitor = PythonOperator(
    task_id='monitor_deployment',
    python_callable=monitor_deployment,
    dag=dag,
)

task_cleanup = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup,
    trigger_rule='all_done',  # Run even if some tasks fail
    dag=dag,
)

# Define task dependencies
task_init >> [task_quality, task_drift]
task_quality >> task_prepare
task_drift >> task_prepare
task_prepare >> task_train
task_train >> task_evaluate
task_evaluate >> task_deploy
task_deploy >> [task_ab_test, task_monitor]
[task_ab_test, task_monitor] >> task_cleanup