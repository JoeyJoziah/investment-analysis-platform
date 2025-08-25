"""Create ML operations tables

Revision ID: 005
Revises: 004
Create Date: 2025-01-19
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None

def upgrade():
    """Create ML operations tables"""
    
    # Create ENUMs first
    op.execute("CREATE TYPE IF NOT EXISTS model_type_enum AS ENUM ('classification', 'regression', 'time_series', 'ensemble', 'reinforcement', 'clustering')")
    op.execute("CREATE TYPE IF NOT EXISTS model_stage_enum AS ENUM ('development', 'staging', 'production', 'archived')")
    op.execute("CREATE TYPE IF NOT EXISTS feature_type_enum AS ENUM ('numerical', 'categorical', 'text', 'embedding', 'time_series', 'derived')")
    op.execute("CREATE TYPE IF NOT EXISTS compute_mode_enum AS ENUM ('batch', 'streaming', 'real_time', 'on_demand')")
    op.execute("CREATE TYPE IF NOT EXISTS feature_status_enum AS ENUM ('development', 'testing', 'production', 'deprecated')")
    op.execute("CREATE TYPE IF NOT EXISTS drift_type_enum AS ENUM ('data_drift', 'concept_drift', 'prediction_drift')")
    op.execute("CREATE TYPE IF NOT EXISTS alert_severity_enum AS ENUM ('low', 'medium', 'high', 'critical')")
    op.execute("CREATE TYPE IF NOT EXISTS model_health_enum AS ENUM ('healthy', 'degraded', 'critical', 'failed')")
    
    # Update ml_models table if it exists, or create it
    op.execute("""
        ALTER TABLE IF EXISTS ml_models 
        ADD COLUMN IF NOT EXISTS name VARCHAR(100),
        ADD COLUMN IF NOT EXISTS model_type model_type_enum,
        ADD COLUMN IF NOT EXISTS stage model_stage_enum DEFAULT 'development',
        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ADD COLUMN IF NOT EXISTS created_by VARCHAR(100),
        ADD COLUMN IF NOT EXISTS description TEXT,
        ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]'::jsonb,
        ADD COLUMN IF NOT EXISTS parameters JSONB DEFAULT '{}'::jsonb,
        ADD COLUMN IF NOT EXISTS model_size_bytes BIGINT,
        ADD COLUMN IF NOT EXISTS training_data_hash VARCHAR(64),
        ADD COLUMN IF NOT EXISTS feature_names JSONB DEFAULT '[]'::jsonb,
        ADD COLUMN IF NOT EXISTS metadata_path TEXT,
        ADD COLUMN IF NOT EXISTS performance_benchmark JSONB DEFAULT '{}'::jsonb,
        ADD COLUMN IF NOT EXISTS dependencies JSONB DEFAULT '{}'::jsonb,
        ADD COLUMN IF NOT EXISTS git_commit VARCHAR(40),
        ADD COLUMN IF NOT EXISTS parent_version VARCHAR(20),
        ADD COLUMN IF NOT EXISTS is_champion BOOLEAN DEFAULT FALSE
    """)
    
    # Create ModelPerformanceMetric table
    op.create_table('model_performance_metrics',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_id', sa.Integer(), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('metric_date', sa.Date(), nullable=False),
        sa.Column('accuracy', sa.Float()),
        sa.Column('precision', sa.Float()),
        sa.Column('recall', sa.Float()),
        sa.Column('f1_score', sa.Float()),
        sa.Column('auc_roc', sa.Float()),
        sa.Column('rmse', sa.Float()),
        sa.Column('mae', sa.Float()),
        sa.Column('mape', sa.Float()),
        sa.Column('sharpe_ratio', sa.Float()),
        sa.Column('max_drawdown', sa.Float()),
        sa.Column('win_rate', sa.Float()),
        sa.Column('profit_factor', sa.Float()),
        sa.Column('predictions_count', sa.Integer()),
        sa.Column('inference_time_ms', sa.Float()),
        sa.Column('custom_metrics', sa.JSON()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('idx_model_perf_model_date', 'model_performance_metrics', ['model_id', 'metric_date'])
    
    # Create ModelDriftDetection table
    op.create_table('model_drift_detection',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_id', sa.Integer(), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('detection_date', sa.DateTime(), nullable=False),
        sa.Column('drift_type', postgresql.ENUM('data_drift', 'concept_drift', 'prediction_drift', name='drift_type_enum', create_type=False), nullable=False),
        sa.Column('drift_score', sa.Float(), nullable=False),
        sa.Column('p_value', sa.Float()),
        sa.Column('feature_drifts', sa.JSON()),
        sa.Column('affected_features', sa.JSON()),
        sa.Column('baseline_window_start', sa.Date()),
        sa.Column('baseline_window_end', sa.Date()),
        sa.Column('detection_window_start', sa.Date()),
        sa.Column('detection_window_end', sa.Date()),
        sa.Column('is_significant', sa.Boolean(), default=False),
        sa.Column('action_taken', sa.String(50)),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('idx_drift_model_date', 'model_drift_detection', ['model_id', 'detection_date'])
    
    # Create ModelAlert table
    op.create_table('model_alerts',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_id', sa.Integer(), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('alert_type', sa.String(50), nullable=False),
        sa.Column('severity', postgresql.ENUM('low', 'medium', 'high', 'critical', name='alert_severity_enum', create_type=False), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('details', sa.JSON()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('resolved_at', sa.DateTime()),
        sa.Column('resolved_by', sa.String(100)),
        sa.Column('resolution_notes', sa.Text()),
        sa.Column('alert_count', sa.Integer(), default=1),
        sa.Column('is_active', sa.Boolean(), default=True),
    )
    op.create_index('idx_alert_model_active', 'model_alerts', ['model_id', 'is_active'])
    
    # Create ModelBacktest table
    op.create_table('model_backtests',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_id', sa.Integer(), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('test_name', sa.String(100), nullable=False),
        sa.Column('start_date', sa.Date(), nullable=False),
        sa.Column('end_date', sa.Date(), nullable=False),
        sa.Column('initial_capital', sa.Numeric(12, 2)),
        sa.Column('final_capital', sa.Numeric(12, 2)),
        sa.Column('total_return', sa.Float()),
        sa.Column('annualized_return', sa.Float()),
        sa.Column('volatility', sa.Float()),
        sa.Column('sharpe_ratio', sa.Float()),
        sa.Column('max_drawdown', sa.Float()),
        sa.Column('win_rate', sa.Float()),
        sa.Column('total_trades', sa.Integer()),
        sa.Column('winning_trades', sa.Integer()),
        sa.Column('losing_trades', sa.Integer()),
        sa.Column('avg_win', sa.Float()),
        sa.Column('avg_loss', sa.Float()),
        sa.Column('profit_factor', sa.Float()),
        sa.Column('results', sa.JSON()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('idx_backtest_model', 'model_backtests', ['model_id'])
    
    # Create FeatureDefinition table
    op.create_table('feature_definitions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(100), unique=True, nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('feature_type', postgresql.ENUM('numerical', 'categorical', 'text', 'embedding', 'time_series', 'derived', name='feature_type_enum', create_type=False), nullable=False),
        sa.Column('compute_mode', postgresql.ENUM('batch', 'streaming', 'real_time', 'on_demand', name='compute_mode_enum', create_type=False), nullable=False),
        sa.Column('status', postgresql.ENUM('development', 'testing', 'production', 'deprecated', name='feature_status_enum', create_type=False), default='development', nullable=False),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('created_by', sa.String(100), nullable=False),
        sa.Column('dependencies', sa.JSON()),
        sa.Column('source_tables', sa.JSON()),
        sa.Column('computation_logic', sa.Text()),
        sa.Column('validation_rules', sa.JSON()),
        sa.Column('tags', sa.JSON()),
        sa.Column('business_context', sa.Text()),
        sa.Column('sla_hours', sa.Float()),
        sa.Column('monitoring_config', sa.JSON()),
    )
    
    # Create FeatureValue table
    op.create_table('feature_values',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('feature_id', sa.Integer(), sa.ForeignKey('feature_definitions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('entity_id', sa.String(100), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('value', sa.JSON(), nullable=False),
        sa.Column('quality_score', sa.Float()),
        sa.Column('is_imputed', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('idx_feature_value_lookup', 'feature_values', ['feature_id', 'entity_id', 'timestamp'])
    
    # Create FeatureDriftMetric table
    op.create_table('feature_drift_metrics',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('feature_id', sa.Integer(), sa.ForeignKey('feature_definitions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('metric_date', sa.Date(), nullable=False),
        sa.Column('mean_shift', sa.Float()),
        sa.Column('std_shift', sa.Float()),
        sa.Column('distribution_distance', sa.Float()),
        sa.Column('missing_rate', sa.Float()),
        sa.Column('cardinality_change', sa.Float()),
        sa.Column('drift_detected', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    
    # Create OnlineLearningMetric table
    op.create_table('online_learning_metrics',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_id', sa.Integer(), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('update_timestamp', sa.DateTime(), nullable=False),
        sa.Column('samples_processed', sa.Integer(), nullable=False),
        sa.Column('learning_rate', sa.Float()),
        sa.Column('loss_before', sa.Float()),
        sa.Column('loss_after', sa.Float()),
        sa.Column('weights_updated', sa.Boolean(), default=True),
        sa.Column('update_duration_ms', sa.Float()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    
    # Create InferenceMetric table
    op.create_table('inference_metrics',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_id', sa.Integer(), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('request_count', sa.Integer(), nullable=False),
        sa.Column('avg_latency_ms', sa.Float()),
        sa.Column('p50_latency_ms', sa.Float()),
        sa.Column('p95_latency_ms', sa.Float()),
        sa.Column('p99_latency_ms', sa.Float()),
        sa.Column('error_count', sa.Integer(), default=0),
        sa.Column('error_rate', sa.Float()),
        sa.Column('throughput', sa.Float()),
        sa.Column('cpu_usage', sa.Float()),
        sa.Column('memory_usage_mb', sa.Float()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('idx_inference_model_time', 'inference_metrics', ['model_id', 'timestamp'])
    
    # Create EnsembleWeight table
    op.create_table('ensemble_weights',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('ensemble_id', sa.Integer(), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('component_model_id', sa.Integer(), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('weight', sa.Float(), nullable=False),
        sa.Column('performance_contribution', sa.Float()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
    )
    
    # Create ABTest table
    op.create_table('ab_tests',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('test_name', sa.String(100), nullable=False),
        sa.Column('control_model_id', sa.Integer(), sa.ForeignKey('ml_models.id'), nullable=False),
        sa.Column('treatment_model_id', sa.Integer(), sa.ForeignKey('ml_models.id'), nullable=False),
        sa.Column('start_date', sa.DateTime(), nullable=False),
        sa.Column('end_date', sa.DateTime()),
        sa.Column('traffic_split', sa.Float(), default=0.5),
        sa.Column('control_samples', sa.Integer(), default=0),
        sa.Column('treatment_samples', sa.Integer(), default=0),
        sa.Column('control_metric', sa.Float()),
        sa.Column('treatment_metric', sa.Float()),
        sa.Column('p_value', sa.Float()),
        sa.Column('is_significant', sa.Boolean()),
        sa.Column('winner', sa.String(20)),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    
    # Create ModelArtifact table
    op.create_table('model_artifacts',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_id', sa.Integer(), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('artifact_type', sa.String(50), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('file_size_bytes', sa.BigInteger()),
        sa.Column('checksum', sa.String(64)),
        sa.Column('compression_type', sa.String(20)),
        sa.Column('is_encrypted', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    
    print("✓ ML operations tables created successfully")

def downgrade():
    """Drop ML operations tables"""
    
    # Drop tables in reverse order
    op.drop_table('model_artifacts')
    op.drop_table('ab_tests')
    op.drop_table('ensemble_weights')
    op.drop_table('inference_metrics')
    op.drop_table('online_learning_metrics')
    op.drop_table('feature_drift_metrics')
    op.drop_table('feature_values')
    op.drop_table('feature_definitions')
    op.drop_table('model_backtests')
    op.drop_table('model_alerts')
    op.drop_table('model_drift_detection')
    op.drop_table('model_performance_metrics')
    
    # Drop ENUMs
    op.execute("DROP TYPE IF EXISTS model_type_enum CASCADE")
    op.execute("DROP TYPE IF EXISTS model_stage_enum CASCADE")
    op.execute("DROP TYPE IF EXISTS feature_type_enum CASCADE")
    op.execute("DROP TYPE IF EXISTS compute_mode_enum CASCADE")
    op.execute("DROP TYPE IF EXISTS feature_status_enum CASCADE")
    op.execute("DROP TYPE IF EXISTS drift_type_enum CASCADE")
    op.execute("DROP TYPE IF EXISTS alert_severity_enum CASCADE")
    op.execute("DROP TYPE IF EXISTS model_health_enum CASCADE")