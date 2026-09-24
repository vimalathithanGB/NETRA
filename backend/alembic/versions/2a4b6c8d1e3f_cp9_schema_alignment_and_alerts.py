"""cp9_schema_alignment_and_alerts

Revision ID: 2a4b6c8d1e3f
Revises: 1c0e9fb651b7
Create Date: 2026-09-22 15:38:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
import geoalchemy2


# revision identifiers, used by Alembic.
revision: str = '2a4b6c8d1e3f'
down_revision: Union[str, Sequence[str], None] = '1c0e9fb651b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema for CP9 alignment."""
    # 1. Update cameras table
    op.add_column('cameras', sa.Column('road_name', sa.String(length=128), nullable=True))
    op.add_column('cameras', sa.Column('speed_limit_kmh', sa.Float(), nullable=True))

    # 2. Update vehicle_observations table
    op.add_column('vehicle_observations', sa.Column('last_bbox', sa.JSON(), nullable=True))
    op.add_column('vehicle_observations', sa.Column('detector_confidence', sa.Float(), nullable=True))

    # 3. Update trajectory_segments table
    op.add_column('trajectory_segments', sa.Column('match_score', sa.Float(), nullable=True))
    op.add_column('trajectory_segments', sa.Column('reid_similarity', sa.Float(), nullable=True))
    op.add_column('trajectory_segments', sa.Column('available_evidence', sa.JSON(), nullable=True))

    # 4. Create alerts table
    op.create_table(
        'alerts',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('alert_type', sa.String(length=64), nullable=False),
        sa.Column('severity', sa.String(length=32), nullable=False, server_default='Medium'),
        sa.Column('message', sa.String(length=512), nullable=True),
        sa.Column('camera_id', sa.String(length=64), nullable=True),
        sa.Column('global_vehicle_id', sa.String(length=64), nullable=True),
        sa.Column('plate_text', sa.String(length=32), nullable=True),
        sa.Column('location_text', sa.String(length=256), nullable=True),
        sa.Column('timestamp_text', sa.String(length=64), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='Pending Review'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['global_vehicle_id'], ['global_vehicles.global_vehicle_id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_alerts_plate_text'), 'alerts', ['plate_text'], unique=False)
    op.create_index(op.f('ix_alerts_status'), 'alerts', ['status'], unique=False)

    # 5. Create traffic_events table
    op.create_table(
        'traffic_events',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('title', sa.String(length=128), nullable=False),
        sa.Column('severity', sa.String(length=32), nullable=False, server_default='medium'),
        sa.Column('camera_id', sa.String(length=64), nullable=True),
        sa.Column('corridor', sa.String(length=128), nullable=True),
        sa.Column('location_geom', geoalchemy2.types.Geometry(geometry_type='POINT', srid=4326, from_text='ST_GeomFromEWKT', name='geometry'), nullable=True),
        sa.Column('latitude', sa.Float(), nullable=True),
        sa.Column('longitude', sa.Float(), nullable=True),
        sa.Column('description', sa.String(length=512), nullable=True),
        sa.Column('timestamp_text', sa.String(length=64), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='active'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema for CP9 alignment."""
    # 1. Drop traffic_events table
    op.drop_table('traffic_events')

    # 2. Drop alerts table
    op.drop_index(op.f('ix_alerts_status'), table_name='alerts')
    op.drop_index(op.f('ix_alerts_plate_text'), table_name='alerts')
    op.drop_table('alerts')

    # 3. Drop columns from trajectory_segments
    op.drop_column('trajectory_segments', 'available_evidence')
    op.drop_column('trajectory_segments', 'reid_similarity')
    op.drop_column('trajectory_segments', 'match_score')

    # 4. Drop columns from vehicle_observations
    op.drop_column('vehicle_observations', 'detector_confidence')
    op.drop_column('vehicle_observations', 'last_bbox')

    # 5. Drop columns from cameras
    op.drop_column('cameras', 'speed_limit_kmh')
    op.drop_column('cameras', 'road_name')
