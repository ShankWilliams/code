from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import Numeric

db = SQLAlchemy()

class Asset(db.Model):
    __tablename__ = 'assets'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=True)
    asset_type = db.Column(db.String(50), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_url = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, nullable=True)
    mime_type = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'project_id': self.project_id,
            'asset_type': self.asset_type,
            'file_name': self.file_name,
            'file_url': self.file_url,
            'file_size': self.file_size,
            'mime_type': self.mime_type,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    is_verified = db.Column(db.Boolean, default=False)
    verification_token = db.Column(db.String(255), nullable=True, index=True)
    reset_token = db.Column(db.String(255), nullable=True, index=True)
    reset_token_expires = db.Column(db.DateTime, nullable=True)
    subscription_tier = db.Column(db.String(20), default='free')
    subscription_status = db.Column(db.String(20), default='active')
    stripe_customer_id = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = db.Column(db.DateTime, nullable=True)
    profile_image_url = db.Column(db.String(500), nullable=True)
    deleted_at = db.Column(db.DateTime, nullable=True, index=True)

    projects = db.relationship('Project', backref='user', lazy=True, cascade='all, delete-orphan')
    ai_generations = db.relationship('AIGeneration', backref='user', lazy=True, cascade='all, delete-orphan')
    assets = db.relationship('Asset', backref='user', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f"<User {self.email}>"

    def to_dict(self):
        data = {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_verified': self.is_verified,
            'subscription_tier': self.subscription_tier,
            'subscription_status': self.subscription_status,
            'profile_image_url': self.profile_image_url,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None,
            'deleted_at': self.deleted_at.isoformat() if self.deleted_at else None
        }
        return data

class Project(db.Model):
    __tablename__ = 'projects'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    script = db.Column(db.Text, nullable=True)
    avatar_type = db.Column(db.String(50), nullable=True)
    avatar_data = db.Column(db.JSON, nullable=True)
    voice_settings = db.Column(db.JSON, nullable=True)
    background_settings = db.Column(db.JSON, nullable=True)
    status = db.Column(db.String(20), default='draft')
    output_video_url = db.Column(db.String(500), nullable=True)
    thumbnail_url = db.Column(db.String(500), nullable=True)
    duration = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    ai_generations = db.relationship('AIGeneration', backref='project', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'description': self.description,
            'script': self.script,
            'avatar_type': self.avatar_type,
            'avatar_data': self.avatar_data,
            'voice_settings': self.voice_settings,
            'background_settings': self.background_settings,
            'status': self.status,
            'output_video_url': self.output_video_url,
            'thumbnail_url': self.thumbnail_url,
            'duration': self.duration,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class AIGeneration(db.Model):
    __tablename__ = 'ai_generations'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=True)
    generation_type = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.JSON, nullable=False)
    output_data = db.Column(db.JSON, nullable=True)
    status = db.Column(db.String(20), default='pending')
    error_message = db.Column(db.Text, nullable=True)
    processing_time = db.Column(db.Float, nullable=True)
    cost = db.Column(Numeric(10, 4), nullable=True)
    provider = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    processed_by = db.Column(db.String(100), nullable=True)
    input_data_length = db.Column(db.Integer, nullable=True)
    output_size = db.Column(db.Integer, nullable=True)

    def to_dict(self):
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'project_id': self.project_id,
            'generation_type': self.generation_type,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'status': self.status,
            'error_message': self.error_message,
            'processing_time': self.processing_time,
            'cost': float(self.cost) if self.cost else None,
            'provider': self.provider,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'input_data_length': self.input_data_length,
            'output_size': self.output_size
        }
        return data

class Subscription(db.Model):
    __tablename__ = 'subscriptions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    stripe_subscription_id = db.Column(db.String(255), unique=True, nullable=False, index=True)
    plan_id = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    current_period_start = db.Column(db.DateTime, nullable=False)
    current_period_end = db.Column(db.DateTime, nullable=False)
    cancel_at_period_end = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship('User', backref='subscription', uselist=False)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'stripe_subscription_id': self.stripe_subscription_id,
            'plan_id': self.plan_id,
            'status': self.status,
            'current_period_start': self.current_period_start.isoformat() if self.current_period_start else None,
            'current_period_end': self.current_period_end.isoformat() if self.current_period_end else None,
            'cancel_at_period_end': self.cancel_at_period_end,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
