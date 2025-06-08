import os
from datetime import datetime
from src.models.database import Project, AIGeneration
from sqlalchemy import func

# Pricing plans configuration
PRICING_PLANS = {
    'free': {
        'name': 'Free',
        'price': 0,
        'stripe_price_id': None,
        'features': {
            'video_minutes_per_month': 1,
            'avatars': 'basic',
            'voices': 'standard',
            'video_quality': '720p',
            'projects_limit': 3,
            'storage_gb': 1
        },
        'limits': {
            'ai_generations_per_day': 5,
            'concurrent_generations': 1
        }
    },
    'pro': {
        'name': 'Pro',
        'price': 29,
        'stripe_price_id': os.getenv('STRIPE_PRO_PRICE_ID'),
        'features': {
            'video_minutes_per_month': 15,
            'avatars': 'premium',
            'voices': 'premium_and_cloning',
            'video_quality': '1080p',
            'projects_limit': 50,
            'storage_gb': 10,
            'priority_support': True,
            'custom_avatars': True
        },
        'limits': {
            'ai_generations_per_day': 100,
            'concurrent_generations': 3
        }
    },
    'enterprise': {
        'name': 'Enterprise',
        'price': 'custom',
        'stripe_price_id': None,
        'features': {
            'video_minutes_per_month': 'unlimited',
            'avatars': 'custom',
            'voices': 'unlimited_cloning',
            'video_quality': '4k',
            'projects_limit': 'unlimited',
            'storage_gb': 'unlimited',
            'api_access': True,
            'dedicated_support': True,
            'white_label': True,
            'sso': True
        },
        'limits': {
            'ai_generations_per_day': 'unlimited',
            'concurrent_generations': 10
        }
    }
}

def get_plan_features(plan_id):
    """Get features for a specific plan"""
    return PRICING_PLANS.get(plan_id, PRICING_PLANS['free'])

def validate_plan_limits(user, action_type):
    """Validate if user can perform an action based on their plan limits."""
    plan = get_plan_features(user.subscription_tier)
    limits = plan.get('limits', {})
    if action_type == 'project_create':
        limit = plan['features'].get('projects_limit')
        if limit != 'unlimited':
            count = Project.query.filter_by(user_id=user.id).count()
            return count < limit
    if action_type == 'ai_generation':
        daily = limits.get('ai_generations_per_day')
        if daily != 'unlimited':
            start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            count = AIGeneration.query.filter(AIGeneration.user_id==user.id, AIGeneration.created_at>=start).count()
            return count < daily
    if action_type == 'video_minutes':
        limit_minutes = plan['features'].get('video_minutes_per_month')
        if limit_minutes != 'unlimited':
            period_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            total_seconds = db.session.query(func.coalesce(func.sum(Project.duration), 0)).filter(
                Project.user_id == user.id,
                Project.status == 'completed',
                Project.updated_at >= period_start
            ).scalar() or 0
            used_minutes = total_seconds / 60
            return used_minutes < limit_minutes
    return True
