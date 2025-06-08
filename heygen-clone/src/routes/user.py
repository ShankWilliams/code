import os
import secrets
import structlog
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from werkzeug.security import generate_password_hash, check_password_hash
from src.models.database import db, User, Project, AIGeneration, Asset
from src.config.pricing import get_plan_features
from src.utils.email_service import send_email
from sqlalchemy import func
import zxcvbn
from src.utils.aws_s3 import get_s3_client
from werkzeug.utils import secure_filename
from io import BytesIO
import uuid

user_bp = Blueprint('user', __name__)
logger = structlog.get_logger()


@user_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user or user.deleted_at:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        return jsonify({'user': user.to_dict()}), 200
    except Exception as e:
        logger.error("Failed to get user profile", exc_info=True)
        return jsonify({'error_code': 'GET_PROFILE_FAILED', 'message': 'Failed to get profile', 'details': str(e)}), 500

@user_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user or user.deleted_at:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        data = request.get_json()
        if 'first_name' in data:
            first_name = data['first_name'].strip()
            if not first_name or len(first_name) > 50:
                return jsonify({'error_code': 'INVALID_FIRST_NAME', 'message': 'First name is required and max 50 chars'}), 400
            user.first_name = first_name
        if 'last_name' in data:
            last_name = data['last_name'].strip()
            if not last_name or len(last_name) > 50:
                return jsonify({'error_code': 'INVALID_LAST_NAME', 'message': 'Last name is required and max 50 chars'}), 400
            user.last_name = last_name
        db.session.commit()
        return jsonify({'message': 'Profile updated successfully', 'user': user.to_dict()}), 200
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to update user profile", exc_info=True)
        return jsonify({'error_code': 'UPDATE_PROFILE_FAILED', 'message': 'Failed to update profile', 'details': str(e)}), 500

@user_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user or user.deleted_at:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        data = request.get_json()
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')
        if not current_password or not new_password or not confirm_password:
            return jsonify({'error_code': 'MISSING_PASSWORDS', 'message': 'Current password, new password, and confirmation are required'}), 400
        if new_password != confirm_password:
            return jsonify({'error_code': 'PASSWORD_MISMATCH', 'message': 'New password and confirmation do not match'}), 400
        if not check_password_hash(user.password_hash, current_password):
            return jsonify({'error_code': 'INCORRECT_CURRENT_PASSWORD', 'message': 'Current password is incorrect'}), 400
        password_strength = zxcvbn.zxcvbn(new_password)
        if password_strength['score'] < 3:
            return jsonify({'error_code': 'WEAK_NEW_PASSWORD', 'message': 'New password is too weak. Please use a stronger password.'}), 400
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        return jsonify({'message': 'Password changed successfully'}), 200
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to change password", exc_info=True)
        return jsonify({'error_code': 'CHANGE_PASSWORD_FAILED', 'message': 'Failed to change password', 'details': str(e)}), 500

@user_bp.route('/usage', methods=['GET'])
@jwt_required()
def get_usage():
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user or user.deleted_at:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        current_period_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        current_period_end = (current_period_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        if user.subscription_status == 'active' and user.subscription:
            current_period_start = user.subscription.current_period_start
            current_period_end = user.subscription.current_period_end
        video_minutes_used = db.session.query(func.sum(Project.duration)).filter(
            Project.user_id == current_user_id,
            Project.status == 'completed',
            Project.updated_at >= current_period_start,
            Project.updated_at <= current_period_end
        ).scalar() or 0
        video_minutes_used = round(video_minutes_used / 60, 2)
        ai_generations_count = AIGeneration.query.filter(
            AIGeneration.user_id == current_user_id,
            AIGeneration.status == 'completed',
            AIGeneration.created_at >= current_period_start,
            AIGeneration.created_at <= current_period_end
        ).count()
        plan_details = get_plan_features(user.subscription_tier)
        return jsonify({'current_period': {'start': current_period_start.isoformat(), 'end': current_period_end.isoformat()}, 'usage': {'video_minutes_used': video_minutes_used, 'ai_generations_count': ai_generations_count}, 'limits': plan_details['features'], 'subscription_tier': user.subscription_tier}), 200
    except Exception as e:
        logger.error("Failed to get user usage", exc_info=True)
        return jsonify({'error_code': 'GET_USAGE_FAILED', 'message': 'Failed to get usage', 'details': str(e)}), 500

@user_bp.route('/delete-account', methods=['POST'])
@jwt_required()
def request_delete_account():
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user or user.deleted_at:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        data = request.get_json()
        password = data.get('password')
        if not password or not check_password_hash(user.password_hash, password):
            return jsonify({'error_code': 'INVALID_PASSWORD', 'message': 'Incorrect password'}), 401
        delete_token = secrets.token_urlsafe(32)
        user.reset_token = delete_token
        user.reset_token_expires = datetime.utcnow() + timedelta(minutes=15)
        db.session.commit()
        delete_confirm_link = f"{request.host_url}confirm-delete-account?token={delete_token}"
        email_subject = "Confirm Your Account Deletion"
        email_body = f"Hello {user.first_name},\n\nYou have requested to delete your HeyGen Clone account. This action is irreversible.\n\nTo confirm your account deletion, please click on the following link within 15 minutes: {delete_confirm_link}\n\nIf you did NOT request to delete your account, please ignore this email and secure your account.\n\nThanks,\nThe HeyGen Clone Team"
        if os.getenv('SENDGRID_API_KEY'):
            send_email(user.email, email_subject, email_body, current_app.config['FROM_EMAIL'])
        else:
            logger.warning(f"Email service not configured. Account deletion confirmation link for {user.email}: {delete_confirm_link}")
        return jsonify({'message': 'Account deletion confirmation email sent. Please check your inbox.'}), 200
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to request account deletion", exc_info=True)
        return jsonify({'error_code': 'REQUEST_DELETE_FAILED', 'message': 'Failed to request account deletion', 'details': str(e)}), 500

@user_bp.route('/confirm-delete-account', methods=['POST'])
def confirm_delete_account():
    try:
        data = request.get_json()
        token = data.get('token')
        if not token:
            return jsonify({'error_code': 'TOKEN_REQUIRED', 'message': 'Deletion token is required'}), 400
        user = User.query.filter_by(reset_token=token).first()
        if not user or not user.reset_token_expires or user.reset_token_expires < datetime.utcnow() or user.deleted_at:
            return jsonify({'error_code': 'INVALID_OR_EXPIRED_TOKEN', 'message': 'Invalid or expired deletion token'}), 400
        user.deleted_at = datetime.utcnow()
        user.reset_token = None
        user.reset_token_expires = None
        user.is_verified = False
        user.subscription_tier = 'free'
        user.subscription_status = 'cancelled'
        db.session.commit()
        if user.stripe_customer_id:
            import stripe
            try:
                subscriptions = stripe.Subscription.list(customer=user.stripe_customer_id, status='active')
                for sub in subscriptions.data:
                    stripe.Subscription.delete(sub.id)
                logger.info(f"Stripe subscriptions cancelled for customer {user.stripe_customer_id}")
            except stripe.error.StripeError as se:
                logger.error(f"Stripe error cancelling subscriptions for user {user.id}: {se}", exc_info=True)
            except Exception as e:
                logger.error(f"Error cancelling Stripe subscriptions for user {user.id}: {e}", exc_info=True)
        return jsonify({'message': 'Account soft-deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to confirm account deletion", exc_info=True)
        return jsonify({'error_code': 'CONFIRM_DELETE_FAILED', 'message': 'Failed to confirm account deletion', 'details': str(e)}), 500

@user_bp.route('/profile/avatar', methods=['POST'])
@jwt_required()
def upload_profile_avatar():
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user or user.deleted_at:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        if 'avatar' not in request.files:
            return jsonify({'error_code': 'NO_FILE_PART', 'message': 'No avatar file part in the request'}), 400
        file = request.files['avatar']
        if file.filename == '':
            return jsonify({'error_code': 'NO_SELECTED_FILE', 'message': 'No selected file'}), 400
        if file and file.content_type.startswith(('image/jpeg', 'image/png')):
            s3_client = get_s3_client()
            bucket_name = current_app.config.get('AWS_S3_BUCKET_NAME')
            if not bucket_name:
                return jsonify({'error_code': 'S3_BUCKET_NOT_CONFIGURED', 'message': 'S3 bucket name is not configured'}), 500
            filename = secure_filename(file.filename)
            s3_key = f"user_avatars/{current_user_id}/{uuid.uuid4().hex}_{filename}"
            file_stream = BytesIO(file.read())
            s3_client.upload_fileobj(file_stream, bucket_name, s3_key, ExtraArgs={'ContentType': file.content_type, 'ACL': 'public-read'})
            avatar_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
            user.profile_image_url = avatar_url
            db.session.commit()
            new_asset = Asset(user_id=current_user_id, asset_type='image', file_name=filename, file_url=avatar_url, file_size=file_stream.getbuffer().nbytes, mime_type=file.content_type)
            db.session.add(new_asset)
            db.session.commit()
            return jsonify({'message': 'Profile avatar uploaded successfully', 'avatar_url': avatar_url, 'user': user.to_dict()}), 200
        else:
            return jsonify({'error_code': 'INVALID_FILE_TYPE', 'message': 'Only JPEG and PNG images are allowed'}), 400
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to upload profile avatar", exc_info=True)
        return jsonify({'error_code': 'UPLOAD_AVATAR_FAILED', 'message': 'Failed to upload profile avatar', 'details': str(e)}), 500
