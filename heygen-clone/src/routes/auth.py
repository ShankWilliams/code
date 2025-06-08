import os
import secrets
import re
import redis
import zxcvbn
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity, get_jwt
from werkzeug.security import generate_password_hash, check_password_hash
from src.models.database import db, User
from src.utils.email_service import send_email
from src.extensions import limiter

auth_bp = Blueprint('auth', __name__)

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

@auth_bp.route('/register', methods=['POST'])
@limiter.limit("5 per minute")
def register():
    try:
        data = request.get_json()
        required_fields = ['email', 'password', 'first_name', 'last_name']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({'error_code': 'MISSING_FIELDS', 'message': f'{", ".join(missing_fields)} are required'}), 400
        email = data['email'].lower().strip()
        password = data['password']
        first_name = data['first_name'].strip()
        last_name = data['last_name'].strip()
        if not EMAIL_REGEX.match(email):
            return jsonify({'error_code': 'INVALID_EMAIL_FORMAT', 'message': 'Invalid email format'}), 400
        password_strength = zxcvbn.zxcvbn(password)
        if password_strength['score'] < 3:
            return jsonify({'error_code': 'WEAK_PASSWORD', 'message': 'Password is too weak. Please use a stronger password with a mix of characters, numbers, and symbols.'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error_code': 'USER_EXISTS', 'message': 'User with this email already exists'}), 409
        password_hash = generate_password_hash(password)
        verification_token = secrets.token_urlsafe(32)
        user = User(
            email=email,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            is_verified=False,
            verification_token=verification_token
        )
        db.session.add(user)
        db.session.commit()
        verification_link = f"{request.host_url}verify-email?token={verification_token}"
        email_subject = "Verify Your HeyGen Clone Account"
        email_body = f"Hello {first_name},\n\nPlease verify your email address by clicking on the following link: {verification_link}\n\nIf you did not register for this service, please ignore this email.\n\nThanks,\nThe HeyGen Clone Team"
        if os.getenv('SENDGRID_API_KEY'):
            send_email(email, email_subject, email_body, current_app.config['FROM_EMAIL'])
        else:
            current_app.logger.warning(f"Email service not configured. Verification link for {email}: {verification_link}")
        return jsonify({'message': 'User registered successfully. Please check your email for verification.', 'user_id': user.id, 'verification_required': True}), 201
    except Exception as e:
        db.session.rollback()
        current_app.logger.error("Registration failed", exc_info=True)
        return jsonify({'error_code': 'REGISTRATION_FAILED', 'message': 'Registration failed', 'details': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    try:
        data = request.get_json()
        if not data.get('email') or not data.get('password'):
            return jsonify({'error_code': 'MISSING_CREDENTIALS', 'message': 'Email and password are required'}), 400
        email = data['email'].lower().strip()
        password = data['password']
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error_code': 'INVALID_CREDENTIALS', 'message': 'Invalid email or password'}), 401
        if not user.is_verified:
            return jsonify({'error_code': 'EMAIL_NOT_VERIFIED', 'message': 'Please verify your email before logging in'}), 401
        user.last_login_at = datetime.utcnow()
        db.session.commit()
        access_token = create_access_token(identity=str(user.id))
        refresh_token = create_refresh_token(identity=str(user.id))
        return jsonify({'message': 'Login successful', 'access_token': access_token, 'refresh_token': refresh_token, 'user': user.to_dict()}), 200
    except Exception as e:
        current_app.logger.error("Login failed", exc_info=True)
        return jsonify({'error_code': 'LOGIN_FAILED', 'message': 'Login failed', 'details': str(e)}), 500

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    try:
        current_user_id = get_jwt_identity()
        jti = get_jwt()['jti']
        redis_client = current_app.extensions['redis_client']
        if redis_client.get(f"revoked_token:{jti}") is not None:
            return jsonify({'error_code': 'TOKEN_REVOKED', 'message': 'Refresh token has been revoked'}), 401
        new_access_token = create_access_token(identity=current_user_id)
        return jsonify({'access_token': new_access_token}), 200
    except Exception as e:
        current_app.logger.error("Token refresh failed", exc_info=True)
        return jsonify({'error_code': 'TOKEN_REFRESH_FAILED', 'message': 'Token refresh failed', 'details': str(e)}), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    try:
        jti = get_jwt()['jti']
        expires = get_jwt()["exp"]
        now = datetime.utcnow()
        redis_client = current_app.extensions['redis_client']
        redis_client.setex(f"revoked_token:{jti}", expires - now.timestamp(), "true")
        return jsonify({'message': 'Successfully logged out'}), 200
    except Exception as e:
        current_app.logger.error("Logout failed", exc_info=True)
        return jsonify({'error_code': 'LOGOUT_FAILED', 'message': 'Logout failed', 'details': str(e)}), 500

@auth_bp.route('/verify-email', methods=['GET', 'POST'])
def verify_email():
    try:
        token = request.args.get('token') if request.method == 'GET' else request.get_json().get('token')
        if not token:
            return jsonify({'error_code': 'TOKEN_REQUIRED', 'message': 'Verification token is required'}), 400
        user = User.query.filter_by(verification_token=token).first()
        if not user:
            return jsonify({'error_code': 'INVALID_TOKEN', 'message': 'Invalid verification token'}), 400
        user.is_verified = True
        user.verification_token = None
        db.session.commit()
        return jsonify({'message': 'Email verified successfully'}), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.error("Email verification failed", exc_info=True)
        return jsonify({'error_code': 'EMAIL_VERIFICATION_FAILED', 'message': 'Email verification failed', 'details': str(e)}), 500

@auth_bp.route('/forgot-password', methods=['POST'])
@limiter.limit("5 per minute")
def forgot_password():
    try:
        data = request.get_json()
        email = data.get('email')
        if not email:
            return jsonify({'error_code': 'EMAIL_REQUIRED', 'message': 'Email is required'}), 400
        email = email.lower().strip()
        user = User.query.filter_by(email=email).first()
        if user:
            reset_token = secrets.token_urlsafe(32)
            user.reset_token = reset_token
            user.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
            db.session.commit()
            reset_link = f"{request.host_url}reset-password?token={reset_token}"
            email_subject = "HeyGen Clone Password Reset"
            email_body = f"Hello {user.first_name},\n\nYou have requested a password reset. Please click on the following link to reset your password: {reset_link}\n\nThis link will expire in 1 hour. If you did not request a password reset, please ignore this email.\n\nThanks,\nThe HeyGen Clone Team"
            if os.getenv('SENDGRID_API_KEY'):
                send_email(user.email, email_subject, email_body, current_app.config['FROM_EMAIL'])
            else:
                current_app.logger.warning(f"Email service not configured. Reset link for {user.email}: {reset_link}")
        return jsonify({'message': 'If an account with that email exists, a password reset link has been sent.'}), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.error("Forgot password failed", exc_info=True)
        return jsonify({'error_code': 'FORGOT_PASSWORD_FAILED', 'message': 'Password reset request failed', 'details': str(e)}), 500

@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        data = request.get_json()
        token = data.get('token')
        new_password = data.get('password')
        if not token or not new_password:
            return jsonify({'error_code': 'TOKEN_AND_PASSWORD_REQUIRED', 'message': 'Token and new password are required'}), 400
        password_strength = zxcvbn.zxcvbn(new_password)
        if password_strength['score'] < 3:
            return jsonify({'error_code': 'WEAK_PASSWORD', 'message': 'New password is too weak.'}), 400
        user = User.query.filter_by(reset_token=token).first()
        if not user or not user.reset_token_expires or user.reset_token_expires < datetime.utcnow():
            return jsonify({'error_code': 'INVALID_OR_EXPIRED_TOKEN', 'message': 'Invalid or expired reset token'}), 400
        user.password_hash = generate_password_hash(new_password)
        user.reset_token = None
        user.reset_token_expires = None
        db.session.commit()
        return jsonify({'message': 'Password reset successfully'}), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.error("Password reset failed", exc_info=True)
        return jsonify({'error_code': 'PASSWORD_RESET_FAILED', 'message': 'Password reset failed', 'details': str(e)}), 500

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    try:
        current_user_id = int(get_jwt_identity())
        user = User.query.get(current_user_id)
        if not user or user.deleted_at:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        return jsonify({'user': user.to_dict()}), 200
    except Exception as e:
        current_app.logger.error("Failed to get user info", exc_info=True)
        return jsonify({'error_code': 'GET_USER_INFO_FAILED', 'message': 'Failed to get user info', 'details': str(e)}), 500
