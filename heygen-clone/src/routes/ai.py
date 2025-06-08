import os
import requests
import openai
import json
import uuid
import time
import random
import structlog
import redis
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from src.models.database import db, AIGeneration, User, Asset
from src.config.pricing import validate_plan_limits
from src.utils.aws_s3 import get_s3_client
from io import BytesIO

ai_bp = Blueprint('ai', __name__)
logger = structlog.get_logger()
ELEVENLABS_BASE_URL = 'https://api.elevenlabs.io/v1'

def upload_bytes_to_s3(data_bytes, s3_key, content_type):
    s3_client = get_s3_client()
    bucket_name = current_app.config.get('AWS_S3_BUCKET_NAME')
    if not bucket_name:
        raise Exception("AWS S3 bucket name not configured.")
    file_stream = BytesIO(data_bytes)
    s3_client.upload_fileobj(file_stream, bucket_name, s3_key, ExtraArgs={'ContentType': content_type, 'ACL': 'public-read'})
    return f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

def api_call_with_retry(api_func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return api_func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"API call failed after {max_retries} attempts: {str(e)}")
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("API call failed, retrying...", attempt=attempt+1, wait_time=wait_time, error=str(e))
            time.sleep(wait_time)
        except openai.OpenAIError as e:
            if attempt == max_retries - 1:
                raise Exception(f"OpenAI API call failed after {max_retries} attempts: {str(e)}")
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("OpenAI API call failed, retrying...", attempt=attempt+1, wait_time=wait_time, error=str(e))
            time.sleep(wait_time)

@ai_bp.route('/text-to-speech', methods=['POST'])
@jwt_required()
def text_to_speech():
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        if not validate_plan_limits(user, 'ai_generation'):
            return jsonify({'error_code': 'GENERATION_LIMIT', 'message': 'AI generation limit reached for your plan.'}), 403
        text = data.get('text')
        voice_id = data.get('voice_id', 'alloy')
        provider = data.get('provider', 'openai')
        if not text:
            return jsonify({'error_code': 'TEXT_REQUIRED', 'message': 'Text is required'}), 400
        ai_generation = AIGeneration(user_id=current_user_id, generation_type='tts', input_data={'text': text, 'voice_id': voice_id, 'provider': provider}, status='processing', provider=provider, input_data_length=len(text))
        db.session.add(ai_generation)
        db.session.commit()
        audio_bytes = None
        error_message = None
        audio_url = None
        start_time = datetime.utcnow()
        try:
            if provider == 'elevenlabs' and current_app.config.get('ELEVENLABS_API_KEY'):
                audio_bytes = generate_elevenlabs_speech_bytes(text, voice_id)
            else:
                audio_bytes = generate_openai_speech_bytes(text, voice_id)
            s3_key = f"tts_audio/{current_user_id}/{uuid.uuid4().hex}.mp3"
            audio_url = upload_bytes_to_s3(audio_bytes, s3_key, 'audio/mpeg')
            new_asset = Asset(user_id=current_user_id, project_id=None, asset_type='audio', file_name=f"tts_{ai_generation.id}.mp3", file_url=audio_url, file_size=len(audio_bytes), mime_type='audio/mpeg')
            db.session.add(new_asset)
            db.session.commit()
            ai_generation.status = 'completed'
            ai_generation.output_data = {'audio_url': audio_url}
            ai_generation.completed_at = datetime.utcnow()
            ai_generation.processing_time = (datetime.utcnow() - start_time).total_seconds()
            ai_generation.output_size = len(audio_bytes)
            ai_generation.cost = (len(text) / 1000) * 0.015 if provider == 'openai' else 0
        except Exception as e:
            error_message = str(e)
            ai_generation.status = 'failed'
            ai_generation.error_message = error_message
            logger.error("Text-to-speech generation failed", error=error_message, exc_info=True, generation_id=ai_generation.id)
        db.session.commit()
        return jsonify({'generation_id': ai_generation.id, 'status': ai_generation.status, 'audio_url': audio_url, 'error': error_message}), 200 if ai_generation.status == 'completed' else 500
    except Exception as e:
        db.session.rollback()
        logger.error("Text-to-speech route failed", exc_info=True)
        return jsonify({'error_code': 'TTS_ROUTE_FAILED', 'message': 'Text-to-speech generation failed', 'details': str(e)}), 500

def generate_openai_speech_bytes(text, voice='alloy'):
    try:
        openai.api_key = current_app.config.get('OPENAI_API_KEY')
        if not openai.api_key:
            raise Exception("OpenAI API key not configured.")
        response = api_call_with_retry(openai.audio.speech.create, model="tts-1", voice=voice, input=text)
        return response.read()
    except Exception as e:
        raise Exception(f"OpenAI TTS failed: {str(e)}")

def generate_elevenlabs_speech_bytes(text, voice_id):
    try:
        elevenlabs_api_key = current_app.config.get('ELEVENLABS_API_KEY')
        if not elevenlabs_api_key:
            raise Exception("ElevenLabs API key not configured.")
        url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}"
        headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": elevenlabs_api_key}
        data = {"text": text, "model_id": "eleven_monolingual_v1", "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}}
        response = api_call_with_retry(requests.post, url, json=data, headers=headers)
        response.raise_for_status()
        return response.content
    except requests.exceptions.HTTPError as e:
        error_details = e.response.json() if e.response.content else {}
        raise Exception(f"ElevenLabs API HTTP error {e.response.status_code}: {error_details.get('detail', str(e))}")
    except Exception as e:
        raise Exception(f"ElevenLabs TTS failed: {str(e)}")

@ai_bp.route('/voices', methods=['GET'])
@jwt_required()
def get_voices():
    try:
        provider = request.args.get('provider', 'openai').lower()
        voices = []
        if provider == 'elevenlabs':
            voices = get_elevenlabs_voices_cached()
        elif provider == 'openai':
            voices = get_openai_voices()
        else:
            return jsonify({'error_code': 'INVALID_PROVIDER', 'message': 'Invalid voice provider specified'}), 400
        return jsonify({'voices': voices}), 200
    except Exception as e:
        logger.error("Failed to get voices", exc_info=True)
        return jsonify({'error_code': 'GET_VOICES_FAILED', 'message': 'Failed to get voices', 'details': str(e)}), 500

def get_openai_voices():
    return [
        {'id': 'alloy', 'name': 'Alloy', 'gender': 'male', 'language': 'en-US', 'description': 'Warm and friendly', 'preview_url': 'https://example.com/openai_alloy.mp3'},
        {'id': 'echo', 'name': 'Echo', 'gender': 'male', 'language': 'en-US', 'description': 'Clear and precise', 'preview_url': 'https://example.com/openai_echo.mp3'},
        {'id': 'fable', 'name': 'Fable', 'gender': 'female', 'language': 'en-US', 'description': 'Dynamic and expressive', 'preview_url': 'https://example.com/openai_fable.mp3'},
        {'id': 'onyx', 'name': 'Onyx', 'gender': 'male', 'language': 'en-US', 'description': 'Deep and resonant', 'preview_url': 'https://example.com/openai_onyx.mp3'},
        {'id': 'nova', 'name': 'Nova', 'gender': 'female', 'language': 'en-US', 'description': 'Bright and versatile', 'preview_url': 'https://example.com/openai_nova.mp3'},
        {'id': 'shimmer', 'name': 'Shimmer', 'gender': 'female', 'language': 'en-US', 'description': 'Smooth and melodic', 'preview_url': 'https://example.com/openai_shimmer.mp3'}
    ]

def get_elevenlabs_voices_cached():
    redis_client = current_app.extensions['redis_client']
    cache_key = 'elevenlabs_voices_cache'
    cached_voices = redis_client.get(cache_key)
    if cached_voices:
        logger.info("Serving ElevenLabs voices from cache")
        return json.loads(cached_voices)
    logger.info("Fetching ElevenLabs voices from API (not in cache)")
    url = f"{ELEVENLABS_BASE_URL}/voices"
    headers = {"xi-api-key": current_app.config.get('ELEVENLABS_API_KEY')}
    try:
        response = api_call_with_retry(requests.get, url, headers=headers)
        response.raise_for_status()
        data = response.json()
        voices = []
        for voice in data.get('voices', []):
            voices.append({'id': voice['voice_id'], 'name': voice['name'], 'gender': voice.get('labels', {}).get('gender', 'unknown'), 'language': voice.get('labels', {}).get('language', 'en-US'), 'description': voice.get('description', 'No description available'), 'preview_url': voice.get('preview_url')})
        redis_client.setex(cache_key, 12 * 3600, json.dumps(voices))
        return voices
    except Exception as e:
        logger.error("Failed to fetch ElevenLabs voices", error=str(e), exc_info=True)
        return []

@ai_bp.route('/avatar-generate', methods=['POST'])
@jwt_required()
def generate_avatar():
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        avatar_type = data.get('avatar_type')
        avatar_data = data.get('avatar_data', {})
        if not avatar_type:
            return jsonify({'error_code': 'AVATAR_TYPE_REQUIRED', 'message': 'Avatar type is required'}), 400
        ai_generation = AIGeneration(user_id=current_user_id, generation_type='avatar', input_data={'avatar_type': avatar_type, 'avatar_data': avatar_data}, status='processing')
        db.session.add(ai_generation)
        db.session.commit()
        try:
            result = {}
            if avatar_type == 'video':
                result = process_video_avatar(avatar_data)
            elif avatar_type == 'photo':
                result = process_photo_avatar(avatar_data)
            elif avatar_type == 'generative':
                result = generate_ai_avatar(avatar_data)
            else:
                raise ValueError(f"Unsupported avatar type: {avatar_type}")
            ai_generation.status = 'completed'
            ai_generation.output_data = result
            ai_generation.completed_at = datetime.utcnow()
            ai_generation.processing_time = (datetime.utcnow() - ai_generation.created_at).total_seconds()
            ai_generation.cost = 0.0
        except Exception as e:
            ai_generation.status = 'failed'
            ai_generation.error_message = str(e)
            logger.error("Avatar generation failed", error=str(e), exc_info=True, generation_id=ai_generation.id)
        db.session.commit()
        return jsonify({'generation_id': ai_generation.id, 'status': ai_generation.status, 'result': ai_generation.output_data, 'error': ai_generation.error_message}), 200 if ai_generation.status == 'completed' else 500
    except Exception as e:
        db.session.rollback()
        logger.error("Avatar generation route failed", exc_info=True)
        return jsonify({'error_code': 'AVATAR_ROUTE_FAILED', 'message': 'Avatar generation failed', 'details': str(e)}), 500

def process_video_avatar(avatar_data):
    logger.info("Processing video avatar (STUB)", avatar_data=avatar_data)
    return {'avatar_id': f"video_avatar_{uuid.uuid4().hex}", 'avatar_url': "https://example.com/avatars/video_avatar_processed.mp4", 'thumbnail_url': "https://example.com/avatars/video_avatar_thumb.jpg", 'status': 'simulated_success'}

def process_photo_avatar(avatar_data):
    logger.info("Processing photo avatar (STUB)", avatar_data=avatar_data)
    return {'avatar_id': f"photo_avatar_{uuid.uuid4().hex}", 'avatar_url': "https://example.com/avatars/photo_avatar_animated.mp4", 'thumbnail_url': "https://example.com/avatars/photo_avatar_thumb.jpg", 'status': 'simulated_success'}

def generate_ai_avatar(avatar_data):
    logger.info("Generating AI avatar (STUB)", avatar_data=avatar_data)
    return {'avatar_id': f"ai_avatar_{uuid.uuid4().hex}", 'avatar_url': "https://example.com/avatars/ai_avatar_generated.mp4", 'thumbnail_url': "https://example.com/avatars/ai_avatar_thumb.jpg", 'status': 'simulated_success'}

@ai_bp.route('/generations/<int:generation_id>', methods=['GET'])
@jwt_required()
def get_generation_status(generation_id):
    try:
        current_user_id = get_jwt_identity()
        generation = AIGeneration.query.filter_by(id=generation_id, user_id=current_user_id).first()
        if not generation:
            return jsonify({'error_code': 'GENERATION_NOT_FOUND', 'message': 'Generation not found'}), 404
        return jsonify({'generation': generation.to_dict()}), 200
    except Exception as e:
        logger.error("Failed to get generation status", exc_info=True)
        return jsonify({'error_code': 'GET_GENERATION_STATUS_FAILED', 'message': 'Failed to get generation status', 'details': str(e)}), 500

@ai_bp.route('/generations', methods=['GET'])
@jwt_required()
def get_generations():
    try:
        current_user_id = get_jwt_identity()
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        generation_type = request.args.get('type')
        query = AIGeneration.query.filter_by(user_id=current_user_id)
        if generation_type:
            query = query.filter_by(generation_type=generation_type)
        generations_pagination = query.order_by(AIGeneration.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
        next_page = generations_pagination.next_num if generations_pagination.has_next else None
        prev_page = generations_pagination.prev_num if generations_pagination.has_prev else None
        return jsonify({'generations': [gen.to_dict() for gen in generations_pagination.items], 'total': generations_pagination.total, 'pages': generations_pagination.pages, 'current_page': page, 'per_page': per_page, 'next_page_url': request.base_url + f'?page={next_page}&per_page={per_page}' if next_page else None, 'prev_page_url': request.base_url + f'?page={prev_page}&per_page={per_page}' if prev_page else None}), 200
    except Exception as e:
        logger.error("Failed to get user's AI generations", exc_info=True)
        return jsonify({'error_code': 'GET_GENERATIONS_FAILED', 'message': 'Failed to get generations', 'details': str(e)}), 500
