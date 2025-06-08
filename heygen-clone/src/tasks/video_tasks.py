from celery import current_task
from celery_app import celery
from src.models.database import db, Project, AIGeneration, Asset, User
from datetime import datetime
from src.utils.aws_s3 import get_s3_client
import requests
import openai
import os
import tempfile
import uuid
import structlog

logger = structlog.get_logger()

def upload_to_s3(file_path, s3_key, content_type='video/mp4'):
    s3_client = get_s3_client()
    bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
    s3_client.upload_file(file_path, bucket_name, s3_key, ExtraArgs={'ContentType': content_type, 'ACL': 'public-read'})
    return f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

def generate_tts_audio(text, voice_settings):
    provider = voice_settings.get('provider', 'openai')
    voice_id = voice_settings.get('voice_id', 'alloy')
    if provider == 'elevenlabs' and os.getenv('ELEVENLABS_API_KEY'):
        return generate_elevenlabs_audio(text, voice_id)
    else:
        return generate_openai_audio(text, voice_id)

def generate_openai_audio(text, voice='alloy'):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.audio.speech.create(model="tts-1", voice=voice, input=text)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        response.stream_to_file(temp_file.name)
        return temp_file.name

def generate_elevenlabs_audio(text, voice_id):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": os.getenv('ELEVENLABS_API_KEY')}
    data = {"text": text, "model_id": "eleven_monolingual_v1", "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}}
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        temp_file.write(response.content)
        return temp_file.name

def process_avatar(avatar_type, avatar_data):
    if avatar_type == 'video':
        return process_video_avatar(avatar_data)
    elif avatar_type == 'photo':
        return process_photo_avatar(avatar_data)
    elif avatar_type == 'generative':
        return generate_ai_avatar(avatar_data)
    else:
        return {'avatar_video_path': None, 'avatar_id': 'default'}

def process_video_avatar(avatar_data):
    return {'avatar_video_path': None, 'avatar_id': f"video_avatar_{uuid.uuid4().hex}"}

def process_photo_avatar(avatar_data):
    return {'avatar_video_path': None, 'avatar_id': f"photo_avatar_{uuid.uuid4().hex}"}

def generate_ai_avatar(avatar_data):
    return {'avatar_video_path': None, 'avatar_id': f"ai_avatar_{uuid.uuid4().hex}"}

def combine_audio_video(audio_path, avatar_result, background_settings):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(b'placeholder video content')
        return temp_file.name

@celery.task(bind=True)
def generate_video_task(self, project_id, generation_id):
    app = celery.flask_app
    audio_path = None
    final_video_path = None
    with app.app_context():
        try:
            current_task.update_state(state='PROGRESS', meta={'step': 'Starting video generation'})
            project = Project.query.get(project_id)
            generation = AIGeneration.query.get(generation_id)
            if not project or not generation:
                raise Exception("Project or generation record not found")
            generation.status = 'processing'
            generation.processed_by = self.request.id
            db.session.commit()
            start_time = datetime.utcnow()
            current_task.update_state(state='PROGRESS', meta={'step': 'Generating audio'})
            logger.info("Generating TTS audio", project_id=project_id)
            audio_path = generate_tts_audio(project.script, project.voice_settings or {})
            audio_s3_key = f"audio/{project.user_id}/{project_id}/{uuid.uuid4().hex}.mp3"
            audio_url = upload_to_s3(audio_path, audio_s3_key, 'audio/mpeg')
            current_task.update_state(state='PROGRESS', meta={'step': 'Processing avatar'})
            logger.info("Processing avatar", project_id=project_id, avatar_type=project.avatar_type)
            avatar_result = process_avatar(project.avatar_type, project.avatar_data or {})
            current_task.update_state(state='PROGRESS', meta={'step': 'Combining audio and video'})
            logger.info("Combining audio and video", project_id=project_id)
            final_video_path = combine_audio_video(audio_path, avatar_result, project.background_settings or {})
            current_task.update_state(state='PROGRESS', meta={'step': 'Uploading video'})
            video_s3_key = f"videos/{project.user_id}/{project_id}/{uuid.uuid4().hex}.mp4"
            video_url = upload_to_s3(final_video_path, video_s3_key, 'video/mp4')
            thumbnail_s3_key = f"thumbnails/{project.user_id}/{project_id}/{uuid.uuid4().hex}.jpg"
            thumbnail_url = f"https://{os.getenv('AWS_S3_BUCKET_NAME')}.s3.amazonaws.com/{thumbnail_s3_key}"
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            script_length = len(project.script)
            estimated_cost = (script_length / 1000) * 0.02
            audio_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            video_size = os.path.getsize(final_video_path) if os.path.exists(final_video_path) else 0
            project.status = 'completed'
            project.output_video_url = video_url
            project.thumbnail_url = thumbnail_url
            project.duration = 30.0
            generation.status = 'completed'
            generation.output_data = {'video_url': video_url, 'audio_url': audio_url, 'thumbnail_url': thumbnail_url, 'avatar_id': avatar_result['avatar_id']}
            generation.processing_time = processing_time
            generation.cost = estimated_cost
            generation.output_size = video_size
            generation.completed_at = end_time
            audio_asset = Asset(user_id=project.user_id, project_id=project_id, asset_type='audio', file_name=f"audio_{project_id}.mp3", file_url=audio_url, file_size=audio_size, mime_type='audio/mpeg')
            video_asset = Asset(user_id=project.user_id, project_id=project_id, asset_type='video', file_name=f"video_{project_id}.mp4", file_url=video_url, file_size=video_size, mime_type='video/mp4')
            db.session.add(audio_asset)
            db.session.add(video_asset)
            db.session.commit()
            logger.info("Video generation completed successfully", project_id=project_id, generation_id=generation_id, processing_time=processing_time)
            return {'status': 'completed', 'video_url': video_url, 'thumbnail_url': thumbnail_url, 'processing_time': processing_time, 'cost': estimated_cost}
        except Exception as e:
            logger.error("Video generation failed", project_id=project_id, generation_id=generation_id, error=str(e))
            try:
                project = Project.query.get(project_id)
                generation = AIGeneration.query.get(generation_id)
                if project:
                    project.status = 'failed'
                if generation:
                    generation.status = 'failed'
                    generation.error_message = str(e)
                    generation.completed_at = datetime.utcnow()
                db.session.commit()
            except:
                pass
            raise e
        finally:
            try:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                if final_video_path and os.path.exists(final_video_path):
                    os.unlink(final_video_path)
            except Exception:
                pass
            db.session.remove()
