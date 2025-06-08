import os
from celery import Celery
from dotenv import load_dotenv
from app import create_app

load_dotenv()

celery = Celery(__name__)
flask_app = create_app(os.getenv('FLASK_ENV', 'development'))
celery.flask_app = flask_app
celery.conf.update(flask_app.config)

celery.conf.broker_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
celery.conf.result_backend = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

celery.autodiscover_tasks(['src.tasks'])

celery.conf.task_routes = {
    'src.tasks.video_tasks.*': {'queue': 'video'},
    'src.tasks.audio_tasks.*': {'queue': 'audio'},
    'src.tasks.email_tasks.*': {'queue': 'email'}
}

celery.conf.task_time_limit = 1800
celery.conf.task_soft_time_limit = 1500

if __name__ == '__main__':
    celery.start()
