import os
import sys
import logging
import structlog
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from dotenv import load_dotenv
import redis
import openai
import stripe

load_dotenv()

logging.basicConfig(format="%(message)s", stream=sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO').upper())
structlog.configure(processors=[structlog.stdlib.add_logger_name, structlog.stdlib.add_log_level, structlog.processors.TimeStamper(fmt="iso"), structlog.dev.ConsoleRenderer()], logger_factory=structlog.stdlib.LoggerFactory(), cache_logger_on_first_use=True)
logger = structlog.get_logger()

from src.models.database import db
from src.routes.auth import auth_bp
from src.routes.user import user_bp
from src.routes.projects import projects_bp
from src.routes.ai import ai_bp
from src.routes.payments import payments_bp
from src.config.config import get_config
from src.config.pricing import PRICING_PLANS
from src.extensions import limiter

def create_app(config_name='development'):
    app = Flask(__name__, static_folder='static')
    config = get_config(config_name)
    app.config.from_object(config)
    required_env_vars = ['SECRET_KEY', 'JWT_SECRET_KEY', 'DB_USERNAME', 'DB_PASSWORD', 'DB_HOST', 'DB_NAME', 'REDIS_URL']
    for var in required_env_vars:
        if not app.config.get(var):
            logger.critical(f"Missing critical environment variable: {var}")
            sys.exit(f"ERROR: Missing critical environment variable: {var}")
    if config_name in ['production', 'development']:
        pro_plan_price_id = PRICING_PLANS['pro']['stripe_price_id']
        if not pro_plan_price_id:
            logger.critical("STRIPE_PRO_PRICE_ID is missing for the 'pro' plan in pricing.py or .env. Cannot start.")
            sys.exit("ERROR: STRIPE_PRO_PRICE_ID is missing for the 'pro' plan. Check .env and pricing.py")
    cors_origins = app.config.get('CORS_ORIGINS', ['http://localhost:3000'])
    CORS(app, origins=cors_origins, supports_credentials=True)
    logger.info("CORS initialized", origins=cors_origins)
    jwt = JWTManager(app)
    logger.info("JWTManager initialized")
    db.init_app(app)
    migrate = Migrate(app, db)
    logger.info("Database initialized with Flask-Migrate")
    limiter.init_app(app, storage_uri=app.config.get('RATELIMIT_STORAGE_URL', 'memory://'), default_limits=["100 per minute"])
    logger.info("Rate limiter initialized", storage_uri=app.config.get('RATELIMIT_STORAGE_URL'))
    try:
        redis_client = redis.from_url(app.config['REDIS_URL'])
        redis_client.ping()
        app.extensions['redis_client'] = redis_client
        logger.info("Redis client initialized and connected")
    except Exception as e:
        logger.critical(f"Failed to connect to Redis: {e}. Ensure Redis server is running and REDIS_URL is correct.")
        sys.exit(f"ERROR: Failed to connect to Redis: {e}")
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(user_bp, url_prefix='/api/users')
    app.register_blueprint(projects_bp, url_prefix='/api/projects')
    app.register_blueprint(ai_bp, url_prefix='/api/ai')
    app.register_blueprint(payments_bp, url_prefix='/api/payments')
    logger.info("Blueprints registered")
    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        jti = jwt_payload['jti']
        is_revoked = app.extensions['redis_client'].get(f"revoked_token:{jti}") is not None
        if is_revoked:
            logger.warning("Token revoked attempt", jti=jti)
        return is_revoked
    @app.errorhandler(400)
    def bad_request(e):
        logger.warning("Bad Request", error=str(e), path=request.path)
        return jsonify({"error_code": "BAD_REQUEST", "message": "Bad Request: " + str(e.description)}), 400
    @app.errorhandler(401)
    def unauthorized(e):
        logger.warning("Unauthorized Access", error=str(e), path=request.path)
        return jsonify({"error_code": "UNAUTHORIZED", "message": "Authentication Required: " + str(e.description)}), 401
    @app.errorhandler(403)
    def forbidden(e):
        logger.warning("Forbidden Access", error=str(e), path=request.path)
        return jsonify({"error_code": "FORBIDDEN", "message": "Access Denied: " + str(e.description)}), 403
    @app.errorhandler(404)
    def not_found(e):
        logger.warning("Not Found", error=str(e), path=request.path)
        return jsonify({"error_code": "NOT_FOUND", "message": "Resource Not Found: " + str(e.description)}), 404
    @app.errorhandler(405)
    def method_not_allowed(e):
        logger.warning("Method Not Allowed", error=str(e), path=request.path, method=request.method)
        return jsonify({"error_code": "METHOD_NOT_ALLOWED", "message": "Method Not Allowed: " + str(e.description)}), 405
    @app.errorhandler(422)
    def unprocessable_entity(e):
        if hasattr(e, 'data') and 'messages' in e.data:
            return jsonify({"error_code": "VALIDATION_ERROR", "message": "Validation failed", "details": e.data['messages']}), 422
        logger.warning("Unprocessable Entity", error=str(e), path=request.path)
        return jsonify({"error_code": "UNPROCESSABLE_ENTITY", "message": "Unprocessable Entity: " + str(e.description)}), 422
    @app.errorhandler(429)
    def too_many_requests(e):
        logger.warning("Rate Limit Exceeded", error=str(e), path=request.path)
        return jsonify({"error_code": "TOO_MANY_REQUESTS", "message": "Rate limit exceeded. Please try again later."}), 429
    @app.errorhandler(500)
    def internal_error(e):
        logger.exception("Internal Server Error", path=request.path)
        return jsonify({"error_code": "INTERNAL_ERROR", "message": "An unexpected error occurred. Please try again or contact support."}), 500
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        static_folder_path = app.static_folder
        if static_folder_path is None:
            logger.error("Static folder not configured for app")
            return "Static folder not configured", 404
        full_path = os.path.join(static_folder_path, path)
        if path != "" and os.path.exists(full_path) and os.path.isfile(full_path):
            return send_from_directory(static_folder_path, path)
        else:
            index_path = os.path.join(static_folder_path, 'index.html')
            if os.path.exists(index_path):
                return send_from_directory(static_folder_path, 'index.html')
            else:
                logger.error("index.html not found in static folder", static_folder=static_folder_path)
                return "index.html not found", 404
    @app.route('/health')
    def health_check():
        status = {'status': 'healthy', 'service': 'heygen-backend'}
        try:
            db.session.execute(db.select(1))
            status['database'] = 'connected'
        except Exception as e:
            status['database'] = 'failed'
            status['database_error'] = str(e)
            status['status'] = 'unhealthy'
            logger.error("Health check failed: Database connection error", error=str(e))
        try:
            app.extensions['redis_client'].ping()
            status['redis'] = 'connected'
        except Exception as e:
            status['redis'] = 'failed'
            status['redis_error'] = str(e)
            status['status'] = 'unhealthy'
            logger.error("Health check failed: Redis connection error", error=str(e))
        try:
            openai_key = app.config.get('OPENAI_API_KEY')
            if openai_key:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                client.models.list()
                status['openai_api'] = 'reachable'
            else:
                status['openai_api'] = 'not_configured'
        except Exception as e:
            status['openai_api'] = 'failed'
            status['openai_api_error'] = str(e)
            status['status'] = 'unhealthy'
            logger.error("Health check failed: OpenAI API error", error=str(e))
        try:
            stripe_key = app.config.get('STRIPE_SECRET_KEY')
            if stripe_key:
                stripe.api_key = stripe_key
                stripe.Customer.retrieve('non_existent_customer_id_123')
                status['stripe_api'] = 'reachable'
            else:
                status['stripe_api'] = 'not_configured'
        except stripe.error.InvalidRequestError:
            status['stripe_api'] = 'reachable'
        except Exception as e:
            status['stripe_api'] = 'failed'
            status['stripe_api_error'] = str(e)
            status['status'] = 'unhealthy'
            logger.error("Health check failed: Stripe API error", error=str(e))
        return jsonify(status), 200 if status['status'] == 'healthy' else 500
    return app

if __name__ == '__main__':
    config_name = os.getenv('FLASK_ENV', 'development')
    app = create_app(config_name)
    logger.info("Flask app starting", config_name=config_name, debug=app.config['DEBUG'])
    app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])
