import structlog
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
from src.models.database import db, Project, AIGeneration, User
from src.tasks.video_tasks import generate_video_task
from marshmallow import Schema, fields, validate, ValidationError
from src.config.pricing import get_plan_features, validate_plan_limits

projects_bp = Blueprint('projects', __name__)
logger = structlog.get_logger()

class ProjectSettingsSchema(Schema):
    avatar_id = fields.String(required=True)
    voice_id = fields.String(required=True)
    background_type = fields.String(validate=validate.OneOf(['color', 'image', 'video']), missing='color')
    background_value = fields.String(allow_none=True)
    resolution = fields.String(validate=validate.OneOf(['720p', '1080p', '4k']), missing='1080p')

class ProjectSchema(Schema):
    title = fields.String(required=True, validate=validate.Length(min=1, max=200))
    description = fields.String(validate=validate.Length(max=500), allow_none=True, missing='')
    script = fields.String(required=True, validate=validate.Length(min=10, max=5000))
    avatar_type = fields.String(validate=validate.OneOf(['video', 'photo', 'generative', 'stock']), required=True)
    avatar_data = fields.Dict(allow_none=True)
    voice_settings = fields.Dict(required=True)
    background_settings = fields.Dict(allow_none=True)
    status = fields.String(validate=validate.OneOf(['draft', 'processing', 'completed', 'failed']), missing='draft')

@projects_bp.route('/', methods=['GET'])
@jwt_required()
def get_projects():
    try:
        current_user_id = int(get_jwt_identity())
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        status_filter = request.args.get('status')
        query = Project.query.filter_by(user_id=current_user_id)
        if status_filter:
            query = query.filter_by(status=status_filter)
        projects_pagination = query.order_by(Project.updated_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
        next_page = projects_pagination.next_num if projects_pagination.has_next else None
        prev_page = projects_pagination.prev_num if projects_pagination.has_prev else None
        return jsonify({'projects': [project.to_dict() for project in projects_pagination.items], 'total': projects_pagination.total, 'pages': projects_pagination.pages, 'current_page': page, 'per_page': per_page, 'next_page_url': request.base_url + f'?page={next_page}&per_page={per_page}' if next_page else None, 'prev_page_url': request.base_url + f'?page={prev_page}&per_page={per_page}' if prev_page else None}), 200
    except Exception as e:
        logger.error("Failed to get projects", exc_info=True)
        return jsonify({'error_code': 'GET_PROJECTS_FAILED', 'message': 'Failed to get projects', 'details': str(e)}), 500

@projects_bp.route('/', methods=['POST'])
@jwt_required()
def create_project():
    try:
        current_user_id = int(get_jwt_identity())
        data = request.get_json()
        try:
            validated_data = ProjectSchema().load(data)
        except ValidationError as err:
            return jsonify({'error_code': 'VALIDATION_ERROR', 'message': 'Invalid project data', 'details': err.messages}), 422
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        plan_features = get_plan_features(user.subscription_tier)
        if not validate_plan_limits(user, 'project_create'):
            return jsonify({'error_code': 'PROJECT_LIMIT_REACHED', 'message': f"You have reached your project limit of {plan_features['features']['projects_limit']} for your current plan."}), 403
        project = Project(user_id=current_user_id, title=validated_data['title'], description=validated_data.get('description'), script=validated_data['script'], avatar_type=validated_data['avatar_type'], avatar_data=validated_data.get('avatar_data'), voice_settings=validated_data['voice_settings'], background_settings=validated_data.get('background_settings'), status='draft')
        db.session.add(project)
        db.session.commit()
        return jsonify({'message': 'Project created successfully', 'project': project.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to create project", exc_info=True)
        return jsonify({'error_code': 'CREATE_PROJECT_FAILED', 'message': 'Failed to create project', 'details': str(e)}), 500

@projects_bp.route('/<int:project_id>', methods=['GET'])
@jwt_required()
def get_project(project_id):
    try:
        current_user_id = int(get_jwt_identity())
        project = Project.query.filter_by(id=project_id, user_id=current_user_id).first()
        if not project:
            return jsonify({'error_code': 'PROJECT_NOT_FOUND', 'message': 'Project not found'}), 404
        return jsonify({'project': project.to_dict()}), 200
    except Exception as e:
        logger.error("Failed to get project", exc_info=True)
        return jsonify({'error_code': 'GET_PROJECT_FAILED', 'message': 'Failed to get project', 'details': str(e)}), 500

@projects_bp.route('/<int:project_id>', methods=['PUT'])
@jwt_required()
def update_project(project_id):
    try:
        current_user_id = int(get_jwt_identity())
        project = Project.query.filter_by(id=project_id, user_id=current_user_id).first()
        if not project:
            return jsonify({'error_code': 'PROJECT_NOT_FOUND', 'message': 'Project not found'}), 404
        if project.status != 'draft':
            return jsonify({'error_code': 'INVALID_PROJECT_STATUS', 'message': 'Cannot update project unless in draft status'}), 403
        data = request.get_json()
        try:
            validated_data = ProjectSchema(partial=True).load(data)
        except ValidationError as err:
            return jsonify({'error_code': 'VALIDATION_ERROR', 'message': 'Invalid project data', 'details': err.messages}), 422
        for field, value in validated_data.items():
            setattr(project, field, value)
        project.updated_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'message': 'Project updated successfully', 'project': project.to_dict()}), 200
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to update project", exc_info=True)
        return jsonify({'error_code': 'UPDATE_PROJECT_FAILED', 'message': 'Failed to update project', 'details': str(e)}), 500

@projects_bp.route('/<int:project_id>', methods=['DELETE'])
@jwt_required()
def delete_project(project_id):
    try:
        current_user_id = int(get_jwt_identity())
        project = Project.query.filter_by(id=project_id, user_id=current_user_id).first()
        if not project:
            return jsonify({'error_code': 'PROJECT_NOT_FOUND', 'message': 'Project not found'}), 404
        db.session.delete(project)
        db.session.commit()
        return jsonify({'message': 'Project deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to delete project", exc_info=True)
        return jsonify({'error_code': 'DELETE_PROJECT_FAILED', 'message': 'Failed to delete project', 'details': str(e)}), 500

@projects_bp.route('/<int:project_id>/generate', methods=['POST'])
@jwt_required()
def trigger_video_generation(project_id):
    try:
        current_user_id = int(get_jwt_identity())
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        project = Project.query.filter_by(id=project_id, user_id=current_user_id).first()
        if not project:
            return jsonify({'error_code': 'PROJECT_NOT_FOUND', 'message': 'Project not found'}), 404
        if not project.script:
            return jsonify({'error_code': 'SCRIPT_REQUIRED', 'message': 'Project script is required for video generation'}), 400
        plan_limits = get_plan_features(user.subscription_tier)['limits']
        if not validate_plan_limits(user, 'ai_generation'):
            return jsonify({'error_code': 'GENERATION_LIMIT', 'message': 'AI generation limit reached for your plan.'}), 403
        active_generations = AIGeneration.query.filter_by(user_id=current_user_id, generation_type='video', status='processing').count()
        if plan_limits['concurrent_generations'] != 'unlimited' and active_generations >= plan_limits['concurrent_generations']:
            return jsonify({'error_code': 'CONCURRENT_GENERATION_LIMIT', 'message': 'You have reached your concurrent video generation limit.'}), 403
        ai_generation = AIGeneration(user_id=current_user_id, project_id=project_id, generation_type='video', input_data={'script': project.script, 'avatar_type': project.avatar_type, 'avatar_data': project.avatar_data, 'voice_settings': project.voice_settings, 'background_settings': project.background_settings}, status='pending', provider='mixed')
        db.session.add(ai_generation)
        db.session.commit()
        generate_video_task.delay(project.id, ai_generation.id)
        project.status = 'processing'
        db.session.commit()
        return jsonify({'message': 'Video generation started successfully. You can check its status via the /ai/generations endpoint.', 'generation_id': ai_generation.id, 'project_id': project.id, 'status': project.status}), 202
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to start video generation", exc_info=True)
        return jsonify({'error_code': 'START_GENERATION_FAILED', 'message': 'Failed to start video generation', 'details': str(e)}), 500
