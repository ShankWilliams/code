import os
import structlog
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from src.models.database import db, User, Subscription
from src.config.pricing import PRICING_PLANS, get_plan_features
from src.utils.email_service import send_email
import stripe

payments_bp = Blueprint('payments', __name__)
logger = structlog.get_logger()

@payments_bp.route('/plans', methods=['GET'])
def get_pricing_plans():
    return jsonify({'plans': PRICING_PLANS}), 200

@payments_bp.route('/create-checkout-session', methods=['POST'])
@jwt_required()
def create_checkout_session():
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        data = request.get_json()
        plan_id = data.get('plan_id')
        if plan_id not in PRICING_PLANS:
            return jsonify({'error_code': 'INVALID_PLAN_ID', 'message': 'Invalid plan ID'}), 400
        plan = PRICING_PLANS[plan_id]
        if not plan['stripe_price_id']:
            if plan_id == 'enterprise':
                return jsonify({'error_code': 'CONTACT_SALES', 'message': 'This plan requires contacting sales.'}), 400
            return jsonify({'error_code': 'PLAN_NOT_PURCHASABLE', 'message': 'Plan not available for online purchase (missing price ID).'}), 400
        if not user.stripe_customer_id:
            logger.info("Creating new Stripe customer for user", user_id=user.id, email=user.email)
            customer = stripe.Customer.create(email=user.email, name=f"{user.first_name} {user.last_name}", metadata={'user_id': user.id})
            user.stripe_customer_id = customer.id
            db.session.commit()
        checkout_session = stripe.checkout.Session.create(customer=user.stripe_customer_id, payment_method_types=['card'], line_items=[{'price': plan['stripe_price_id'], 'quantity': 1}], mode='subscription', success_url=request.host_url + 'dashboard?payment=success', cancel_url=request.host_url + 'pricing?payment=cancelled', metadata={'user_id': str(user.id), 'plan_id': plan_id})
        return jsonify({'checkout_url': checkout_session.url, 'session_id': checkout_session.id}), 200
    except stripe.error.StripeError as se:
        logger.error("Stripe error creating checkout session", error=str(se), exc_info=True)
        return jsonify({'error_code': 'STRIPE_ERROR', 'message': 'Stripe payment processing failed', 'details': str(se)}), 500
    except Exception as e:
        logger.error("Failed to create checkout session", error=str(e), exc_info=True)
        return jsonify({'error_code': 'CHECKOUT_SESSION_FAILED', 'message': 'Failed to create checkout session', 'details': str(e)}), 500

@payments_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = current_app.config.get('STRIPE_WEBHOOK_SECRET')
    if not endpoint_secret:
        logger.error("Stripe webhook secret not configured.", payload=payload.decode())
        return jsonify({'error_code': 'WEBHOOK_SECRET_MISSING', 'message': 'Stripe webhook secret not configured'}), 500
    event = None
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except ValueError as e:
        logger.error("Invalid Stripe webhook payload", error=str(e))
        return jsonify({'error_code': 'INVALID_PAYLOAD', 'message': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        logger.error("Invalid Stripe webhook signature", error=str(e))
        return jsonify({'error_code': 'INVALID_SIGNATURE', 'message': 'Invalid signature'}), 400
    logger.info("Received Stripe webhook event", event_type=event['type'], event_id=event['id'])
    try:
        if event['type'] == 'checkout.session.completed':
            handle_checkout_completed(event['data']['object'])
        elif event['type'] == 'invoice.payment_succeeded':
            handle_payment_succeeded(event['data']['object'])
        elif event['type'] == 'customer.subscription.updated':
            handle_subscription_updated(event['data']['object'])
        elif event['type'] == 'customer.subscription.deleted':
            handle_subscription_cancelled(event['data']['object'])
        else:
            logger.warning("Unhandled Stripe webhook event type", event_type=event['type'])
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logger.error("Error processing Stripe webhook event", event_type=event['type'], event_id=event['id'], error=str(e), exc_info=True)
        return jsonify({'status': 'error', 'message': 'Internal processing error'}), 200

def handle_checkout_completed(session):
    try:
        user_id = session['metadata'].get('user_id')
        plan_id = session['metadata'].get('plan_id')
        if not user_id or not plan_id:
            logger.error("Missing metadata in checkout.session.completed event", session_id=session.id)
            return
        user = User.query.get(user_id)
        if not user:
            logger.error("User not found for checkout.session.completed event", user_id=user_id)
            return
        subscription = stripe.Subscription.retrieve(session['subscription'])
        user.subscription_tier = plan_id
        user.subscription_status = subscription.status
        existing_subscription = Subscription.query.filter_by(stripe_subscription_id=subscription.id).first()
        if not existing_subscription:
            existing_subscription = Subscription.query.filter_by(user_id=user_id).first()
        if existing_subscription:
            existing_subscription.stripe_subscription_id = subscription.id
            existing_subscription.plan_id = plan_id
            existing_subscription.status = subscription.status
            existing_subscription.current_period_start = datetime.fromtimestamp(subscription.current_period_start)
            existing_subscription.current_period_end = datetime.fromtimestamp(subscription.current_period_end)
            existing_subscription.cancel_at_period_end = subscription.cancel_at_period_end
            logger.info("Existing subscription updated", sub_id=existing_subscription.id)
        else:
            new_subscription = Subscription(user_id=user.id, stripe_subscription_id=subscription.id, plan_id=plan_id, status=subscription.status, current_period_start=datetime.fromtimestamp(subscription.current_period_start), current_period_end=datetime.fromtimestamp(subscription.current_period_end), cancel_at_period_end=subscription.cancel_at_period_end)
            db.session.add(new_subscription)
            logger.info("New subscription created", user_id=user.id, plan_id=plan_id)
        db.session.commit()
        email_subject = f"Your HeyGen Clone {plan_id.capitalize()} Plan is Active!"
        email_body = f"Hello {user.first_name},\n\nYour subscription to the {plan_id.capitalize()} plan is now active! You can access all your new features.\n\nThank you for your purchase!\n\nThe HeyGen Clone Team"
        if os.getenv('SENDGRID_API_KEY'):
            send_email(user.email, email_subject, email_body, current_app.config['FROM_EMAIL'])
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error handling checkout.session.completed: {e}", exc_info=True)

def handle_payment_succeeded(invoice):
    try:
        subscription_id = invoice.get('subscription')
        if not subscription_id:
            logger.warning("Invoice.payment_succeeded event without subscription ID", invoice_id=invoice.id)
            return
        subscription_obj = stripe.Subscription.retrieve(subscription_id)
        db_subscription = Subscription.query.filter_by(stripe_subscription_id=subscription_id).first()
        if db_subscription:
            db_subscription.status = subscription_obj.status
            db_subscription.current_period_start = datetime.fromtimestamp(subscription_obj.current_period_start)
            db_subscription.current_period_end = datetime.fromtimestamp(subscription_obj.current_period_end)
            db.session.commit()
            logger.info("Subscription payment succeeded and updated", sub_id=db_subscription.id)
        else:
            logger.warning("Subscription not found in DB for payment_succeeded", stripe_sub_id=subscription_id)
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error handling invoice.payment_succeeded: {e}", exc_info=True)

def handle_subscription_updated(subscription_obj):
    try:
        db_subscription = Subscription.query.filter_by(stripe_subscription_id=subscription_obj['id']).first()
        if db_subscription:
            old_status = db_subscription.status
            db_subscription.status = subscription_obj['status']
            db_subscription.cancel_at_period_end = subscription_obj['cancel_at_period_end']
            db_subscription.current_period_start = datetime.fromtimestamp(subscription_obj['current_period_start'])
            db_subscription.current_period_end = datetime.fromtimestamp(subscription_obj['current_period_end'])
            user = User.query.get(db_subscription.user_id)
            if user:
                user.subscription_status = subscription_obj['status']
                user.subscription_tier = db_subscription.plan_id
            db.session.commit()
            logger.info("Subscription updated in DB", sub_id=db_subscription.id, old_status=old_status, new_status=subscription_obj['status'])
            if old_status != 'canceled' and subscription_obj['cancel_at_period_end'] and user and os.getenv('SENDGRID_API_KEY'):
                email_subject = "Your HeyGen Clone Subscription Cancellation"
                email_body = f"Hello {user.first_name},\n\nYour subscription will be canceled at the end of your current billing period ({db_subscription.current_period_end.strftime('%Y-%m-%d')}). We're sad to see you go!\n\nIf you've changed your mind, you can resubscribe anytime from your dashboard.\n\nThe HeyGen Clone Team"
                send_email(user.email, email_subject, email_body, current_app.config['FROM_EMAIL'])
                logger.info("Sent cancellation notification email", user_id=user.id)
        else:
            logger.warning("Subscription not found in DB for update", stripe_sub_id=subscription_obj['id'])
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error handling customer.subscription.updated: {e}", exc_info=True)

def handle_subscription_cancelled(subscription_obj):
    try:
        db_subscription = Subscription.query.filter_by(stripe_subscription_id=subscription_obj['id']).first()
        if db_subscription:
            db_subscription.status = 'cancelled'
            user = User.query.get(db_subscription.user_id)
            if user:
                user.subscription_tier = 'free'
                user.subscription_status = 'cancelled'
            db.session.commit()
            logger.info("Subscription cancelled and user downgraded", sub_id=db_subscription.id, user_id=user.id if user else 'N/A')
            if user and os.getenv('SENDGRID_API_KEY'):
                email_subject = "Your HeyGen Clone Subscription Has Been Canceled"
                email_body = f"Hello {user.first_name},\n\nYour subscription to the {db_subscription.plan_id.capitalize()} plan has been successfully canceled. Your plan features will remain active until {db_subscription.current_period_end.strftime('%Y-%m-%d')}.\n\nThank you for using HeyGen Clone!\n\nThe HeyGen Clone Team"
                send_email(user.email, email_subject, email_body, current_app.config['FROM_EMAIL'])
        else:
            logger.warning("Subscription not found in DB for cancellation event", stripe_sub_id=subscription_obj['id'])
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error handling customer.subscription.deleted: {e}", exc_info=True)

@payments_bp.route('/subscription', methods=['GET'])
@jwt_required()
def get_subscription():
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'error_code': 'USER_NOT_FOUND', 'message': 'User not found'}), 404
        subscription = Subscription.query.filter_by(user_id=current_user_id).first()
        return jsonify({'subscription_tier': user.subscription_tier, 'subscription_status': user.subscription_status, 'subscription': subscription.to_dict() if subscription else None, 'plan_details': get_plan_features(user.subscription_tier)}), 200
    except Exception as e:
        logger.error("Failed to get subscription info", exc_info=True)
        return jsonify({'error_code': 'GET_SUBSCRIPTION_FAILED', 'message': 'Failed to get subscription', 'details': str(e)}), 500

@payments_bp.route('/cancel-subscription', methods=['POST'])
@jwt_required()
def cancel_subscription():
    try:
        current_user_id = get_jwt_identity()
        subscription = Subscription.query.filter_by(user_id=current_user_id).first()
        if not subscription:
            return jsonify({'error_code': 'NO_ACTIVE_SUBSCRIPTION', 'message': 'No active subscription found'}), 404
        if subscription.cancel_at_period_end:
            return jsonify({'message': 'Subscription is already set to cancel at period end'}), 200
        stripe.Subscription.modify(subscription.stripe_subscription_id, cancel_at_period_end=True)
        subscription.cancel_at_period_end = True
        db.session.commit()
        return jsonify({'message': 'Subscription will be cancelled at the end of the current period'}), 200
    except stripe.error.StripeError as se:
        logger.error("Stripe error cancelling subscription", error=str(se), exc_info=True)
        return jsonify({'error_code': 'STRIPE_ERROR', 'message': 'Stripe subscription cancellation failed', 'details': str(se)}), 500
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to cancel subscription", exc_info=True)
        return jsonify({'error_code': 'CANCEL_SUBSCRIPTION_FAILED', 'message': 'Failed to cancel subscription', 'details': str(e)}), 500

@payments_bp.route('/billing-portal', methods=['POST'])
@jwt_required()
def create_billing_portal():
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user or not user.stripe_customer_id:
            return jsonify({'error_code': 'NO_BILLING_ACCOUNT', 'message': 'No billing account found or customer ID missing'}), 404
        session = stripe.billing_portal.Session.create(customer=user.stripe_customer_id, return_url=request.host_url + 'dashboard')
        return jsonify({'portal_url': session.url}), 200
    except stripe.error.StripeError as se:
        logger.error("Stripe error creating billing portal", error=str(se), exc_info=True)
        return jsonify({'error_code': 'STRIPE_ERROR', 'message': 'Stripe billing portal creation failed', 'details': str(se)}), 500
    except Exception as e:
        logger.error("Failed to create billing portal", exc_info=True)
        return jsonify({'error_code': 'BILLING_PORTAL_FAILED', 'message': 'Failed to create billing portal', 'details': str(e)}), 500
