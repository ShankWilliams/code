import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import structlog

logger = structlog.get_logger()

def send_email(to_email, subject, body, from_email):
    """Sends an email using SendGrid."""
    if not os.getenv('SENDGRID_API_KEY'):
        logger.warning("SendGrid API key not found. Email not sent.", to_email=to_email, subject=subject)
        return False

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        html_content=body
    )
    try:
        sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        response = sg.send(message)
        logger.info("Email sent successfully", to_email=to_email, status_code=response.status_code)
        return True
    except Exception as e:
        logger.error("Error sending email", to_email=to_email, error=str(e), exc_info=True)
        return False
