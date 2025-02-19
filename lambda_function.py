import os
import json
import base64
import uuid
import boto3
import logging
from twilio.rest import Client
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail,
    Email,
    To,
    Content,
    Attachment,
    FileContent,
    FileName,
    FileType,
    Disposition,
)

# Hard-coded constants (example values)
S3_BUCKET_NAME = "your-s3-bucket-name"

FROM_PHONE_NUMBER = "your-twilio-sms-phone-number"
TO_PHONE_NUMBER = "sms-sent-to-number"
FROM_EMAIL_ADDRESS = "your-email-address"
TO_EMAIL_ADDRESS = "your-email-address"

# Environment variables for credentials
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
SENDGRID_API_KEY = os.environ["SENDGRID_API_KEY"]

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    logger.info("Lambda started, event: %s", event)

    full_payload = json.loads(event["body"])
    body = full_payload["body"]
    anomalies = body.get("anomalies")
    image_data_base64 = body.get("image_data")

    logger.info("Number of anomalies: %s", anomalies)

    # Decode the base64 image
    image_bytes = base64.b64decode(image_data_base64)

    logger.info("length of image bytes: %s", len(image_bytes))

    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}.jpeg"

    logger.info("filename: %s", unique_filename)

    # Upload to S3
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=unique_filename,
        Body=image_bytes,
        ContentType="image/jpeg",
    )

    logger.info("image file saved!")

    # Generate a presigned URL for the file
    presigned_url = s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": unique_filename},
        ExpiresIn=3600,  # 1 hour
    )

    logger.info("image url: %s", presigned_url)

    # Send an email with Twilio SendGrid
    sg_client = SendGridAPIClient(SENDGRID_API_KEY)
    from_email = Email(FROM_EMAIL_ADDRESS)
    to_email = To(TO_EMAIL_ADDRESS)
    subject = "Anomalies Detected!"
    content = Content(
        "text/plain", f"Please see the attached image. Number of anomalies: {anomalies}"
    )

    mail = Mail(from_email, to_email, subject, content)

    # Attach the image
    encoded_file = base64.b64encode(image_bytes).decode()
    attachment = Attachment(
        FileContent(encoded_file),
        FileName(unique_filename),
        FileType("image/jpeg"),
        Disposition("attachment"),
    )
    mail.add_attachment(attachment)

    response = sg_client.send(mail)

    logger.info("sendgrid email sent: %s", response)

    # Send SMS/MMS with Twilio
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    sms_message = twilio_client.messages.create(
        body=f"Alert! Number of anomalies detected: {anomalies}",
        from_=FROM_PHONE_NUMBER,
        to=TO_PHONE_NUMBER,
        media_url=[presigned_url],
    )

    logger.info("twilio sms sent!")

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "Image uploaded, SMS and Email sent successfully.",
                "sms_sid": sms_message.sid,
                "email_status_code": response.status_code,
            }
        ),
    }
