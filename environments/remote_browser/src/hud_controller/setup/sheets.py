"""Setup functions for Google Sheets operations."""

import base64
import io
import json
import logging
import os
import httpx
from typing import Dict, Any, Optional
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2.service_account import Credentials
from .registry import setup
from ..evaluators.context import RemoteBrowserContext

logger = logging.getLogger(__name__)


def get_gcp_credentials() -> Dict[str, str]:
    """
    Get GCP credentials from environment variable.

    Expects either:
    1. GCP_CREDENTIALS_JSON - A JSON string containing the full credentials
    2. Individual environment variables for each field

    Returns:
        Dict containing GCP service account credentials
    """
    # First try to get from JSON env var
    creds_json = os.getenv("GCP_CREDENTIALS_JSON")
    if creds_json:
        try:
            return json.loads(creds_json)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in GCP_CREDENTIALS_JSON")
            raise ValueError("Invalid GCP_CREDENTIALS_JSON format")

    # Otherwise try to build from individual env vars
    required_fields = [
        "type",
        "project_id",
        "private_key_id",
        "private_key",
        "client_email",
        "client_id",
        "auth_uri",
        "token_uri",
        "auth_provider_x509_cert_url",
        "client_x509_cert_url",
    ]

    credentials = {}
    for field in required_fields:
        env_key = f"GCP_{field.upper()}"
        value = os.getenv(env_key)
        if not value:
            raise ValueError(f"Missing required GCP credential field: {env_key}")
        credentials[field] = value

    # Add universe_domain with default
    credentials["universe_domain"] = os.getenv("GCP_UNIVERSE_DOMAIN", "googleapis.com")

    return credentials


@setup("sheets_from_xlsx")
class SheetsFromXlsxSetup:
    """Setup function to create a Google Sheet from an Excel file URL."""

    name = "sheets_from_xlsx"

    def __init__(self, context):
        """Initialize the setup class."""
        self.context = context

    async def __call__(
        self,
        file_url: str | None = None,
        sheet_name: str = "Worksheet",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a Google Sheet from an Excel file URL.

        Args:
            file_url: URL of the Excel file to convert
            sheet_name: Name for the new Google Sheet (default: "Worksheet")

        Returns:
            Status dictionary with sheet information
        """
        logger.info("Starting sheets_from_xlsx setup")

        # Handle backward compatibility for list args
        if "args" in kwargs and isinstance(kwargs["args"], list):
            args_list = kwargs["args"]
            if len(args_list) > 0:
                file_url = args_list[0]
            if len(args_list) > 1:
                sheet_name = args_list[1]

        # Validate parameters
        if not file_url:
            logger.error("Missing required file_url parameter")
            return {
                "status": "error",
                "message": "Missing required parameter: file_url",
            }

        logger.info(f"Creating sheet from URL: {file_url}, name: {sheet_name}")

        try:
            # Download the Excel file
            logger.info("Downloading Excel file")
            async with httpx.AsyncClient() as client:
                response = await client.get(file_url)
                response.raise_for_status()
                file_content = response.content

            logger.info(f"Downloaded {len(file_content)} bytes")

            # Create Google Drive service
            scopes = ["https://www.googleapis.com/auth/drive"]
            gcp_creds = get_gcp_credentials()
            credentials = Credentials.from_service_account_info(gcp_creds, scopes=scopes)
            drive_service = build("drive", "v3", credentials=credentials)

            # Upload to Google Drive with conversion
            file_metadata = {
                "name": sheet_name,
                "mimeType": "application/vnd.google-apps.spreadsheet",
            }

            media = MediaIoBaseUpload(
                io.BytesIO(file_content),
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                resumable=True,
            )

            logger.info("Uploading to Google Drive with conversion to Sheets")
            drive_file = (
                drive_service.files()
                .create(body=file_metadata, media_body=media, fields="id,webViewLink")
                .execute()
            )

            sheet_id = drive_file.get("id")
            sheet_url = drive_file.get("webViewLink")

            logger.info(f"Created Google Sheet: {sheet_id}")

            # Set sharing permissions
            permission = {"type": "anyone", "role": "writer", "allowFileDiscovery": False}

            drive_service.permissions().create(
                fileId=sheet_id, body=permission, fields="id"
            ).execute()

            logger.info("Set sharing permissions")

            # Navigate to the sheet
            page = self.context.page
            if page:
                logger.info(f"Navigating to sheet: {sheet_url}")
                await page.goto(sheet_url, wait_until="domcontentloaded", timeout=15000)

                # Wait for sheet to load
                try:
                    await page.wait_for_selector(".grid-container", timeout=10000)
                    logger.info("Sheet loaded successfully")
                except:
                    logger.warning("Timeout waiting for sheet to fully load")

            sheet_info = {"sheet_id": sheet_id, "sheet_url": sheet_url, "sheet_name": sheet_name}

            return {
                "status": "success",
                "message": f"Created and navigated to Google Sheet: {sheet_name}",
                "sheet_info": sheet_info,
            }

        except Exception as e:
            logger.error(f"Error in sheets_from_xlsx: {str(e)}")
            return {"status": "error", "message": f"Failed to create sheet: {str(e)}"}


@setup("sheets_from_bytes")
class SheetsFromBytesSetup:
    """Setup function to create a Google Sheet from base64 encoded Excel bytes."""

    name = "sheets_from_bytes"

    def __init__(self, context):
        """Initialize the setup class."""
        self.context = context

    async def __call__(
        self,
        base64_bytes: str | None = None,
        sheet_name: str = "Worksheet",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a Google Sheet from base64 encoded Excel bytes.

        Args:
            base64_bytes: Base64 encoded Excel file bytes
            sheet_name: Name for the new Google Sheet (default: "Worksheet")

        Returns:
            Status dictionary with sheet information
        """
        logger.info("Starting sheets_from_bytes setup")

        # Handle backward compatibility for list args
        if "args" in kwargs and isinstance(kwargs["args"], list):
            args_list = kwargs["args"]
            if len(args_list) > 0:
                base64_bytes = args_list[0]
            if len(args_list) > 1:
                sheet_name = args_list[1]

        # Validate parameters
        if not base64_bytes:
            logger.error("Missing required base64_bytes parameter")
            return {
                "status": "error",
                "message": "Missing required parameter: base64_bytes",
            }

        file_bytes_b64 = base64_bytes

        logger.info(f"Creating sheet from bytes, name: {sheet_name}")

        try:
            # Decode base64 bytes
            file_bytes = base64.b64decode(file_bytes_b64)
            logger.info(f"Decoded {len(file_bytes)} bytes")

            # Create Google Drive service
            scopes = ["https://www.googleapis.com/auth/drive"]
            gcp_creds = get_gcp_credentials()
            credentials = Credentials.from_service_account_info(gcp_creds, scopes=scopes)
            drive_service = build("drive", "v3", credentials=credentials)

            # Upload to Google Drive with conversion
            file_metadata = {
                "name": sheet_name,
                "mimeType": "application/vnd.google-apps.spreadsheet",
            }

            media = MediaIoBaseUpload(
                io.BytesIO(file_bytes),
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                resumable=True,
            )

            logger.info("Uploading to Google Drive with conversion to Sheets")
            drive_file = (
                drive_service.files()
                .create(body=file_metadata, media_body=media, fields="id,webViewLink")
                .execute()
            )

            sheet_id = drive_file.get("id")
            sheet_url = drive_file.get("webViewLink")

            logger.info(f"Created Google Sheet: {sheet_id}")

            # Set sharing permissions
            permission = {"type": "anyone", "role": "writer", "allowFileDiscovery": False}

            drive_service.permissions().create(
                fileId=sheet_id, body=permission, fields="id"
            ).execute()

            logger.info("Set sharing permissions")

            # Navigate to the sheet
            page = self.context.page
            if page:
                logger.info(f"Navigating to sheet: {sheet_url}")
                await page.goto(sheet_url, wait_until="domcontentloaded", timeout=15000)

                # Wait for sheet to load
                try:
                    await page.wait_for_selector(".grid-container", timeout=10000)
                    logger.info("Sheet loaded successfully")
                except:
                    logger.warning("Timeout waiting for sheet to fully load")

            sheet_info = {"sheet_id": sheet_id, "sheet_url": sheet_url, "sheet_name": sheet_name}

            return {
                "status": "success",
                "message": f"Created and navigated to Google Sheet: {sheet_name}",
                "sheet_info": sheet_info,
            }

        except Exception as e:
            logger.error(f"Error in sheets_from_bytes: {str(e)}")
            return {"status": "error", "message": f"Failed to create sheet: {str(e)}"}
