"""Upload folders to Google Drive using OAuth2 installed app flow."""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = "token.json"


def get_credentials(creds_file: str) -> Credentials:
    token_path = Path(TOKEN_FILE)
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
            return creds

    flow = Flow.from_client_secrets_file(creds_file, scopes=SCOPES, redirect_uri="http://localhost")
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")

    print(f"\n1. Open this URL in your browser:\n\n{auth_url}\n")
    print("2. Authorize, then copy the FULL URL from the browser address bar")
    print("   (it will look like http://localhost/?code=4/0A...&scope=...)\n")
    redirect_response = input("Paste the full redirect URL here: ").strip()

    flow.fetch_token(authorization_response=redirect_response)
    creds = flow.credentials
    token_path.write_text(creds.to_json())
    log.info("Token saved to %s", TOKEN_FILE)
    return creds


def create_folder(service, name: str, parent_id: str) -> str:
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def upload_file(service, local_path: Path, parent_id: str) -> None:
    mime = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
    meta = {"name": local_path.name, "parents": [parent_id]}
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    service.files().create(body=meta, media_body=media, fields="id").execute()
    log.info("Uploaded: %s", local_path.name)


def upload_folder(service, local_dir: Path, parent_id: str) -> None:
    for item in sorted(local_dir.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            sub_id = create_folder(service, item.name, parent_id)
            log.info("Created folder: %s", item.name)
            upload_folder(service, item, sub_id)
        else:
            upload_file(service, item, parent_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload folder to Google Drive")
    parser.add_argument("--creds", default="credentials.json", help="OAuth2 client credentials JSON")
    parser.add_argument("--folder", required=True, help="Local folder to upload")
    parser.add_argument("--drive-id", required=True, help="Drive folder ID to upload into")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    local = Path(args.folder)
    if not local.exists():
        log.error("Folder not found: %s", local)
        return

    creds = get_credentials(args.creds)
    service = build("drive", "v3", credentials=creds)

    sub_id = create_folder(service, local.name, args.drive_id)
    log.info("Uploading %s ...", local)
    upload_folder(service, local, sub_id)
    log.info("Done! https://drive.google.com/drive/folders/%s", args.drive_id)


if __name__ == "__main__":
    main()
