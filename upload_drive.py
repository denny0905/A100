"""Upload folders to Google Drive using access token or service account."""

from __future__ import annotations

import argparse
import logging
import mimetypes
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]


def get_service_token(token: str):
    from google.oauth2.credentials import Credentials
    creds = Credentials(token=token)
    return build("drive", "v3", credentials=creds)


def get_service_sa(creds_json: str):
    from google.oauth2.service_account import Credentials as SACreds
    creds = SACreds.from_service_account_file(creds_json, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def create_folder(service, name: str, parent_id: str | None = None) -> str:
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def upload_file(service, local_path: Path, parent_id: str) -> str:
    mime = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
    meta = {"name": local_path.name, "parents": [parent_id]}
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    f = service.files().create(body=meta, media_body=media, fields="id").execute()
    return f["id"]


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
            log.info("Uploaded: %s", item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload folder to Google Drive")
    parser.add_argument("--token", help="OAuth2 access token (from OAuth Playground)")
    parser.add_argument("--creds", help="Path to service account JSON key")
    parser.add_argument("--folder", required=True, help="Local folder to upload")
    parser.add_argument("--drive-id", required=True, help="Drive folder ID to upload into")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    local = Path(args.folder)
    if not local.exists():
        log.error("Folder not found: %s", local)
        return

    if args.token:
        service = get_service_token(args.token)
    elif args.creds:
        service = get_service_sa(args.creds)
    else:
        log.error("Provide --token or --creds")
        return

    sub_id = create_folder(service, local.name, args.drive_id)
    log.info("Uploading %s ...", local)
    upload_folder(service, local, sub_id)

    log.info("Done! Uploaded to https://drive.google.com/drive/folders/%s", args.drive_id)


if __name__ == "__main__":
    main()
