"""Upload results folder to Google Drive using a service account."""

from __future__ import annotations

import argparse
import logging
import mimetypes
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]


def get_service(creds_json: str):
    creds = Credentials.from_service_account_file(creds_json, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def create_folder(service, name: str, parent_id: str | None = None) -> str:
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def set_public_edit(service, file_id: str) -> None:
    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "writer"},
    ).execute()


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
            log.info("Uploaded: %s", item.relative_to(local_dir.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload results to Google Drive")
    parser.add_argument("--creds", required=True, help="Path to service account JSON key")
    parser.add_argument("--folder", default="results", help="Local folder to upload")
    parser.add_argument("--drive-id", default=None, help="Existing Drive folder ID to upload into")
    parser.add_argument("--drive-name", default=None, help="Name for a new Drive folder (ignored if --drive-id is set)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    local = Path(args.folder)
    if not local.exists():
        log.error("Folder not found: %s", local)
        return

    service = get_service(args.creds)

    if args.drive_id:
        root_id = args.drive_id
        log.info("Uploading to existing folder: %s", root_id)
    else:
        drive_name = args.drive_name or local.name
        root_id = create_folder(service, drive_name)
        set_public_edit(service, root_id)
        log.info("Created shared folder: %s", drive_name)

    upload_folder(service, local, root_id)

    link = f"https://drive.google.com/drive/folders/{root_id}"
    log.info("Done! Link: %s", link)
    print(f"\n{'='*60}")
    print(f"Google Drive link: {link}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
