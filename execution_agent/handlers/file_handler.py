import logging
import os
import shutil

from execution_agent.handlers.base_handler import BaseHandler
from execution_agent.handlers.registry import registry

logger = logging.getLogger(__name__)

def _local_upload(local_path: str, destination: str, config: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    shutil.copy2(local_path, destination)
    return destination

def _local_download(source: str, local_path: str, config: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    shutil.copy2(source, local_path)
    return local_path


def _s3_client(config: dict):
    import boto3
    return boto3.client(
        "s3",
        region_name           = config.get("region") or os.getenv("AWS_DEFAULT_REGION"),
        aws_access_key_id     = config.get("aws_access_key") or os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = config.get("aws_secret_key") or os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

def _s3_upload(local_path: str, destination: str, config: dict) -> str:
    bucket = config.get("bucket", "")
    if not bucket:
        raise ValueError("S3: 'bucket' is required.")
    _s3_client(config).upload_file(local_path, bucket, destination)
    return f"s3://{bucket}/{destination}"

def _s3_download(source: str, local_path: str, config: dict) -> str:
    bucket = config.get("bucket", "")
    if not bucket:
        raise ValueError("S3: 'bucket' is required.")
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    _s3_client(config).download_file(bucket, source, local_path)
    return local_path


# ── Google Cloud Storage ──────────────────────────────────────────────────────

def _gcs_client(config: dict):
    from google.cloud import storage
    creds = os.getenv(config.get("credentials_env", "GOOGLE_APPLICATION_CREDENTIALS"), "")
    if creds:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
    return storage.Client(project=config.get("project_id"))

def _gcs_upload(local_path: str, destination: str, config: dict) -> str:
    bucket_name = config.get("bucket", "")
    if not bucket_name:
        raise ValueError("GCS: 'bucket' is required.")
    client = _gcs_client(config)
    client.bucket(bucket_name).blob(destination).upload_from_filename(local_path)
    return f"gs://{bucket_name}/{destination}"

def _gcs_download(source: str, local_path: str, config: dict) -> str:
    bucket_name = config.get("bucket", "")
    if not bucket_name:
        raise ValueError("GCS: 'bucket' is required.")
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    client = _gcs_client(config)
    client.bucket(bucket_name).blob(source).download_to_filename(local_path)
    return local_path


FILE_STORAGES = {
    "local": {
        "upload":   _local_upload,
        "download": _local_download,
        "requires": "built-in (no install needed)",
    },
    "s3": {
        "upload":   _s3_upload,
        "download": _s3_download,
        "requires": "boto3 — pip install boto3",
    },
    "gcs": {
        "upload":   _gcs_upload,
        "download": _gcs_download,
        "requires": "google-cloud-storage — pip install google-cloud-storage",
    },
}


def _run_file_op(step: dict, state: dict, operation: str) -> dict:
    storage_key = step.get("storage", "local").lower()

    if storage_key not in FILE_STORAGES:
        return {
            **state, "status": "FAILED",
            "error": (
                f"[FileHandler] Unknown storage '{storage_key}'. "
                f"Available: {list(FILE_STORAGES.keys())}"
            ),
        }

    storage_cfg = FILE_STORAGES[storage_key]
    source      = step.get("source", "")
    destination = step.get("destination", "")

    try:
        if operation == "upload":
            logger.info("[FileHandler] Uploading %s → %s (%s)", source, destination, storage_key)
            result = storage_cfg["upload"](source, destination, step)
            msg    = f"[FileHandler] Uploaded → {result}"
            new_files = []
        else:
            logger.info("[FileHandler] Downloading %s → %s (%s)", source, destination, storage_key)
            result = storage_cfg["download"](source, destination, step)
            msg    = f"[FileHandler] Downloaded → {result}"
            new_files = [result]

        logger.info(msg)
        return {
            **state,
            "files_created":      state.get("files_created", []) + new_files,
            "logs":               state.get("logs", []) + [msg],
            "last_step_output":   msg,
            "current_step_index": state.get("current_step_index", 0) + 1,
            "error": None,
        }

    except ImportError as exc:
        req = storage_cfg.get("requires", "unknown")
        msg = f"[FileHandler] Missing package for '{storage_key}'. Install: {req}. Error: {exc}"
        logger.error(msg)
        return {**state, "status": "FAILED", "error": msg}

    except Exception as exc:
        msg = f"[FileHandler] {operation} failed: {exc}"
        logger.exception(msg)
        return {**state, "status": "FAILED", "error": msg}


class FileUploadHandler(BaseHandler):
    @property
    def name(self) -> str:
        return "file_upload"

    def execute(self, step: dict, state: dict) -> dict:
        return _run_file_op(step, state, "upload")


class FileDownloadHandler(BaseHandler):
    @property
    def name(self) -> str:
        return "file_download"

    def execute(self, step: dict, state: dict) -> dict:
        return _run_file_op(step, state, "download")


# Auto-register both
registry.register(FileUploadHandler())
registry.register(FileDownloadHandler())










