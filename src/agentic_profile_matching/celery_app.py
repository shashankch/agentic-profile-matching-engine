from celery import Celery
from agentic_profile_matching import config

celery_app = Celery(
    "agentic_profile_matching",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
    include=["agentic_profile_matching.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    result_expires=3600,
)
