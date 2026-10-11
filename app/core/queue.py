import os

import redis
from dotenv import load_dotenv
from rq import Queue

redis_conn = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=False,
)

ingestion_queue = Queue(
    "ingestion",
    connection=redis_conn,
    default_timeout=3600,
)