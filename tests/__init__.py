import os
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")  # unit tests must not depend on Redis or share buckets
