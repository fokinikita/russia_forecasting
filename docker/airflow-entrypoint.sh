#!/usr/bin/env bash
# Starts Airflow in one container: `airflow standalone` = API server / UI + scheduler + DAG processor
# + triggerer, LocalExecutor, SQLite metadata DB in the airflow_home volume (only run states,
# all results and task logs go to S3).
set -euo pipefail

: "${S3_BUCKET:?S3_BUCKET is not set, copy .env.example to .env}"
: "${S3_ACCESS_KEY:?S3_ACCESS_KEY is not set}"
: "${S3_SECRET_KEY:?S3_SECRET_KEY is not set}"
: "${AIRFLOW_ADMIN_PASSWORD:?AIRFLOW_ADMIN_PASSWORD is not set}"
export S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://s3.twcstorage.ru}"
export S3_REGION="${S3_REGION:-ru-1}"

AF=/opt/airflow-venv/bin
# standalone starts its components as `airflow ...`; appended, so `python` stays the models' environment
export PATH="$PATH:$AF"
mkdir -p "$AIRFLOW_HOME"

# S3 connection for remote task logs (JSON connection format)
export AIRFLOW_CONN_S3_STORAGE="$("$AF/python" - <<'EOF'
import json, os
print(json.dumps({
    "conn_type": "aws",
    "login": os.environ["S3_ACCESS_KEY"],
    "password": os.environ["S3_SECRET_KEY"],
    "extra": {
        "endpoint_url": os.environ["S3_ENDPOINT_URL"],
        "region_name": os.environ["S3_REGION"],
        "config_kwargs": {
            "s3": {"addressing_style": "path"},
            "request_checksum_calculation": "when_required",
            "response_checksum_validation": "when_required",
        },
    },
}))
EOF
)"
export AIRFLOW__LOGGING__REMOTE_LOGGING=True
export AIRFLOW__LOGGING__REMOTE_LOG_CONN_ID=s3_storage
export AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER="s3://${S3_BUCKET}/${S3_PREFIX:+${S3_PREFIX}/}airflow_logs"
export AIRFLOW__LOGGING__DELETE_LOCAL_LOGS=False

# login of the web UI: user admin, password from .env
"$AF/python" - <<'EOF'
import json, os
path = os.path.join(os.environ["AIRFLOW_HOME"], "passwords.json")
with open(path, "w") as f:
    json.dump({"admin": os.environ["AIRFLOW_ADMIN_PASSWORD"]}, f)
os.chmod(path, 0o600)
EOF

# keep the JWT secret stable across restarts (stored in the airflow_home volume)
SECRET_FILE="$AIRFLOW_HOME/jwt_secret"
[ -s "$SECRET_FILE" ] || "$AF/python" -c "import secrets; print(secrets.token_urlsafe(48))" > "$SECRET_FILE"
export AIRFLOW__API_AUTH__JWT_SECRET="$(cat "$SECRET_FILE")"

"$AF/airflow" db migrate
# new DAGs start active: the Monday schedule works right after the first deploy
exec "$AF/airflow" standalone
