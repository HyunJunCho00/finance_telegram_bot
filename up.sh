#!/bin/bash
PROJECT=mcpknu-493907
for KEY in EXECUTION_DB_URL REDIS_URL AVNADMIN_PASSWORD; do
  VALUE=$(grep "^${KEY}=" .env | cut -d'=' -f2-)
  gcloud secrets create $KEY --project=$PROJECT --replication-policy=automatic 2>/dev/null
  echo -n "$VALUE" | gcloud secrets versions add $KEY --project=$PROJECT --data-file=-
  echo "OK: $KEY"
done
