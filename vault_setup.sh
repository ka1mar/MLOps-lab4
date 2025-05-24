#!/bin/bash
set -e

VAULT_ADDR=${VAULT_ADDR}
VAULT_TOKEN=${VAULT_DEV_ROOT_TOKEN_ID}
HOST=${HOST}
PORT=${PORT}
DBNAME=${DBNAME}
USER=${USER}
PASSWORD=${PASSWORD}
KAFKA_GROUP=${KAFKA_GROUP}


echo "Enabling KV secrets engine at path 'db'..."
curl -s \
  --header "X-Vault-Token: $VAULT_TOKEN" \
  --request POST \
  --data '{"type": "kv"}' \
  $VAULT_ADDR/v1/sys/mounts/db || {
    echo "The KV secrets engine might already be enabled at path 'db'"
  }

echo "Enabling KV secrets engine at path 'kafka'..."
curl -s \
  --header "X-Vault-Token: $VAULT_TOKEN" \
  --request POST \
  --data '{"type": "kv"}' \
  $VAULT_ADDR/v1/sys/mounts/kafka || {
    echo "The KV secrets engine might already be enabled at path 'kafka'"
  }

echo "Storing database credentials..."
curl -s \
  --header "X-Vault-Token: $VAULT_TOKEN" \
  --request POST \
  --data "{\"host\":\"$HOST\", \"port\":\"$PORT\", \"dbname\":\"$DBNAME\", \"user\":\"$USER\", \"password\":\"$PASSWORD\"}" \
  $VAULT_ADDR/v1/db/credentials

echo "Storing kafka credentials..."
curl -s \
  --header "X-Vault-Token: $VAULT_TOKEN" \
  --request POST \
  --data "{\"group\":\"$KAFKA_GROUP\"}" \
  $VAULT_ADDR/v1/kafka/credentials

echo "Verifying db secrets..."
RESPONSE=$(curl -s \
  --header "X-Vault-Token: $VAULT_TOKEN" \
  $VAULT_ADDR/v1/db/credentials)

if echo "$RESPONSE" | grep -q "\"host\":\"$HOST\""; then
  echo "Success! Database credentials stored in Vault."
else
  echo "Warning: Could not verify if db credentials were stored correctly."
  echo "Response: $RESPONSE"
fi

echo "Verifying kafka secrets..."
RESPONSE=$(curl -s \
  --header "X-Vault-Token: $VAULT_TOKEN" \
  $VAULT_ADDR/v1/kafka/credentials)

if echo "$RESPONSE" | grep -q "\"group\":\"$KAFKA_GROUP\""; then
  echo "Success! Kafka credentials stored in Vault."
else
  echo "Warning: Could not verify if kafka credentials were stored correctly."
  echo "Response: $RESPONSE"
fi

echo "Vault setup completed."