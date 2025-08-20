#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKER_TAG="toothfairy3-multiinstance-algorithm"

### LOCAL VERSION

# HOST_UID=$(id -u)
# HOST_GID=$(id -g)

# echo "Building LOCAL Docker $DOCKER_TAG image with UID=$HOST_UID and GID=$HOST_GID..."

# docker build "$SCRIPTPATH" \
#     --platform=linux/amd64 \
#     --quiet \
#     --build-arg HOST_UID=$HOST_UID \
#     --build-arg HOST_GID=$HOST_GID \
#     --tag $DOCKER_TAG

### CHALLENGE VERSION

echo "Building CHALLENGE $DOCKER_TAG ToothFairy3 Multi-Instance-Segmentation algorithm Docker image..."

docker build "$SCRIPTPATH" \
    --platform=linux/amd64 \
    --quiet \
    --tag $DOCKER_TAG

if [ $? -eq 0 ]; then
    echo "Docker image built successfully: $DOCKER_TAG"
else
    echo "Error: Docker build failed with exit code $?"
    exit 1
fi