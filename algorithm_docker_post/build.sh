#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKER_TAG="toothfairy3-multiinstance-algorithm"

# Usage: ./build.sh [--local]
#   (no flag)  -> CHALLENGE build (default, no UID/GID args)
#   --local    -> LOCAL build with host UID/GID for easier file permissions

MODE="challenge"
for arg in "$@"; do
    [[ "$arg" == "--local" ]] && MODE="local"
done

if [[ "$MODE" == "local" ]]; then
    HOST_UID=$(id -u)
    HOST_GID=$(id -g)
    echo "Building LOCAL Docker $DOCKER_TAG image with UID=$HOST_UID and GID=$HOST_GID..."
    docker build "$SCRIPTPATH" \
        --platform=linux/amd64 \
        --quiet \
        --build-arg HOST_UID=$HOST_UID \
        --build-arg HOST_GID=$HOST_GID \
        --tag $DOCKER_TAG
else
    echo "Building CHALLENGE Docker $DOCKER_TAG image..."
    docker build "$SCRIPTPATH" \
        --platform=linux/amd64 \
        --quiet \
        --tag $DOCKER_TAG
fi

########################################
if [ $? -eq 0 ]; then
    echo "Docker image built successfully: $DOCKER_TAG (mode=$MODE)"
else
    echo "Error: Docker build failed with exit code $?"
    exit 1
fi
