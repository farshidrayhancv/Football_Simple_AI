#!/bin/bash
# run_docker.sh - Convenience script to run the Docker container

# Allow X11 forwarding for GUI support
xhost +local:docker

# Run docker compose
docker-compose up -d

# Execute bash in the container
docker-compose exec football-ai bash

# When done, remove X11 permission
xhost -local:docker
