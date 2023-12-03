#!/bin/bash

set -o errexit

# Build the docker image
docker build -t pytorch .

container_name="pytorch"
# remove container with the same name if it exists
if [ "$(docker ps -aq -f name=$container_name)" ]; then
  docker rm -f $container_name
fi

docker run --name $container_name -d \
  -v $(pwd):/opt/pytorch \
  pytorch

docker exec $container_name  ./src/build_and_execute.sh
