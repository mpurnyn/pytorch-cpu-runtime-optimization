$containerName = "pytorch"

# Build the docker image
docker build -t $containerName .

# Remove container with the same name if it exists
if (docker ps -aq -f name=$containerName) {
    docker rm -f $containerName
}

# Run the docker container
docker run --name $containerName -d `
  -v ${PWD}:/opt/pytorch `
  pytorch

# Execute the script within the running container
docker exec $containerName /bin/bash -c "./src/build_and_execute.sh"