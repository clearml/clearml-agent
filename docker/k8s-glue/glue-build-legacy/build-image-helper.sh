#!/bin/bash

# Check if image name and Dockerfile path are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <image_name> <dockerfile_path> <build_context>"
    exit 1
fi

# Build the Docker image
image_name=$1
dockerfile_path=$2
build_context=$3

if [ $build_context == "glue-build-aws" ] || [ $build_context == "glue-build-gcp" ]; then
    if [ ! -f $build_context/clearml.conf ]; then
        cp build-resources/clearml.conf $build_context
    fi
    if [ ! -f $build_context/entrypoint.sh ]; then
        cp build-resources/entrypoint.sh $build_context
        chmod +x $build_context/entrypoint.sh
    fi
    if [ ! -f $build_context/setup.sh ]; then
        cp build-resources/setup.sh $build_context
        chmod +x $build_context/setup.sh
    fi
fi
cp ../../examples/k8s_glue_example.py $build_context

docker build -f $dockerfile_path -t $image_name $build_context

# cleanup
if [ $build_context == "glue-build-aws" ] || [ $build_context == "glue-build-gcp" ]; then
    rm $build_context/clearml.conf
    rm $build_context/entrypoint.sh
    rm $build_context/setup.sh
fi
rm $build_context/k8s_glue_example.py
