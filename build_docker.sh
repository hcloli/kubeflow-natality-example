#!/bin/bash

PROJECT=ai-roadmap-new
IMAGE=kubeflow-natality
VERSION=0.0.1

docker build -t gcr.io/${PROJECT}/${IMAGE}:${VERSION} .
docker push gcr.io/${PROJECT}/${IMAGE}:${VERSION}