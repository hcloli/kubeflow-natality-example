#!/bin/bash

PROJECT=ai-roadmap-new
IMAGE=kubeflow-natality-test
VERSION=0.0.5

docker build -t gcr.io/${PROJECT}/${IMAGE}:${VERSION} .
docker push gcr.io/${PROJECT}/${IMAGE}:${VERSION}