#!/bin/bash

PROJECT=ai-roadmap-new
IMAGE=kubeflow-natality-conf-matrix
VERSION=0.0.10

docker build -t gcr.io/${PROJECT}/${IMAGE}:${VERSION} .
docker push gcr.io/${PROJECT}/${IMAGE}:${VERSION}