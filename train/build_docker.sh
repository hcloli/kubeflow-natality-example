#!/bin/bash

PROJECT=ai-roadmap-new
IMAGE=kubeflow-natality-train
VERSION=0.0.9

docker build -t gcr.io/${PROJECT}/${IMAGE}:${VERSION} .
docker push gcr.io/${PROJECT}/${IMAGE}:${VERSION}