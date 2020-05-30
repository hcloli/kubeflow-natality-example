#!/bin/bash

PROJECT=ai-roadmap-new
IMAGE=kubeflow-natality-test
VERSION=0.0.6

docker build -t gcr.io/${PROJECT}/${IMAGE}:${VERSION} .
docker push gcr.io/${PROJECT}/${IMAGE}:${VERSION}