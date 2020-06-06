# Kubeflow Pipeline Example

This repository contains an example of a KubeFlow pipeline that trains a model to predict babies 
weight according to pre-birth data about the mother and the pregnancy.

The aim of this repository is to show how to use KubeFlow Pipelines. 

## Build & Deploy Instructions
### Pre-requisites
This code assumes GCP project. Therefore, a service account is required for the code to run. Also, it assumes
your docker environment is configured to push docker images to the project. You can configure your docker 
using `gcloud` tool:
```commandline
gcloud auth configure-docker
```
### Docker Image Build.
1. Edit the `Dockerfile` file, line 6 to state your account service file.
2. Edit the `build_docker.sh` file, line 3 and set your GCP project name.
3. Run the `build_docker.sh`.
### Create a pipeline YAML
Run the following command:
```commandline
python pipeline/natality_pipeline.sh
```
### Upload Pipeline
Upload the newly created YAML file `pipeline/natality_pipeline.py.yaml` to Kubeflow using the 
Kubeflow Pipelines UI.

![Natality Pipeline](https://github.com/hcloli/kubeflow-natality-example/blob/master/blob/natality-pipeline.png?raw=true)