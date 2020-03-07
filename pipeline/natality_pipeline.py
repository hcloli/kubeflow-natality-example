import kfp
from kfp import dsl

TRAIN_IMAGE_TAG = "0.0.6"
TEST_IMAGE_TAG = "0.0.5"
ROC_CURVE_IMAGE_TAG = "0.0.9"
CONF_MATRIX_IMAGE_TAG = "0.0.9"


def train_op(max_depth, drop_features, input_bucket, output_bucket):
    return dsl.ContainerOp(
        name="train",
        image=f'gcr.io/ai-roadmap-new/kubeflow-natality-train:{TRAIN_IMAGE_TAG}',
        arguments=[
            '--max_depth', max_depth,
            '--drop_features', drop_features,
            '--input_bucket', input_bucket,
            '--output_bucket', output_bucket
        ],
        file_outputs={
            'train_output_path': '/tmp/train_output_path.txt'
        }
    )


def test_op(drop_features, train_output_path, output_bucket, input_bucket):
    return dsl.ContainerOp(
        name="test",
        image=f'gcr.io/ai-roadmap-new/kubeflow-natality-test:{TEST_IMAGE_TAG}',
        arguments=[
            "--train_output_path", train_output_path,
            "--drop_features", drop_features,
            "--output_bucket", output_bucket,
            "--input_bucket", input_bucket
        ],
        file_outputs={
            'test_output_path': '/tmp/test_output_path.txt'
        }
    )


def roc_curve_op(test_output_path, train_output_path, output_bucket, input_bucket):
    return dsl.ContainerOp(
        name="roc-curve-calc",
        image=f'gcr.io/ai-roadmap-new/kubeflow-natality-roc-curve:{ROC_CURVE_IMAGE_TAG}',
        arguments=[
            "--test_output_path", test_output_path,
            '--train_output_path', train_output_path,
            "--output_bucket", output_bucket,
            "--input_bucket", input_bucket
        ],
        file_outputs={
            'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json',
            'mlpipeline-metrics': '/mlpipeline-metrics.json'
        }
    )


def conf_matrix_op(test_output_path, input_bucket, output_bucket):
    return dsl.ContainerOp(
        name="conf-matrix-calc",
        image=f'gcr.io/ai-roadmap-new/kubeflow-natality-conf-matrix:{CONF_MATRIX_IMAGE_TAG}',
        arguments=[
            "--test_output_path", test_output_path,
            "--output_bucket", output_bucket,
            "--input_bucket", input_bucket
        ],
        file_outputs={
            'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json',
            'mlpipeline-metrics': '/mlpipeline-metrics.json'
        }
    )


@dsl.pipeline(
    name="Baby Weight Research Pipeline",
    description="Predict baby weight category"
)
def message_pipeline(
        max_depth: int = 3,
        drop_features: str = 'None',
        input_bucket: str = "ai-roadmap-new-data",
        output_bucket: str = "ai-roadmap-new-experiments-output"
):
    train_task = train_op(max_depth=max_depth,
                          drop_features=drop_features,
                          input_bucket=input_bucket,
                          output_bucket=output_bucket)
    train_task.set_memory_request("2G")
    train_task.set_cpu_request("1")

    test_task = test_op(train_output_path=train_task.outputs['train_output_path'],
                        drop_features=drop_features,
                        input_bucket=input_bucket,
                        output_bucket=output_bucket)
    test_task.after(train_task)

    roc_curve_task = roc_curve_op(test_output_path=test_task.outputs['test_output_path'],
                                  train_output_path=train_task.outputs['train_output_path'],
                                  input_bucket=input_bucket,
                                  output_bucket=output_bucket)
    roc_curve_task.after(test_task)

    conf_matrix_task = conf_matrix_op(test_output_path=test_task.outputs['test_output_path'],
                                      input_bucket=input_bucket,
                                      output_bucket=output_bucket)
    conf_matrix_task.after(test_task)


if __name__ == '__main__':
    export_file = __file__ + ".yaml"
    print(f"Exporting pipeline to {export_file}")
    kfp.compiler.Compiler().compile(message_pipeline, export_file)
    print(f"Done")
