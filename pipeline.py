from clearml import Task
from clearml.automation import PipelineController


def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(
    name="Pipeline demo", project="examples", version="0.0.1", add_pipeline_tags=False
)

# pipe.add_parameter(
#     "url",
#     "https://files.community.clear.ml/examples%252F.pipelines%252FPipeline%20demo/stage_data.8f17b6316ce442ce8904f6fccb1763de/artifacts/dataset/f6d08388e9bc44c86cab497ad31403c4.iris_dataset.pkl",
#     "dataset_url",
# )

pipe.add_parameter(
    name="columns"   
)

pipe.add_parameter(
    name="folds"   
)

pipe.add_parameter(
    name="score"   
)


pipe.set_default_execution_queue("default")

pipe.add_step(
    name="Raw Data Collection",
    base_task_project="Iris",
    base_task_name="Raw Data Collection"
)

pipe.add_step(
    name="Data Preparation",
    parents=["Raw Data Collection"],
    base_task_project="Iris",
    base_task_name="Data Preparation",
    pre_execute_callback=pre_execute_callback_example,
    post_execute_callback=post_execute_callback_example,
)
pipe.add_step(
    name="Train",
    parents=["Data Preparation"],
    base_task_project="Iris",
    base_task_name="Train"
)

# for debugging purposes use local jobs
# pipe.start_locally(run_pipeline_steps_locally=True)

# Starting the pipeline (in the background)
pipe.start(queue="default")

print("done")
