import clearml


class ClearMLLogger:

    def __init__(
        self,
        project_name: str,
        task_name: str,
        task_type=clearml.Task.TaskTypes.training,
        params: dict = None,
    ):
        self.task = clearml.Task.init(
            project_name=project_name, task_name=task_name, task_type=task_type
        )
        if params:
            self.task.connect(params)
        self.logger = clearml.Logger.current_logger()

    def log_scalar(self, title: str, series: str, value: float, iteration: int):
        self.logger.report_scalar(
            title=title, series=series, value=value, iteration=iteration
        )

    def upload_artifact(self, name: str, object_file):
        self.task.upload_artifact(name, object_file)

    def close(self):
        self.task.close()
