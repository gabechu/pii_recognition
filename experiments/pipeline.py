from dagster import solid, pipeline


@solid
def register_recognisers():
    ...


@solid
def get_recogniser():
    ...


@solid
def initialise_recogniser():
    ...


@solid
def get_evaluation_data():
    ...


@solid
def initialise_evaluator():
    ...


@solid
def evaluate_and_logging():
    ...


@pipeline
def evaluation_pipeline():
    ...