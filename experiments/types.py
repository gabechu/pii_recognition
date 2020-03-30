from typing import TYPE_CHECKING

from dagster import PythonObjectDagsterType

from recognisers.recogniser_registry import RecogniserRegistry

if TYPE_CHECKING:
    RecogniserRegistryDT = RecogniserRegistry
else:
    RecogniserRegistryDT = PythonObjectDagsterType(RecogniserRegistry)
