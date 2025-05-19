from transform.opaggregate import OperAggregate
from transform.typeanalysis import TypeAnalysis

class Transforms:
    def __init__(self, name):
        self.name = name
        self.passes = []

        # Add passes
        # Add operator aggregation pass
        self.passes.append(OperAggregate("operator aggregation"))
        # Add type analysis pass
        self.passes.append(TypeAnalysis("type analysis"))

    def apply(self, module):
        for tranpass in self.passes:
            tranpass.apply(module)
