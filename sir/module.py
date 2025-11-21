from sir.function import Function

class Module :
    def __init__(self, name, parser):
        self.name = name
        # Propagate ISA string from parser for arch-specific transforms
        self.isa = getattr(parser, 'isa', None)

        # Parse the module
        self.parse_module(parser)

    def parse_module(self, parser):
        self.functions = parser.apply()
