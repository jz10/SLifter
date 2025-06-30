from transform.opaggregate import OperAggregate
from transform.typeanalysis import TypeAnalysis
from transform.ssa import SSA
from transform.xmad_to_imad import XmadToImad
from transform.sr_substitute import SRSubstitute
from transform.inttoptr import IntToPtr

class Transforms:
    def __init__(self, name):
        self.name = name
        self.passes = []

        # Add passes
        self.passes.append(SSA("SSA"))
        # Add operator aggregation pass
        self.passes.append(OperAggregate("operator aggregation"))
        # Add special register substitution pass
        self.passes.append(SRSubstitute("SR Substitute"))
        # Add xmad to mul64 pass
        self.passes.append(XmadToImad("xmad to mul64"))
        # Add SSA pass again
        self.passes.append(SSA("SSA"))
        # Add int to ptr pass
        self.passes.append(IntToPtr("int to ptr"))
        # Add SSA pass again
        self.passes.append(SSA("SSA"))
        # Add type analysis pass
        self.passes.append(TypeAnalysis("type analysis"))

    def apply(self, module):

        # print instructions
        for func in module.functions:
            print(f"Function: {func.name}")
            for block in func.blocks:
                print(f"  Block: {block.addr_content}")
                for inst in block.instructions:
                    print(f"    {inst}")
            print("")

            
        for tranpass in self.passes:
            tranpass.apply(module)

        # print instructions
        for func in module.functions:
            print(f"Function: {func.name}")
            for block in func.blocks:
                print(f"  Block: {block.addr_content}")
                for inst in block.instructions:
                    print(f"    {inst}")
            print("")

        print("done")
