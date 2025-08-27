from transform.opaggregate import OperAggregate
from transform.typeanalysis import TypeAnalysis
from transform.ssa import SSA
from transform.xmad_to_imad import XmadToImad
from transform.sr_substitute import SRSubstitute
from transform.inttoptr import IntToPtr
from transform.pack64 import Pack64
from transform.defuse_analysis import DefUseAnalysis
from transform.mov_eliminate import MovEliminate
from transform.dce import DCE
from transform.fp_hack import FPHack
from transform.set_zero import SetZero

class Transforms:
    def __init__(self, name):
        self.name = name
        self.passes = []

        # Add int32 to int64 pass
        self.passes.append(Pack64("pack64"))
        # Add passes
        self.passes.append(SSA("SSA"))
        # Add special register substitution pass
        self.passes.append(SRSubstitute("SR Substitute"))
        # Add def-use analysis pass
        self.passes.append(DefUseAnalysis("def-use analysis"))
        # Add xmad to mul64 pass
        self.passes.append(XmadToImad("xmad to mul64"))
        # Add FP hack pass
        self.passes.append(FPHack("FP hack"))
        # Add set zero pass
        self.passes.append(SetZero("set zero"))
        # Add passes
        # self.passes.append(SSA("SSA"))
        # Add def-use analysis pass
        self.passes.append(DefUseAnalysis("def-use analysis"))
        # Add mov elimination pass
        self.passes.append(MovEliminate("mov elimination"))
        # Add def-use analysis pass again
        self.passes.append(DefUseAnalysis("def-use analysis"))
        # Add operator aggregation pass
        self.passes.append(OperAggregate("operator aggregation"))
        # Add SSA pass again
        self.passes.append(SSA("SSA"))
        # Add int to ptr pass
        self.passes.append(IntToPtr("int to ptr"))
        # Add def-use analysis pass again
        self.passes.append(DefUseAnalysis("def-use analysis"))
        # Add dead code elimination pass
        self.passes.append(DCE("dead code elimination"))
        # Add SSA pass again
        self.passes.append(SSA("SSA"))
        # Add type analysis pass
        self.passes.append(TypeAnalysis("type analysis"))

    def apply(self, module):

        # print instructions
        for func in module.functions:
            print(f"Function: {func.name}")
            for block in func.blocks:
                print(f"  Block: {block.addr_content}", end="")
                print(f" from: [", end="")
                for pred in block._preds:
                    print(f"{pred.addr_content},", end="")
                print(f"]", end="")
                print(f" to: [", end="")
                for succ in block._succs:
                    print(f"{succ.addr_content},", end="")
                print(f"]")
                for inst in block.instructions:
                    print(f"    {inst.id}    {inst}")
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
