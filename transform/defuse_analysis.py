from transform.transform import SaSSTransform
from collections import deque

class DefUseAnalysis(SaSSTransform):
    def __init__(self, name):
        super().__init__(name)
        
    def apply(self, module):
        print("=== Start of DefUseAnalysis ===")
        
        for func in module.functions:
            self.BuildDefUse(func.blocks)

        print("=== End of DefUseAnalysis ===")
    
    def BuildDefUse(self, blocks):
        # Clear previous results
        for block in blocks:
            for inst in block.instructions:
                inst.ReachingDefs = {}
                inst.ReachingDefsSet = {}
                inst.Users = {}


        # Build the def-use chains
        InDefs = {BB : {} for BB in blocks}
        Queue = deque(blocks)

        while Queue:
            BB = Queue.popleft()

            CurrDefs = {r : set(defs) for r, defs in InDefs[BB].items()}

            BB.InDefs = {r : set(defs) for r, defs in CurrDefs.items()}
            BB.GenDefs = {}

            for Inst in BB.instructions:

                for UseOp in Inst.GetUsesWithPredicate():
                    if not UseOp.Reg:
                        continue

                    # Each entry in CurrDefs is (DefInst, DefOp)
                    for DefInst, DefOp in CurrDefs.get(UseOp.Reg, set()):
                        DefInst.Users.setdefault(DefOp, set()).add((Inst, UseOp))
                        Inst.ReachingDefs[UseOp] = DefInst
                        Inst.ReachingDefsSet.setdefault(UseOp, set()).add((DefInst, DefOp))

                DefOps = Inst.GetDefs()
                for DefOp in DefOps:
                    if DefOp and DefOp.Reg:
                        if not DefOp.IsWritableReg:
                            continue
                        # Track which specific def operand defined the register
                        CurrDefs[DefOp.Reg] = {(Inst, DefOp)}
                        BB.GenDefs[Inst] = DefOp

            BB.OutDefs = {r : set(defs) for r, defs in CurrDefs.items()}

            for SuccBB in BB._succs:
                Changed = False
                for r, defs in CurrDefs.items():
                    succ_defs = InDefs[SuccBB].get(r, set())
                    if not defs.issubset(succ_defs):
                        InDefs[SuccBB][r] = succ_defs | defs
                        Changed = True

                if Changed:
                    Queue.append(SuccBB)