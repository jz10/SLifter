from transform.transform import SaSSTransform
from collections import defaultdict, deque

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
                inst.Users = set()


        # Build the def-use chains
        InDefs = {BB : defaultdict(set) for BB in blocks}
        Queue = deque(blocks)

        while Queue:
            BB = Queue.popleft()

            CurrDefs = {r : set(defs) for r, defs in InDefs[BB].items()}

            for Inst in BB.instructions:

                for UseOp in Inst.GetUses():
                    if not UseOp.Reg:
                        continue

                    for Def in CurrDefs.get(UseOp.Reg, set()):
                        Def.Users.add((Inst, UseOp))
                        Inst.ReachingDefs[UseOp] = Def

                DefOp = Inst.GetDef()
                if DefOp and DefOp.Reg:
                    CurrDefs[DefOp.Reg] = {Inst}

            for SuccBB in BB._succs:
                Changed = False
                for r, defs in CurrDefs.items():
                    if not defs.issubset(InDefs[SuccBB][r]):
                        InDefs[SuccBB][r] |= defs
                        Changed = True  

                if Changed:
                    Queue.append(SuccBB)

        