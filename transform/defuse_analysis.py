from transform.transform import SaSSTransform
from collections import deque

class DefUseAnalysis(SaSSTransform):
    def __init__(self):
        super().__init__()
        
    def apply(self, module):
        print("=== Start of DefUseAnalysis ===")
        
        for func in module.functions:
            self.build_def_use(func.blocks)

        print("=== End of DefUseAnalysis ===")
    
    def build_def_use(self, blocks):
        # Clear previous results
        for block in blocks:
            for inst in block.instructions:
                inst.reaching_defs = {}
                inst.ReachingDefsSet = {}
                inst.users = {}
                for op in inst.operands:
                    op.defining_insts = set()


        # Build the def-use chains
        in_defs = {bb: {} for bb in blocks}
        queue = deque(blocks)

        while queue:
            bb = queue.popleft()

            curr_defs = {reg: set(defs) for reg, defs in in_defs[bb].items()}

            bb.InDefs = {reg: set(defs) for reg, defs in curr_defs.items()}
            bb.GenDefs = {}

            for inst in bb.instructions:

                for use_op in inst.get_uses_with_predicate():
                    if not use_op.reg:
                        continue

                    # Each entry in CurrDefs is (DefInst, DefOp)
                    for def_inst, def_op in curr_defs.get(use_op.reg, set()):
                        def_inst.users.setdefault(def_op, set()).add((inst, use_op))
                        use_op.defining_insts.add(def_inst)
                        inst.reaching_defs[use_op] = def_inst
                        inst.ReachingDefsSet.setdefault(use_op, set()).add((def_inst, def_op))

                def_ops = inst.get_defs()
                for def_op in def_ops:
                    if def_op and def_op.reg:
                        if not def_op.is_writable_reg:
                            continue
                        # Track which specific def operand defined the register
                        curr_defs[def_op.reg] = {(inst, def_op)}
                        bb.GenDefs[inst] = def_op

            bb.OutDefs = {reg: set(defs) for reg, defs in curr_defs.items()}

            for succ_bb in bb.succs:
                changed = False
                for reg, defs in curr_defs.items():
                    succ_defs = in_defs[succ_bb].get(reg, set())
                    if not defs.issubset(succ_defs):
                        in_defs[succ_bb][reg] = succ_defs | defs
                        changed = True

                if changed:
                    queue.append(succ_bb)
