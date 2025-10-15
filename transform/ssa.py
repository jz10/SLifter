from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand
from collections import deque, defaultdict

from transform.defuse_analysis import DefUseAnalysis
from transform.domtree import DominatorTree

class SSA(SaSSTransform):

    def apply(self, module):
        print("=== Start of SSA ===")

        # Build def-use for potential downstream consumers
        defuse = DefUseAnalysis("def-use analysis for SSA")
        defuse.apply(module)

        for func in module.functions:
            if not func.blocks:
                continue

            dom_tree = self.computeDominatorTree(func)
            self.insertPhiNodes(func, dom_tree)
            self.renameVariables(func, dom_tree)
            self.insertSetZeroForUndefs(func)

        print("=== End of SSA ===")

    def computeDominatorTree(self, function):
        if not function.blocks:
            return None

        entry_block = function.blocks[0]
        return DominatorTree(entry_block, function.blocks)

    def insertPhiNodes(self, function, dom_tree):
        # Iterated dominance frontier algorithm on register defs.
        def_sites = defaultdict(set)
        all_vars = set()
        for block in function.blocks:
            for inst in block.instructions:
                # Only regs that are writable should be tracked
                for def_op in inst.GetDefs():
                    if def_op and def_op.IsWritableReg:
                        var_name = self.getOriginalReg(def_op.Reg)
                        if var_name and not self.shouldSkipVariable(var_name, None, None):
                            def_sites[var_name].add(block)
                            all_vars.add(var_name)

        for var in all_vars:
            worklist = list(def_sites[var])
            phi_added_at = set()

            while worklist:
                block = worklist.pop(0)
                for df_block in dom_tree.dominance_frontier.get(block, []):
                    if df_block not in phi_added_at:
                        phi = self.createPhiNode(var, df_block)
                        # Insert PHI at the top of the block
                        df_block.instructions.insert(0, phi)
                        phi_added_at.add(df_block)
                        # Treat the PHI as a new def site of var
                        worklist.append(df_block)

    def shouldSkipVariable(self, var, reaching_defs, block):
        # Skip only special non-SSA globals: PT/UPT and zero registers RZ/URZ.
        # Do NOT skip predicate registers like P0/UP1, they should be in SSA.
        skip_set = {"PT", "UPT", "RZ", "URZ"}
        return var in skip_set

    def createPhiNode(self, reg, block):
        dest_op = Operand(reg, reg, None, None, True, False, False)

        operands = [dest_op]
        # Allocate one incoming operand per predecessor (will be filled in rename)
        for _ in block._preds:
            src_op = Operand(reg, reg, None, None, True, False, False)
            operands.append(src_op)

        phi = Instruction(
            id=f"phi_{reg}_{block.addr_content}",
            opcodes=["PHI"],
            operands=operands,
            parentBB=block
        )
        return phi

    def renameVariables(self, function, dom_tree):
        self.var_stacks = defaultdict(list)
        self.ssa_counter = defaultdict(int)
        self.undefined_ssa = {}
        self.entry_block = dom_tree.entry if dom_tree else None

        # Start from the dom-tree entry, not function.blocks[0] (in case they differ)
        self.renameBlock(dom_tree.entry, dom_tree)

    def renameBlock(self, block, dom_tree):
        pushed = defaultdict(list)

        # First: give names to this block's PHI destinations
        self.processPhiNodes(block, pushed)

        # Then: rename uses/defs within the block
        self.processInstructions(block, pushed)

        # After seeing the final names for this block, fill successor PHIs
        self.updateSuccessorPhis(block)

        # Recurse over dominance children
        for child in dom_tree.getChildren(block):
            self.renameBlock(child, dom_tree)

        # Finally: pop the names we pushed when exiting this block's scope
        self.restoreStacks(pushed)

    def processPhiNodes(self, block, pushed):
        for inst in block.instructions:
            if not inst.IsPhi():
                continue

            dest_op = inst.operands[0]
            if dest_op.IsWritableReg:
                original_reg = self.getOriginalReg(dest_op.Reg)
                new_name = self.generateSSAName(original_reg)

                dest_op.SetReg(new_name)
                dest_op.IRRegName = None

                self.var_stacks[original_reg].append(new_name)
                pushed[original_reg].append(new_name)

    def processInstructions(self, block, pushed):
        for inst in block.instructions:
            if inst.IsPhi():
                continue

            # Rename uses (including predicate flags)
            for use_op in inst.GetUsesWithPredicate():
                if not use_op.IsWritableReg:
                    continue

                original_reg = self.getOriginalReg(use_op.Reg)
                current_name = self.getCurrentNameOrVersion0(original_reg)
                use_op.SetReg(current_name)
                use_op.IRRegName = None

            # Rename defs
            for def_op in inst.GetDefs():
                if def_op and def_op.IsWritableReg:
                    original_reg = self.getOriginalReg(def_op.Reg)
                    new_name = self.generateSSAName(original_reg)

                    def_op.SetReg(new_name)
                    def_op.IRRegName = None

                    self.var_stacks[original_reg].append(new_name)
                    pushed[original_reg].append(new_name)

    def updateSuccessorPhis(self, block):
        for succ in block._succs:
            # Find our position in succ's predecessor list
            pred_idx = succ._preds.index(block)

            for inst in succ.instructions:
                if not inst.IsPhi():
                    continue

                dest_op = inst.operands[0]
                original_reg = self.getOriginalReg(dest_op.Reg)

                if pred_idx + 1 < len(inst.operands):
                    phi_op = inst.operands[pred_idx + 1]
                    current_name = self.getCurrentNameOrVersion0(original_reg)
                    phi_op.SetReg(current_name)
                    phi_op.IRRegName = None

    def restoreStacks(self, pushed):
        for var, names in pushed.items():
            for _ in names:
                if self.var_stacks[var]:
                    self.var_stacks[var].pop()

    def getOriginalReg(self, reg_name):
        if not reg_name:
            return None
        if '@' in reg_name:
            return reg_name.split('@')[0]
        return reg_name

    def generateSSAName(self, original_reg):
        if not original_reg:
            return None
        self.ssa_counter[original_reg] += 1
        return f"{original_reg}@{self.ssa_counter[original_reg]}"

    def getCurrentNameOrVersion0(self, original_reg):
        if not original_reg:
            return None

        stack = self.var_stacks[original_reg]
        if stack:
            return stack[-1]

        zero_name = self.undefined_ssa.get(original_reg)
        if not zero_name:
            zero_name = f"{original_reg}@0"
            self.undefined_ssa[original_reg] = zero_name

        stack.append(zero_name)
        return zero_name

    def insertSetZeroForUndefs(self, function):
        

        entry_block = function.blocks[0]

        # Insert after phi
        insert_idx = 0
        while insert_idx < len(entry_block.instructions) and entry_block.instructions[insert_idx].IsPhi():
            insert_idx += 1

        new_insts = []
        for original_reg in sorted(self.undefined_ssa):
            ssa_name = self.undefined_ssa[original_reg]
            dest_op = Operand.fromReg(original_reg, original_reg)
            dest_op.SetReg(ssa_name)
            dest_op.IRRegName = None

            inst_id = f"setzero_{ssa_name}"
            if entry_block.addr_content:
                inst_id = f"{entry_block.addr_content}_{inst_id}"

            new_insts.append(
                Instruction(
                    id=inst_id,
                    opcodes=["SETZERO"],
                    operands=[dest_op],
                    parentBB=entry_block
                )
            )

        entry_block.instructions[insert_idx:insert_idx] = new_insts
        self.undefined_ssa = {}
