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
        defuse = DefUseAnalysis()
        defuse.apply(module)

        for func in module.functions:
            if not func.blocks:
                continue

            dom_tree = self.compute_dominator_tree(func)
            self.insert_phi_nodes(func, dom_tree)
            self.rename_variables(func, dom_tree)
            self.insert_set_zero_for_undefs(func)

        print("=== End of SSA ===")

    def compute_dominator_tree(self, function):
        if not function.blocks:
            return None

        entry_block = function.blocks[0]
        return DominatorTree(entry_block, function.blocks)

    def insert_phi_nodes(self, function, dom_tree):
        # Iterated dominance frontier algorithm on register defs.
        def_sites = defaultdict(set)
        all_vars = set()
        for block in function.blocks:
            for inst in block.instructions:
                # Only regs that are writable should be tracked
                for def_op in inst.get_defs():
                    if def_op and def_op.is_writable_reg:
                        var_name = self.get_original_reg(def_op.reg)
                        if var_name and not self.should_skip_variable(var_name, None, None):
                            def_sites[var_name].add(block)
                            all_vars.add(var_name)

        for var in all_vars:
            worklist = list(def_sites[var])
            phi_added_at = set()

            while worklist:
                block = worklist.pop(0)
                for df_block in dom_tree.dominance_frontier.get(block, []):
                    # if self.predicatesContradict(block, df_block):
                    #     continue

                    if df_block not in phi_added_at:
                        phi = self.create_phi_node(var, df_block)
                        # Insert PHI at the top of the block
                        df_block.instructions.insert(0, phi)
                        phi_added_at.add(df_block)
                        # Treat the PHI as a new def site of var
                        worklist.append(df_block)

    def should_skip_variable(self, var, reaching_defs, block):
        # Skip only special non-SSA globals: PT/UPT and zero registers RZ/URZ.
        # Do NOT skip predicate registers like P0/UP1, they should be in SSA.
        skip_set = {"PT", "UPT", "RZ", "URZ"}
        return var in skip_set

    def create_phi_node(self, reg, block):
        dest_op = Operand.from_reg(reg, reg)

        operands = [dest_op]
        # Allocate one incoming operand per predecessor (will be filled in rename)
        for _ in block.preds:
            src_op = Operand.from_reg(reg, reg)
            operands.append(src_op)

        phi = Instruction(
            id=f"phi_{reg}_{block.addr_content}",
            opcodes=["PHI"],
            operands=operands,
            parentBB=block
        )
        return phi

    def rename_variables(self, function, dom_tree):
        self.var_stacks = defaultdict(list)
        self.ssa_counter = defaultdict(int)
        self.undefined_ssa = {}
        self.entry_block = dom_tree.entry if dom_tree else None

        # Start from the dom-tree entry, not function.blocks[0] (in case they differ)
        self.rename_block(dom_tree.entry, dom_tree)

    def rename_block(self, block, dom_tree):
        pushed = defaultdict(list)

        # First: give names to this block's PHI destinations
        self.process_phi_nodes(block, pushed)   
        # Then: rename uses/defs within the block
        self.process_instructions(block, pushed)

        # After seeing the final names for this block, fill successor PHIs
        self.update_successor_phis(block)

        # Recurse over dominance children
        for child in dom_tree.get_children(block):
            self.rename_block(child, dom_tree)
        # Finally: pop the names we pushed when exiting this block's scope
        self.restore_stacks(pushed)

    def process_phi_nodes(self, block, pushed):
        for inst in block.instructions:
            if not inst.is_phi():
                continue

            dest_op = inst.operands[0]
            if dest_op.is_writable_reg:
                original_reg = self.get_original_reg(dest_op.reg)
                new_name = self.generate_ssa_name(original_reg)

                dest_op.set_reg(new_name)
                dest_op.ir_reg_name = None

                self.var_stacks[original_reg].append(new_name)
                pushed[original_reg].append(new_name)

    def process_instructions(self, block, pushed):
        for inst in block.instructions:
            if inst.is_phi():
                continue

            # Rename uses (including predicate flags)
            for use_op in inst.get_uses_with_predicate():
                if not use_op.is_writable_reg:
                    continue

                original_reg = self.get_original_reg(use_op.reg)
                current_name = self.get_current_name_or_version0(original_reg)
                use_op.set_reg(current_name)
                use_op.ir_reg_name = None

            # Rename defs
            for def_op in inst.get_defs():
                if def_op and def_op.is_writable_reg:
                    original_reg = self.get_original_reg(def_op.reg)
                    new_name = self.generate_ssa_name(original_reg)

                    def_op.set_reg(new_name)
                    def_op.ir_reg_name = None

                    self.var_stacks[original_reg].append(new_name)
                    pushed[original_reg].append(new_name)

    def update_successor_phis(self, block):
        for succ in block.succs:
            # Find our position in succ's predecessor list
            pred_idx = succ.preds.index(block)

            for inst in succ.instructions:
                if not inst.is_phi():
                    continue

                dest_op = inst.operands[0]
                original_reg = self.get_original_reg(dest_op.reg)

                if pred_idx + 1 < len(inst.operands):
                    phi_op = inst.operands[pred_idx + 1]
                    current_name = self.get_current_name_or_version0(original_reg)
                    phi_op.set_reg(current_name)
                    phi_op.ir_reg_name = None

    def restore_stacks(self, pushed):
        for var, names in pushed.items():
            for _ in names:
                if self.var_stacks[var]:
                    self.var_stacks[var].pop()

    def get_original_reg(self, reg_name):
        if not reg_name:
            return None
        if '@' in reg_name:
            return reg_name.split('@')[0]
        return reg_name

    def generate_ssa_name(self, original_reg):
        if not original_reg:
            return None
        self.ssa_counter[original_reg] += 1
        return f"{original_reg}@{self.ssa_counter[original_reg]}"

    def get_current_name_or_version0(self, original_reg):
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

    def insert_set_zero_for_undefs(self, function):
        

        entry_block = function.blocks[0]

        # Insert after phi
        insert_idx = 0
        while insert_idx < len(entry_block.instructions) and entry_block.instructions[insert_idx].is_phi():
            insert_idx += 1

        new_insts = []
        for original_reg in sorted(self.undefined_ssa):
            ssa_name = self.undefined_ssa[original_reg]
            dest_op = Operand.from_reg(original_reg, original_reg)
            dest_op.set_reg(ssa_name)
            dest_op.ir_reg_name = None

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
