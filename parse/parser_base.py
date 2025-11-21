import re

from sir.basicblock import BasicBlock
from sir.controlcode import ControlCode
from sir.function import Function
from sir.instruction import Instruction
from sir.operand import InvalidOperandException, Operand


class NoParsingEffort(Exception):
    pass


class UnmatchedControlCode(Exception):
    pass


class SaSSParserBase:
    def __init__(self, isa, file):
        self.file = file

    def apply(self):
        funcs = []
        curr_func = None
        insts = []

        lines = self.file.split("\n")
        modified_lines = []
        skip_next = False
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
            if "{" in line:
                line = line.replace("{", "")
            if "}" in line:
                if line[-1] != "/":
                    line = line + lines[i + 1]
                    line = line.replace("\n", "")
                    skip_next = True
                line = line.replace("}", ";")
            modified_lines.append(line)

        lines = modified_lines

        for _, line in enumerate(lines):
            prev_func = curr_func
            curr_func = self.create_function(line, prev_func, insts)
            if prev_func != curr_func and curr_func is not None:
                if prev_func is not None:
                    funcs.append(prev_func)
                    insts = []
                continue
            curr_func = prev_func
            self.parse_func_body(line, insts, curr_func)

        if curr_func is not None:
            curr_func.blocks = self.create_cfg(insts)
            funcs.append(curr_func)

        return funcs

    def create_function(self, line, prev_func, insts):
        if "/*" not in line or "*/" not in line:
            if "Function : " in line:
                if prev_func is not None:
                    prev_func.blocks = self.create_cfg(insts)
                items = line.split(" : ")
                return Function(items[1])
        return None

    def parse_func_body(self, line, insts, curr_func):
        raise NoParsingEffort

    def get_inst_num(self, line):
        items = line.split("/*")
        return items[1]

    def get_inst_opcode(self, line):
        items = line.split(";")
        line = items[0].lstrip()
        items = line.split(" ")
        opcode = items[0]
        pflag = None

        pred_reg = None
        is_not = False
        if opcode.startswith("@"):
            opcode = opcode[1:]
        if opcode.startswith("!"):
            is_not = True
            opcode = opcode[1:]
        if opcode.startswith("P") and opcode[1].isdigit():
            pred_reg = opcode

        rest_content = line.replace(items[0], "")

        if pred_reg:
            prefix = "!" if is_not else None
            pflag = Operand.from_reg(pred_reg, pred_reg, prefix=prefix)
            opcode = items[1]
            rest_content = rest_content.replace(items[1], "")

        return opcode, pflag, rest_content

    def get_inst_operands(self, line):
        items = line.split(",")
        ops = []
        for item in items:
            operand = item.lstrip()
            if operand != "":
                ops.append(operand)
        return ops

    def parse_instruction(
        self, inst_id, opcode_content, pflag, operands_content, operands_detail, curr_func
    ):
        opcodes = opcode_content.split(".")

        operands = []
        for operand_content in operands_content:
            operands.append(Operand.parse(operand_content))

        inst = Instruction(inst_id, opcodes, operands, None, pflag)
        raw_content = opcode_content
        if operands_detail:
            raw_content = f"{opcode_content} {operands_detail}"
        inst.inst_content = raw_content
        return inst

    def get_arg_offset(self, offset):
        offset = offset.replace("[", "")
        offset = offset.replace("]", "")
        return int(offset, base=16)

    def parse_control_code(self, content, control_codes):
        raise NoParsingEffort

    def create_cfg(self, insts):
        addr_set = {int(inst.id, 16) for inst in insts}
        addr_list = sorted(a for a in addr_set)

        def _align_up_to_inst(addr_hex):
            addr = int(addr_hex, 16)
            max_addr = addr_list[-1]

            while addr not in addr_set and addr <= max_addr:
                addr += 0x8
            return addr

        for inst in insts:
            if inst.is_branch():
                target_addr = _align_up_to_inst(inst.operands[-1].__str__().zfill(4))
                inst.operands[0].name = format(target_addr, "04x")
                inst.operands[0].immediate_value = target_addr

        leaders = set()
        predicated_leaders = set()
        curr_pred = None
        for i, inst in enumerate(insts):
            if inst.is_return() or inst.is_exit():
                if i + 1 < len(insts):
                    leaders.add(insts[i + 1].id)
            if inst.is_branch():
                leaders.add(inst.operands[0].name.zfill(4))
            if str(curr_pred) != str(inst.pflag):
                leaders.add(insts[i].id)
                if inst.predicated():
                    predicated_leaders.add(insts[i].id)
                curr_pred = inst.pflag

        blocks = []
        block_insts = []
        pflag = None
        prev_block = None
        next_block = {}
        block_by_addr = {}
        predicated_blocks = set()
        for inst in insts:
            if inst.id in leaders:
                if block_insts:
                    block_id = f"{int(block_insts[0].id, 16):04X}"

                    predicated_block = False
                    if block_insts[0].id in predicated_leaders:
                        predicated_block = True
                        pflag = block_insts[0].pflag.clone()
                        pbra_inst = Instruction(
                            id=f"{int(block_insts[0].id, 16):04X}",
                            opcodes=["PBRA"],
                            operands=[block_insts[0].pflag.clone()],
                            parentBB=None,
                            pflag=None,
                        )
                        block_insts.insert(0, pbra_inst)
                        block_insts[1].id = f"{int(block_insts[0].id, 16)+1:04X}"
                        for pred_inst in block_insts:
                            pred_inst.pflag = None

                    block = BasicBlock(block_id, pflag, block_insts)
                    pflag = None
                    blocks.append(block)

                    if prev_block:
                        next_block[prev_block] = block
                    prev_block = block

                    block_by_addr[block.addr_content] = block

                    if predicated_block:
                        predicated_blocks.add(block)

                block_insts = [inst]
            else:
                block_insts.append(inst)

        for block in blocks:
            terminator = block.get_terminator()

            if not terminator:
                succ_block = next_block[block]
                dest_op = Operand.from_immediate(
                    succ_block.addr_content, int(succ_block.addr_content, 16)
                )
                new_inst = Instruction(
                    id=f"{int(block.instructions[-1].id, 16)+1:04X}",
                    opcodes=["BRA"],
                    operands=[dest_op],
                )
                block.instructions.append(new_inst)

                block.add_succ(succ_block)
                succ_block.add_pred(block)
            elif terminator.is_branch():
                dest_op = terminator.get_uses()[0]
                succ_block = block_by_addr[dest_op.name]
                block.add_succ(succ_block)
                succ_block.add_pred(block)
            elif terminator.is_return() or terminator.is_exit():
                pass
            else:
                raise Exception("Unrecognized terminator")

        insert_block = {}
        for block in predicated_blocks:
            pbra_inst = block.instructions[0]
            new_block_insts = block.instructions[1:]
            new_block = BasicBlock(new_block_insts[0].id, None, new_block_insts)
            block.instructions = [pbra_inst]

            new_block.pflag = block.pflag
            block.pflag = None

            insert_block[block] = new_block
            block_by_addr[new_block.addr_content] = new_block

            new_block.succs = block.succs
            block.succs = []
            for succ in new_block.succs:
                succ.preds.remove(block)
                succ.add_pred(new_block)

            new_block.add_pred(block)
            block.add_succ(new_block)

            block.add_succ(next_block[block])
            next_block[block].add_pred(block)

            true_br_block = new_block
            false_br_block = next_block[block]
            pbra_inst.operands.append(
                Operand.from_immediate(
                    true_br_block.addr_content, int(true_br_block.addr_content, 16)
                )
            )
            pbra_inst.operands.append(
                Operand.from_immediate(
                    false_br_block.addr_content, int(false_br_block.addr_content, 16)
                )
            )

        old_blocks = blocks.copy()
        blocks = []
        for block in old_blocks:
            blocks.append(block)
            if block in insert_block:
                blocks.append(insert_block[block])
                
        # print("Predicate converted CFG:")
        # for block in Blocks:
        #     print(f"  Block: {block.addr_content}", end="")
        #     print(f" from: [", end="")
        #     for pred in block.preds:
        #         print(f"{pred.addr_content},", end="")
        #     print(f"]", end="")
        #     print(f" to: [", end="")
        #     for succ in block.succs:
        #         print(f"{succ.addr_content},", end="")
        #     print(f"]")
        #     for inst in block.instructions:
        #         print(f"    {inst.id}    {inst}")

        for block in blocks:
            for inst in block.instructions:
                inst.parent = block

        return blocks

    def check_and_add_target(self, curr_bb, target_addr, jump_targets):
        if target_addr > 0:
            if target_addr not in jump_targets:
                jump_targets[target_addr] = []
            jump_targets[target_addr].append(curr_bb)
