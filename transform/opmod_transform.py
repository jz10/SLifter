from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand


class OpModTransform(SaSSTransform):

    def apply(self, module):
        print("=== Start of OpModeTransform ===")

        total_inserted = 0
        new_regs_count = {}

        for func in module.functions:
            for block in func.blocks:
                new_insts = []
                for inst in block.instructions:
                    inserted_here = 0
                    # Scan all operands; 
                    for op in inst.operands:
                        # Handle any memory operand with .X4 suffix
                        if op.IsMemAddr and op.Suffix == "X4":
                            # Create SHL tmp, base, 0x2
                            base_reg = op.Reg  # underlying register name (e.g., R7)

                            # Source operand (as a plain register, not a memory address)
                            src_op = Operand.fromReg(base_reg, base_reg)

                            # Unique destination register name
                            base_dst = f"{base_reg}_x4"
                            new_regs_count[base_dst] = new_regs_count.get(base_dst, 0) + 1
                            dst_name = f"{base_dst}_{new_regs_count[base_dst]}"
                            dst_op = Operand.fromReg(dst_name, dst_name)

                            # Immediate 2 (shift by 2 == multiply by 4)
                            imm_op = Operand.fromImmediate("0x2", 2)

                            shl_inst = Instruction(
                                id=f"{inst.id}_x4",
                                opcodes=["SHL"],
                                operands=[dst_op, src_op, imm_op],
                                parentBB=inst.parent
                            )

                            # Insert before the current instruction
                            new_insts.append(shl_inst)
                            inserted_here += 1
                            total_inserted += 1

                            # Rewrite memory operand to use the new register (and drop suffix)
                            op.SetReg(dst_name)
                            op.Suffix = None

                    new_insts.append(inst)

                block.instructions = new_insts

        self.total_x4_expansions = total_inserted
        print(f"Total .X4 expansions inserted: {total_inserted}")
        print("=== End of OpModeTransform ===")
