from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand


class IntToPtr(SaSSTransform):
    def apply(self, module):
        print("=== Start of IntToPtr Transformation ===")
        count = 0
        new_regs_count = {}
        for func in module.functions:
            for block in func.blocks:
                new_insts = []
                for inst in block.instructions:
                    if inst.IsGlobalLoad() or inst.IsGlobalStore():
                        ptr_idx = inst.useOpStartIdx

                        src_op = inst.operands[ptr_idx]

                        new_reg = src_op.Reg + "_to_ptr"
                        new_regs_count[new_reg] = new_regs_count.get(new_reg, 0) + 1
                        new_reg = f"{new_reg}_{new_regs_count[new_reg]}"
                        dst_op = Operand(new_reg, new_reg, None, None, True, False, True)

                        inst_content = f"INTTOPTR {dst_op.Name}, {src_op.Name}"
                        cast_inst = Instruction(
                            id=f"{inst.id}",
                            opcodes=["INTTOPTR"],
                            operands=[
                                dst_op,
                                Operand(src_op.Name, src_op.Reg, src_op._Suffix,
                                        src_op._ArgOffset, True, False, False)
                            ],
                            inst_content=inst_content,
                            parentBB=inst.parent
                        )
                        new_insts.append(cast_inst)
                        count += 1

                        inst.operands[ptr_idx] = dst_op
                    new_insts.append(inst)
                block.instructions = new_insts

        print(f"Total INTTOPTR instructions added: {count}")
        print(f"=== End of IntToPtr Transformation ===")
