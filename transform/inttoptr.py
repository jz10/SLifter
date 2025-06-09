from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand


class IntToPtr(SaSSTransform):
    def apply(self, module):
        for func in module.functions:
            for block in func.blocks:
                new_insts = []
                for inst in block.instructions:
                    if inst.IsLoad() or inst.IsStore():
                        if inst.IsLoad():
                            ptr_idx = 1
                        else:
                            ptr_idx = 0

                        src_op = inst.operands[ptr_idx]

                        new_reg = src_op.Reg + "_to_ptr"
                        dst_op = Operand(new_reg, new_reg, None, -1, True, False, True)

                        inst_content = f"INTTOPTR {dst_op.Name}, {src_op.Name}"
                        cast_inst = Instruction(
                            id=f"{inst.id}",
                            opcodes=["INTTOPTR"],
                            operands=[
                                dst_op,
                                Operand(src_op.Name, src_op.Reg, src_op._Suffix,
                                        src_op._ArgOffset, True, False, False)
                            ],
                            inst_content=inst_content
                        )
                        new_insts.append(cast_inst)

                        inst.operands[ptr_idx] = dst_op
                    new_insts.append(inst)
                block.instructions = new_insts
