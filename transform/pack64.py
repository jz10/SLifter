from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand


class Pack64(SaSSTransform):
    def apply(self, module):
        print("=== Start of Pack64 Transformation ===")

        count = 0
        for func in module.functions:
            for block in func.blocks:
                new_insts = []
                for inst in block.instructions:
                    if (inst.IsLoad() or inst.IsStore()) and 'E' in inst.opcodes:
                        
                        if inst.IsLoad():
                            ptr_idx = 1
                        else:
                            ptr_idx = 0

                        src_op_lower = Operand(inst.operands[ptr_idx].Name, 
                                               inst.operands[ptr_idx].Reg,
                                               inst.operands[ptr_idx]._Suffix,
                                               inst.operands[ptr_idx]._ArgOffset,
                                               True, False, False)
                        
                        src_op_upper_name = 'R' + str(int(inst.operands[ptr_idx].Reg[1:]) + 1)
                        src_op_upper = Operand(src_op_upper_name,
                                               src_op_upper_name,
                                               src_op_lower._Suffix,
                                               src_op_lower._ArgOffset,
                                               True, False, False) 

                        new_reg = src_op_lower.Name + "_int64"
                        dst_op = Operand(new_reg, new_reg, None, -1, True, False, False)

                        inst_content = f"PACK64 {dst_op.Name}, {src_op_lower.Name} {src_op_upper.Name}"
                        cast_inst = Instruction(
                            id=f"{inst.id}_pack64",
                            opcodes=["PACK64"],
                            operands=[
                                dst_op,
                                src_op_lower,
                                src_op_upper
                            ],
                            inst_content=inst_content,
                            parentBB=inst.parent
                        )
                        count += 1
                        new_insts.append(cast_inst)

                        inst.operands[ptr_idx]._Name = new_reg
                        inst.operands[ptr_idx]._Reg = new_reg
                    new_insts.append(inst)
                block.instructions = new_insts

        print(f"Total PACK64 instructions added: {count}")
        print("=== End of Pack64 Transformation ===")
