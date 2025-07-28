from transform.transform import SaSSTransform
from sir.operand import Operand
from sir.instruction import Instruction

class MovEliminate(SaSSTransform):
    def apply(self, module):
        print("=== Start of MovEliminate ===")
        count = 0

        for func in module.functions:
            for block in func.blocks:
                count += process(block)

        print(f"MovEliminate: removed {count} mov instructions")
        print("=== End of MovEliminate ===")

def process(block):
    count = 0

    new_instructions = []
    
    for inst in block.instructions:
        if inst.opcodes and inst.opcodes[0] == 'MOV' and inst.operands[1].IsReg:
            srcReg = inst.operands[1].Reg

            for _, UseOp in inst.Users:
                UseOp.SetReg(srcReg)
            
            count += 1
        else:
            new_instructions.append(inst)

    block.instructions = new_instructions
    
    return count