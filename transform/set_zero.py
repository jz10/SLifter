from transform.transform import SaSSTransform
from sir.operand import Operand
from sir.instruction import Instruction

class SetZero(SaSSTransform):
    def __init__(self, name):
        super().__init__(name)

    def apply(self, module):
        print("=== Start of SetZero ===")
        count = 0

        for func in module.functions:
            count += self.process(func)

        print(f"SetZero: replaced {count} zero-setting instructions with SETZERO")
        print("=== End of SetZero ===")

    def handleZeroPattern(self, inst, replaceInsts):
        # Pattern 1: MOV Rx, RZ
        if (inst.opcodes[0] == "MOV" and 
            len(inst.operands) >= 2 and 
            inst.operands[1].IsRZ):
            dest_reg = inst.GetDef()
            setzero_inst = self.create_setzero_instruction(inst, dest_reg)
            replaceInsts[inst] = setzero_inst
            return 1
        
        # Pattern 2: IMAD.MOV.U32 Rx, RZ, RZ, RZ
        if (inst.opcodes[0] == "IMAD" and 
            "MOV" in inst.opcodes and
            len(inst.operands) >= 4 and
            inst.operands[1].IsRZ and 
            inst.operands[2].IsRZ and
            inst.operands[3].IsRZ):
            dest_reg = inst.GetDef()
            setzero_inst = self.create_setzero_instruction(inst, dest_reg)
            replaceInsts[inst] = setzero_inst
            return 1
        
        # Pattern 3: XOR.32 Rx, Ry, Ry (XOR a register with itself)
        if (inst.opcodes[0] == "XOR" and 
            len(inst.operands) >= 3 and
            inst.operands[1].Name == inst.operands[2].Name):
            dest_reg = inst.GetDef()
            setzero_inst = self.create_setzero_instruction(inst, dest_reg)
            replaceInsts[inst] = setzero_inst
            return 1
        
        # Pattern 4: IADD3.RS Rx, RZ, RZ, RZ
        if (inst.opcodes[0] == "IADD3" and 
            len(inst.operands) >= 4 and
            inst.operands[1].IsRZ and 
            inst.operands[2].IsRZ and
            inst.operands[3].IsRZ):
            dest_reg = inst.GetDef()
            setzero_inst = self.create_setzero_instruction(inst, dest_reg)
            replaceInsts[inst] = setzero_inst
            return 1

        # Pattern 5: CS2R R26 = SRZ
        if (inst.opcodes[0] == "CS2R" and 
            len(inst.operands) >= 2 and
            inst.operands[1].IsRZ):
            dest_reg = inst.GetDef()
            setzero_inst = self.create_setzero_instruction(inst, dest_reg)
            replaceInsts[inst] = setzero_inst
            return 1
        
        return 0

    def create_setzero_instruction(self, original_inst, dest_reg):
        new_dest = dest_reg.Clone()
        
        setzero_inst = Instruction(
            id=f"{original_inst.id}_setzero",
            opcodes=["SETZERO"],
            operands=[new_dest],
            inst_content=f"SETZERO {new_dest.Name}",
            parentBB=original_inst.parent
        )
        
        return setzero_inst

    def process(self, func):
        replaceInsts = {}
        count = 0

        for block in func.blocks:
            for inst in block.instructions:
                count += self.handleZeroPattern(inst, replaceInsts)

        for block in func.blocks:
            new_instructions = []
            for inst in block.instructions:
                if inst in replaceInsts:
                    new_instructions.append(replaceInsts[inst])
                else:
                    new_instructions.append(inst)
            block.instructions = new_instructions
        
        return count