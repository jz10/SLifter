from transform.transform import SaSSTransform
from sir.operand import Operand
from sir.instruction import Instruction

class SRSubstitute(SaSSTransform):
    def apply(self, module):
        print("=== Start of SRSubstitute ===")
        count = 0

        for func in module.functions:
            for block in func.blocks:
                count += process(block.instructions)

        print(f"SRSubstitute: processed {count} operands")
        print("=== End of SRSubstitute ===")

def process(instructions):
    count = 0
    
    OFFSET_TO_SR = {
        0x08: 'SR_NTID.X',  # blockDim.x
        0x10: 'SR_NTID.Y',  # blockDim.y  
        0x18: 'SR_NTID.Z',  # blockDim.z
        0x04: 'SR_GRID_DIM.X',  # gridDim.x
        0x0c: 'SR_GRID_DIM.Y',  # gridDim.y
        0x14: 'SR_GRID_DIM.Z'   # gridDim.z
        # gridDim SR do not exist, but we use here for easier parsing
    }
    
    # Track which offsets have been processed and their corresponding temp register names
    offset_to_temp_reg = {}
    new_instructions = []
    
    # Identify all constant memory accesses and create S2R instructions at the earliest position
    for inst in instructions:
        for i, operand in enumerate(inst.operands):
            if operand.IsArg and operand.ArgOffset in OFFSET_TO_SR:
                offset = operand.ArgOffset
                
                if offset not in offset_to_temp_reg:
                    sr_name = OFFSET_TO_SR[offset]
                    temp_reg_name = f"sr_temp_{hex(offset)}"
                    offset_to_temp_reg[offset] = temp_reg_name
                    
                    # Create the S2R instruction
                    sr_operand = Operand(sr_name, sr_name, None, -1, False, False, False)
                    temp_operand = Operand(temp_reg_name, temp_reg_name, None, -1, True, False, False)
                    
                    s2r_inst = Instruction(
                        id=f"s2r_{hex(offset)}",
                        opcodes=["S2R"],
                        operands=[temp_operand, sr_operand],
                        parentBB=inst.parent
                    )
                    
                    new_instructions.append(s2r_inst)
                    count += 1
        
        # Replace operands and add the original instruction
        for i, operand in enumerate(inst.operands):
            if operand.IsArg and operand.ArgOffset in OFFSET_TO_SR:
                offset = operand.ArgOffset
                temp_reg_name = offset_to_temp_reg[offset]
                
                new_operand = Operand(temp_reg_name, temp_reg_name, operand.Suffix, -1, True, False, operand.IsMemAddr)
                new_operand.SetTypeDesc(operand.GetTypeDesc())
                inst._operands[i] = new_operand
        
        new_instructions.append(inst)
    
    instructions.clear()
    instructions.extend(new_instructions)
    
    return count
