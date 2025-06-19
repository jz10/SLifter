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
    
    SR_TO_OFFSET = {
        'SR_NTID.X': 0x08,      # blockDim.x
        'SR_NTID.Y': 0x0C,      # blockDim.y  
        'SR_NTID.Z': 0x10,      # blockDim.z
        'SR_GRID_DIM.X': 0x14,  # gridDim.x
        'SR_GRID_DIM.Y': 0x18,  # gridDim.y
        'SR_GRID_DIM.Z': 0x1C,  # gridDim.z
        'SR_CTAID.X': 0x20,     # blockIdx.x (fake)
        'SR_CTAID.Y': 0x24,     # blockIdx.y (fake)
        'SR_CTAID.Z': 0x28,     # blockIdx.z (fake)
        'SR_TID.X': 0x2C,       # threadIdx.x (fake)
        'SR_TID.Y': 0x30,       # threadIdx.y (fake)
        'SR_TID.Z': 0x34        # threadIdx.z (fake)
    }
    
    # Replace special register operands with memory addresses
    for inst in instructions:
        if inst.opcodes and inst.opcodes[0] == 'S2R':
            offset = SR_TO_OFFSET[inst.operands[1].Name]

            inst.opcodes[0] = 'MOV'            
            new_operand = Operand(
                f"c[0x0][{hex(offset)}]", 
                None, 
                None, 
                offset, 
                False, 
                True, 
                False
            )
            new_operand.SetTypeDesc("Int32")
            inst._operands[1] = new_operand
            count += 1
    
    return count