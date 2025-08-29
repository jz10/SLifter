from transform.transform import SaSSTransform
from sir.operand import Operand
from sir.instruction import Instruction


def _sr_map_for_isa(isa: str | None):
    # Default mapping (sm_35/sm_52): NTID.X at 0x08
    base = {
        'SR_NTID.X': 0x08,
        'SR_NTID.Y': 0x0C,
        'SR_NTID.Z': 0x10,
        'SR_GRID_DIM.X': 0x14,
        'SR_GRID_DIM.Y': 0x18,
        'SR_GRID_DIM.Z': 0x1C,
        'SR_CTAID.X': 0x20,
        'SR_CTAID.Y': 0x24,
        'SR_CTAID.Z': 0x28,
        'SR_TID.X': 0x2C,
        'SR_TID.Y': 0x30,
        'SR_TID.Z': 0x34,
        'SR_LANEID': 0x38,
    }
    if isa == 'sm_75':
        base = dict(base)
        base['SR_NTID.X'] = 0x0
        base['SR_NTID.Y'] = 0x4
        base['SR_NTID.Z'] = 0x8
        base['SR_GRID_DIM.X'] = 0xC
        base['SR_GRID_DIM.Y'] = 0x10
        base['SR_GRID_DIM.Z'] = 0x14
    return base


class SRSubstitute(SaSSTransform):
    def apply(self, module):
        print("=== Start of SRSubstitute ===")
        count = 0
        sr_to_offset = _sr_map_for_isa(getattr(module, 'isa', None))

        for func in module.functions:
            for block in func.blocks:
                count += process(block.instructions, sr_to_offset)

        print(f"SRSubstitute: processed {count} operands")
        print("=== End of SRSubstitute ===")


def process(instructions, sr_to_offset):
    count = 0

    # Replace special register operands with memory addresses
    for inst in instructions:
        if inst.opcodes and inst.opcodes[0] == 'S2R':
            offset = sr_to_offset[inst.operands[1].Name]

            inst.opcodes[0] = 'MOV'
            new_operand = Operand(
                f"c[0x0][{hex(offset)}]",
                None,
                None,
                offset,
                False,
                True,
                False,
            )
            inst._operands[1] = new_operand
            count += 1

    return count

# Keep a default export for other modules that import it
SR_TO_OFFSET = _sr_map_for_isa(None)
