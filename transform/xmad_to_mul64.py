from transform.transform import SaSSTransform
from sir.operand import Operand
from sir.instruction import Instruction

class XmadToMul64(SaSSTransform):
    def apply(self, module):
        print("=== Start of XmadToMul64 ===")
        count = 0

        for func in module.functions:
            for block in func.blocks:
                # Find XMAD patterns in the basic block
                patterns = self.find_patterns(block.instructions)

                # Replace XMAD with MUL64
                self.replace_xmad_with_mul64(block.instructions, patterns)

                count += len(patterns)

        print(f"Transformed {count} set of xmad instructions to mul64 instructions.")
        print("=== End of XmadToMul64 ===")
            
    def find_patterns(self, instructions):
        candidates1 = {}
        candidates2 = {}
        patterns = {}
        
        for idx in range(len(instructions)-1, -1, -1):
            instr = instructions[idx]

            if len(instr.opcodes) > 0 and instr.opcodes[0] != "XMAD":
                continue
            
            # Match XMAD.PSL.CBCC (third instruction)
            if len(instr.opcodes) == 3 and instr.opcodes[1] == "PSL" and instr.opcodes[2] == "CBCC" and instr.operands[1]._Suffix == "H1":
                def_reg = instr.operands[0].Reg
                A_base_third = instr.operands[1].Reg
                T_reg = instr.operands[2].Reg  # From first XMAD.MRG
                L_reg = instr.operands[3].Reg  # From second XMAD
                
                candidates1[T_reg] = (A_base_third, T_reg, def_reg)
                candidates2[L_reg] = (A_base_third, T_reg, def_reg)

                patterns[def_reg] = (instr.id, instr.id, instr.id)  # Placeholder for now
            
            # Match XMAD (second instruction)
            elif len(instr.opcodes) == 1:
                if instr.operands[0].Reg in candidates2:
                    A_base, T_reg, def_reg = candidates2[instr.operands[0].Reg]
                    if (instr.operands[1].Reg == A_base):
                        i1, i2, i3 = patterns[def_reg]
                        patterns[def_reg] = (i1, instr.id, i3)
            
            # Match XMAD.MRG (first instruction)
            elif len(instr.opcodes) == 2 and instr.opcodes[1] == "MRG" and  instr.operands[2]._Suffix == "H1":
                if instr.operands[0].Reg in candidates1:
                    A_base, T_reg, def_reg = candidates1[instr.operands[0].Reg]
                    if (instr.operands[1].Reg == A_base):
                        i1, i2, i3 = patterns[def_reg]
                        patterns[def_reg] = (instr.id, i2, i3)
        
        return patterns

    def replace_xmad_with_mul64(self, instructions, patterns):
        # Sort patterns by earliest instruction index to process in order
        sorted_patterns = sorted(patterns.items(), key=lambda x: min(x[1]))
        
        # Process patterns from end to beginning to maintain instruction indices
        for def_reg, (i1, i2, i3) in reversed(sorted_patterns):
            if i1 is not None and i2 is not None and i3 is not None:
                # Find instruction objects by ID
                instr1 = next(instr for instr in instructions if instr.id == i1)  # XMAD.MRG
                instr2 = next(instr for instr in instructions if instr.id == i2)  # XMAD
                instr3 = next(instr for instr in instructions if instr.id == i3)  # XMAD.PSL.CBCC
                
                # Extract multiplicands from first instruction (always operands 1 and 2)
                if len(instr1.operands) < 3:
                    raise ValueError(f"XMAD.MRG instruction {instr1.id} missing required operands")
                multiplicand1 = instr1.operands[1]
                multiplicand2 = instr1.operands[2]
                
                # Get addend from second instruction for IMAD.WIDE
                # First instruction should always have addend as RZ
                imad_addend = Operand("RZ", "RZ", None, None, True, False, False)
                if len(instr2.operands) > 3 and instr2.operands[3].Reg != "RZ":
                    imad_addend = instr2.operands[3]
                
                # Create IMAD.WIDE instruction with second instruction's addend
                imad_operands = [
                    Operand(def_reg, def_reg, None, None, True, False, False),
                    multiplicand1,
                    multiplicand2,
                    imad_addend
                ]
                
                imad_inst_content = f"IMAD.WIDE {def_reg}, {multiplicand1.Name}, {multiplicand2.Name}, {imad_addend.Name}"
                
                imad_instr = Instruction(
                    id=f"imad_wide_{def_reg}",
                    opcodes=["IMAD", "WIDE"],
                    operands=imad_operands,
                    inst_content=imad_inst_content
                )
                
                # Find the actual indices of the instructions in the list
                idx1 = next(i for i, instr in enumerate(instructions) if instr.id == i1)
                idx2 = next(i for i, instr in enumerate(instructions) if instr.id == i2)
                idx3 = next(i for i, instr in enumerate(instructions) if instr.id == i3)
                
                # Insert IMAD.WIDE at earliest instruction location
                earliest_idx = min(idx1, idx2, idx3)
                instructions.insert(earliest_idx, imad_instr)
                
                # Delete original XMAD instructions
                instruction_ids_to_delete = {i1, i2, i3}
                instructions[:] = [instr for instr in instructions if instr.id not in instruction_ids_to_delete]