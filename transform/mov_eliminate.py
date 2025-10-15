from transform.transform import SaSSTransform

class MovEliminate(SaSSTransform):
    def apply(self, module):
        print("=== Start of MovEliminate ===")
        count = 0

        for func in module.functions:
            print(f"Processing function: {func.name}")
            for block in func.blocks:
                count += self.processBlock(block)
                

        print(f"MovEliminate: removed {count} mov instructions")
        print("=== End of MovEliminate ===")

    def processBlock(self, block):
        count = 0
        newInstructions = []
        
        
        for inst in block.instructions:
            if self.checkInstruction(inst):
                newInstructions.append(inst)
            else:
                print(f"Removed: {inst}")
                count += 1
        
        block.instructions = newInstructions
        return count
    
    
    def convertMov(self, inst):
        
        # UMOV UR4, UR7 => MOV UR4, UR7
        if inst.opcodes and inst.opcodes[0] == 'UMOV' and len(inst.operands) >= 2:
            dstOp = inst.operands[0].Clone()
            srcOp = inst.operands[1].Clone()
            newInst = inst.Clone()
            newInst._opcodes = ['MOV']
            newInst._operands = [dstOp, srcOp]
            return newInst
        
        # IMAD.MOV R13, RZ, RZ, -R13 => MOV R13, -R13
        if (inst.opcodes and len(inst.opcodes) >= 2 and inst.opcodes[0] == 'IMAD' and inst.opcodes[1] == "MOV"):
            if len(inst.operands) >= 4:
                op1 = inst.operands[1]
                op2 = inst.operands[2]
                if op1.IsRZ and op2.IsRZ:
                    dstOp = inst.operands[0].Clone()
                    srcOp = inst.operands[3].Clone()
                    newInst = inst.Clone()
                    newInst._opcodes = ['MOV']
                    newInst._operands = [dstOp, srcOp]
                    return newInst
                
    def checkInstruction(self, inst):
        
        status = True
        
        # MOV R17, R58
        if inst.opcodes and inst.opcodes[0] == 'MOV' and len(inst.operands) >= 2 and inst.operands[1].IsReg:
            status = False
            reg = inst.operands[1]
        
        # UMOV UR4, UR7
        if inst.opcodes and inst.opcodes[0] == 'UMOV' and len(inst.operands) >= 2 and inst.operands[1].IsReg:
            status = False
            reg = inst.operands[1]
        
        # IMAD.MOV R13, RZ, RZ, -R13
        if (inst.opcodes and len(inst.opcodes) >= 2 and inst.opcodes[0] == 'IMAD' and inst.opcodes[1] == "MOV"):
            if len(inst.operands) >= 4:
                op1 = inst.operands[1]
                op2 = inst.operands[2]
                if op1.IsRZ and op2.IsRZ and inst.operands[3].IsReg:
                    status = False
                    reg = inst.operands[3]
            
        # Rename        
        if not status:
            for _, userOp in inst.Users.get(inst.operands[0], set()):
                if userOp.Reg != inst.operands[0].Reg:
                    raise Exception("Should not happen")
                userOp.Replace(reg)

            return False

        return True