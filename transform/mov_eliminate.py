from transform.transform import SaSSTransform

class MovEliminate(SaSSTransform):
    def apply(self, module):
        print("=== Start of MovEliminate ===")
        count = 0

        for func in module.functions:
            for block in func.blocks:
                count += self.processBlock(block)

        print(f"MovEliminate: removed {count} mov instructions")
        print("=== End of MovEliminate ===")
    
    def processBlock(self, block):
        count = 0
        newInstructions = []
        
        for inst in block.instructions:
            if self.shouldKeepInstruction(inst, block):
                newInstructions.append(inst)
            else:
                print(f"Removed: {inst}")
                count += 1
        
        block.instructions = newInstructions
        return count
    
    def shouldKeepInstruction(self, inst, block):
        if inst.pflag is not None:
            return True
        
        copyInfo = self.getCopyInfo(inst)
        if not copyInfo:
            return True
        
        srcOp, _ = copyInfo
        if not (srcOp.IsReg or srcOp.IsArg or srcOp.IsImmediate):
            return True
        
        return not self.canEliminateCopy(inst, srcOp, block)
    
    def getCopyInfo(self, inst):
        if not inst.opcodes:
            return None
        
        op0 = inst.opcodes[0]
        
        if (op0 == 'MOV' or op0 == 'UMOV') and len(inst.operands) >= 2:
            return inst.operands[1], inst.operands[0]
        
        elif (op0 == 'IMAD' and 'MOV' in inst.opcodes and len(inst.operands) >= 4 and
              inst.operands[1].IsRZ and inst.operands[2].IsRZ):
            return inst.operands[3], inst.operands[0]
        
        elif op0 == 'MOV32I' and len(inst.operands) >= 2:
            return inst.operands[1], inst.operands[0]
        
        return None
    
    def canEliminateCopy(self, inst, srcOp, block):
        usesList = []
        for users in inst.Users.values():
            for useInst, useOp in users:
                usesList.append((useInst, useOp))
        
        sameBlockUses = [(ui, uo) for ui, uo in usesList if ui.parent is block]
        crossBlockUses = [(ui, uo) for ui, uo in usesList if ui.parent is not block]
        
        canReplaceSameBlock = self.canReplaceSameBlockUses(
            inst, srcOp, sameBlockUses, block)
        
        if canReplaceSameBlock:
            for _, useOp in sameBlockUses:
                useOp.Replace(srcOp)
            
            return len(crossBlockUses) == 0
        
        return False
    
    def canReplaceSameBlockUses(self, copyInst, srcOp, sameBlockUses, block):
        if not srcOp.IsReg:
            return True

        pos = {inst: i for i, inst in enumerate(block.instructions)}
        if copyInst not in pos:
            return False
        instIdx = pos[copyInst]

        srcReg = srcOp.Reg
        
        for useInst, _ in sameBlockUses:
            if useInst not in pos:
                return False
            useIdx = pos[useInst]

            for k in range(instIdx + 1, useIdx + 1):
                for defOp in block.instructions[k].GetDefs():
                    if defOp and defOp.Reg == srcReg:
                        return False
        
        return True

def process(block):
    eliminator = MovEliminate("legacy")
    return eliminator.processBlock(block)
