from transform.transform import SaSSTransform
from sir.operand import Operand
from sir.instruction import Instruction

class FPHack(SaSSTransform):
    def apply(self, module):
        print("=== Start of FPHack ===")
        count = 0

        for func in module.functions:
            count += self.process(func)

        print(f"FPHack: replaced {count} pattern1")
        print("=== End of FPHack ===")

    # Fast integer division
    # I2F.F32.U32.RP R4, R0 ;
    # MUFU.RCP R4, R4 ;
    # IADD32I R5, R4, 0xffffffe ;  or IADD3 R5, R4, 0xffffffe, RZ
    # F2I.FTZ.U32.F32.TRUNC R5, R5 ; 
    # This hack produces a magic number for fast integer division
    
    # to make type system working, insert BITCAST before and after IADD32I
    def handlePattern1(self, inst, insertInsts, removeInsts):
        if inst.opcodes[0] != "I2F":
            return 0
        
        inst1 = inst
        inst2 = None
        inst3 = None
        inst4 = None

        for users in inst.Users.values():
            for useInst, useOp in users:
                if useInst.opcodes[0] == "MUFU":
                    inst2 = useInst

        if not inst2:
            return 0
        
        for users in inst2.Users.values():
            for useInst, useOp in users:
                if useInst.opcodes[0] == "IADD32I" or useInst.opcodes[0] == "IADD3":
                    inst3 = useInst
                    iaddUseOp = useOp

        if not inst3:
            return 0
        
        for users in inst3.Users.values():
            for useInst, useOp in users:
                if useInst.opcodes[0] == "F2I":
                    inst4 = useInst
                    f2iUseOp = useOp

        if not inst4:
            return 0

        src_op = inst2.GetDef().Clone()
        dest_op = src_op.Clone()
        dest_op.SetReg(src_op.Name + "_cast")
        iaddUseOp.SetReg(src_op.Name + "_cast")

        for users in inst2.Users.values():
            for useInst, useOp in users:
                useOp.SetReg(src_op.Name + "_cast")

        bitcastInstBefore = Instruction(
            id=f"{inst2.id}_bitcast_before",
            opcodes=["BITCAST"],
            operands=[dest_op, src_op],
            parentBB=inst2.parent
        )
        insertInsts[inst2] = bitcastInstBefore
        
        src_op = inst3.GetDef().Clone()
        dest_op = src_op.Clone()
        dest_op.SetReg(src_op.Name + "_cast")
        f2iUseOp.SetReg(src_op.Name + "_cast")

        bitcastInstAfter = Instruction(
            id=f"{inst3.id}_bitcast_after",
            opcodes=["BITCAST"],
            operands=[dest_op, src_op],
            parentBB=inst3.parent
        )
        insertInsts[inst3] = bitcastInstAfter

        return 1


    def process(self, func):

        insertInsts = {}
        removeInsts = set()

        count = 0

        for block in func.blocks:
            for inst in block.instructions:
                count += self.handlePattern1(inst, insertInsts, removeInsts)

        for block in func.blocks:
            new_instructions = []
            for inst in block.instructions:
                if inst not in removeInsts:
                    new_instructions.append(inst)
                if inst in insertInsts:
                    new_instructions.append(insertInsts[inst])

            block.instructions = new_instructions
        
        return count
