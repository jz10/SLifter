from transform.transform import SaSSTransform
from collections import deque

class OperAggregate(SaSSTransform):
    # Apply operator aggregation on module 
    def apply(self, module):
        for func in module.functions:
            SeedInsts = self.GetPack64Instructions(func)
            
            for Inst in SeedInsts:
                self.MergeInstructionPairs(Inst)
            
    def GetPack64Instructions(self, func):
        for bb in func.blocks:
            Pack64Insts = []
            for inst in bb.instructions:
                if len(inst.opcodes) > 0 and inst.opcodes[0] == "PACK64":
                    Pack64Insts.append(inst)

        return Pack64Insts
    
    def MergeInstructionPairs(self, inst):
        
        InstPair = (inst.getUsesInsts()[0], inst.getUsesInsts()[1])

        RemoveInsts = set()
        Queue = deque([InstPair])

        while Queue:
            CurrPair = Queue.popleft()
            Inst1, Inst2 = CurrPair

            if Inst1.opcodes[0] == "SHL" and Inst2.opcodes[0] == "SHR":
                # First opcode SHL, second opcode SHR
                # First use operand same register, second use operand sums up to 32
                # (SHL R6, R0.reuse, 0x2 ; SHR R0, R0, 0x1e ;) => SHL R6, R0, 0x2
                RemoveInsts.add(Inst2)
                InstPair = (Inst1.getUsesInsts()[0], Inst2.getUsesInsts()[1])
                Queue.append(InstPair)
            if Inst1.opcodes[0] == "IADD" and Inst2.opcodes[0] == "IADD":
                # First opcode IADD, second opcode IADD.X
                # First use operand same register, second use operand offset difference is 4    
                # (IADD R4.CC, R6.reuse, c[0x0][0x140] ; IADD.X R5, R0.reuse, c[0x0][0x144] ;) => IADD.WIDE R4, R6, c[0x0][0x140]
                RemoveInsts.add(Inst2)
                Inst1.opcodes = ["IADD", "WIDE"]
                InstPair = (Inst1.getUsesInsts()[0], Inst2.getUsesInsts()[1])
                Queue.append(InstPair)

        # Remove instructions
        for inst in RemoveInsts:
            inst.block.instructions.remove(inst)

            