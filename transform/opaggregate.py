from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand
from collections import deque

class OperAggregate(SaSSTransform):
    # Apply operator aggregation on module 
    def apply(self, module):
        print("=== Start of Operator Aggregation Transformation ===")

        for func in module.functions:

            DualInsts = {}
            MergePoints = set()
            Pack64Insts = self.GetPack64Instructions(func)
            
            for Inst in Pack64Insts:
                self.MergeInstructionPairs(Inst, MergePoints, DualInsts)

            self.ApplyChanges(func, Pack64Insts, MergePoints, DualInsts)

            print(f"Function {func.name}:")
            print(f"Dual instructions merged: {len(DualInsts)}")
            print(f"CAST64 created: {len(MergePoints)}")
            print(f"PACK64 instructions removed: {len(Pack64Insts)}")
        print("=== End of Operator Aggregation Transformation ===")
            
    def GetPack64Instructions(self, func):
        Pack64Insts = []
        for bb in func.blocks:
            for inst in bb.instructions:
                if len(inst.opcodes) > 0 and inst.opcodes[0] == "PACK64":
                    Pack64Insts.append(inst)

        return Pack64Insts

    def MergeInstructionPairs(self, Pack64Inst, MergePoints, DualInsts):

        UseOps = Pack64Inst.GetUses()

        InstPair = (Pack64Inst.ReachingDefs[UseOps[0]], Pack64Inst.ReachingDefs[UseOps[1]])

        Queue = deque([InstPair])

        while Queue:
            CurrPair = Queue.popleft()
            Inst1, Inst2 = CurrPair

            if Inst1.opcodes[0] == "SHL" and Inst2.opcodes[0] == "SHR":
                # First opcode SHL, second opcode SHR
                # First use operand same register, second use operand sums up to 32
                # (SHL R6, R0.reuse, 0x2 ; SHR R0, R0, 0x1e ;) => SHL64 R6, R0, 0x2
                InstPair = (Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]])


            elif Inst1.opcodes[0] == "IADD" and Inst2.opcodes[0] == "IADD":
                # First opcode IADD, second opcode IADD.X
                # First use operand same register, second use operand offset difference is 4    
                # (IADD R4.CC, R6.reuse, c[0x0][0x140] ; IADD.X R5, R0.reuse, c[0x0][0x144] ;) => IADD64 R4, R6, c[0x0][0x140]
                InstPair = (Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]])

            elif Inst1.opcodes[0] == "IADD32I" and Inst2.opcodes[0] == "IADD":
                # First opcode IADD32I, second opcode IADD.X
                # First instruction has .CC, second instruction has .X
                # First instruction is an immediate, second must use zero
                # (IADD32I R20.CC R7 0x4 ; IADD.X R21 RZ R22;) => IADD64 R20, R7, 0x4

                if Inst2.GetUses()[0].Name != "RZ":
                    raise ValueError("Second instruction must use RZ as a use operand for IADD64 aggregation")

                InstPair = (Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[1]])
            
            elif Inst1.opcodes[0] == "MOV" and Inst2.opcodes[0] == "MOV":
                # First opcode MOV, second opcode MOV.X
                # First use operand same register, second use operand offset difference is 4
                # (MOV R4, R11 ; MOV R5, R12) => MOV64 R4, R11
                InstPair = (Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]])

            else:
                # Current pair not matching known patterns
                raise ValueError(f"Unhandled instruction pair: {Inst1} and {Inst2}")

            print(f"Processing pair: {Inst1} and {Inst2}")
            print(f"Next pair: {InstPair[0]} and {InstPair[1]}")
            DualInsts[Inst1] = Inst1, Inst2

            # Merge point reached, insert CAST64 before the starting instruction pair
            if InstPair[0] == InstPair[1]:
                MergePoints.add(Inst1)
                break

            Queue.append(InstPair)


    def ApplyChanges(self, func, Pack64Insts, MergePoints, DualInsts):

        RemoveInsts = set()

        # Merge dual insts by removing the second and updating the first
        for Inst1, Inst2 in DualInsts.values():
            # Remove .CC in def operand
            Inst1.GetDef()._Suffix = ''
            Inst1._opcodes = [Inst1.opcodes[0] + "64"] 
            RemoveInsts.add(Inst2)

        # Remove Pack64Insts and handle register dependencies
        for Inst in Pack64Insts:
            mergeOp = Inst.GetUses()[0]
            for _, UseOp in Inst.Users:
                UseOp.SetReg(mergeOp.Name)

            RemoveInsts.add(Inst)

        for bb in func.blocks:

            new_insts = []

            for inst in bb.instructions:
                if inst in RemoveInsts:
                    continue

                if inst in MergePoints:
                    # Create the CAST64 instruction
                    src_op_name = inst.GetUses()[0].Name
                    src_op = Operand(src_op_name, src_op_name, None, -1, True, False, False)

                    dest_op_name = src_op_name + "_int64"
                    dest_op = Operand(dest_op_name, dest_op_name, None, -1, True, False, False)

                    inst_content = f"CAST64 {dest_op.Name}, {src_op.Name}"
                    cast_inst = Instruction(
                        id=f"{inst.id}_cast64",
                        opcodes=["CAST64"],
                        operands=[dest_op, src_op],
                        inst_content=inst_content
                    )
                    new_insts.append(cast_inst)

                    # Update the instruction to use the new register
                    inst.GetUses()[0].SetReg(dest_op_name)

                new_insts.append(inst)

            bb.instructions = new_insts
