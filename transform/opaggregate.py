from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand
from collections import deque

class OperAggregate(SaSSTransform):
    # Apply operator aggregation on module 
    def apply(self, module):
        print("=== Start of Operator Aggregation Transformation ===")

        for func in module.functions:

            InsertInsts = {}
            RemoveInsts = set()
            Cast64Inserts = set()
            Pack64Insts = self.GetPack64Instructions(func)
            
            for Inst in Pack64Insts:
                self.MergeInstructionPairs(Inst, Cast64Inserts, InsertInsts, RemoveInsts)

            self.ApplyChanges(func, Pack64Insts, Cast64Inserts, InsertInsts, RemoveInsts)

            print(f"Function {func.name}:")
            print(f"Cast64 Inserts: {len(Cast64Inserts)}")
            print(f"Insert Instructions: {len(InsertInsts)}")
            print(f"Remove Instructions: {len(RemoveInsts)}")
        print("=== End of Operator Aggregation Transformation ===")
            
    def GetPack64Instructions(self, func):
        Pack64Insts = []
        for bb in func.blocks:
            for inst in bb.instructions:
                if len(inst.opcodes) > 0 and inst.opcodes[0] == "PACK64":
                    Pack64Insts.append(inst)

        return Pack64Insts

    def MergeInstructionPairs(self, Pack64Inst, Cast64Inserts, InsertInsts, RemoveInsts):

        UseOps = Pack64Inst.GetUses()

        InstPair = (Pack64Inst.ReachingDefs[UseOps[0]], Pack64Inst.ReachingDefs[UseOps[1]])

        Queue = deque([InstPair])

        while Queue:
            CurrPair = Queue.popleft()
            Inst1, Inst2 = CurrPair

            if Inst1 in RemoveInsts:
                # Already processed this instruction pair
                continue

            NextInstPairs = []

            if Inst1.opcodes[0] == "SHL" and Inst2.opcodes[0] == "SHR":
                # First opcode SHL, second opcode SHR
                # First use operand same register, second use operand sums up to 32
                # (SHL R6, R0.reuse, 0x2 ; SHR R0, R0, 0x1e ;) => SHL64 R6, R0, 0x2

                dest_op = Inst1.GetDef().Clone()
                src_op = Inst2.GetUses()[0].Clone()
                imm_op = Inst1.GetUses()[1].Clone()
                inst = Instruction(
                    id=f"{Inst1.id}_shl64",
                    opcodes=["SHL64"],
                    operands=[dest_op, src_op, imm_op],
                    inst_content=f"SHL64 {dest_op.Name}, {src_op.Name}, {imm_op.Name}",
                    parentBB=Inst1.parent
                )
                InsertInsts[Inst1] = inst

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]]))


            elif Inst1.opcodes[0] == "IADD" and Inst2.opcodes[0] == "IADD":
                # First opcode IADD, second opcode IADD.X
                # (IADD R4.CC, R6.reuse, c[0x0][0x140] ; IADD.X R5, R0.reuse, c[0x0][0x144] ;) => IADD64 R4, R6, c[0x0][0x140]
                # (IADD R29.CC R28 R20, IADD.X R31 R30 R21) => IADD64 R29, R28, R20

                dest_op = Inst1.GetDef().Clone()
                src_op = Inst1.GetUses()[0].Clone()
                src_op2 = Inst1.GetUses()[1].Clone()
                inst = Instruction(
                    id=f"{Inst1.id}_iadd64",
                    opcodes=["IADD64"],
                    operands=[dest_op, src_op, src_op2],
                    inst_content=f"IADD64 {dest_op.Name}, {src_op.Name}, {src_op2.Name}",
                    parentBB=Inst1.parent
                )

                InsertInsts[Inst1] = inst

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                for i in range(len(Inst1.GetUses())):
                    if Inst1.GetUses()[i].IsReg:
                        NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[i]], Inst2.ReachingDefs[Inst2.GetUses()[i]]))

            elif Inst1.opcodes[0] == "IADD32I" and Inst2.opcodes[0] == "IADD":
                # First opcode IADD32I, second opcode IADD.X
                # First instruction has .CC, second instruction has .X
                # First instruction is an immediate, second must use zero
                # (IADD32I R20.CC R7 0x4 ; IADD.X R21 RZ R22;) => IADD64 R20, R7, 0x4

                if Inst2.GetUses()[0].Name != "RZ":
                    raise ValueError("Second instruction must use RZ as a use operand for IADD64 aggregation")
                
                dest_op = Inst1.GetDef().Clone()
                src_op = Inst1.GetUses()[0].Clone()
                imm_op = Inst1.GetUses()[1].Clone()
                inst = Instruction(
                    id=f"{Inst1.id}_iadd64",
                    opcodes=["IADD64"],
                    operands=[dest_op, src_op, imm_op],
                    inst_content=f"IADD64 {dest_op.Name}, {src_op.Name}, {imm_op.Name}",
                    parentBB=Inst1.parent
                )

                InsertInsts[Inst1] = inst

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[1]]))

            elif Inst1.opcodes[0] == "MOV" and Inst2.opcodes[0] == "MOV":
                # First opcode MOV, second opcode MOV.X
                # First use operand same register, second use operand offset difference is 4
                # (MOV R4, R11 ; MOV R5, R12) => MOV64 R4, R11

                dest_op = Inst1.GetDef().Clone()
                src_op = Inst1.GetUses()[0].Clone()
                src_op2 = Inst1.GetUses()[1].Clone()
                inst = Instruction(
                    id=f"{Inst1.id}_mov64",
                    opcodes=["MOV64"],
                    operands=[dest_op, src_op, src_op2],
                    inst_content=f"MOV64 {dest_op.Name}, {src_op.Name}, {src_op2.Name}",
                    parentBB=Inst1.parent
                )

                InsertInsts[Inst1] = inst

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]]))

            elif Inst1.opcodes[0] == "PHI" and Inst2.opcodes[0] == "PHI":
                # Both instructions are PHI
                # (PHI R4, R5 ; PHI R6, R7) => PHI64 R4, R5, R7

                dest_op = Inst1.GetDef().Clone()
                src_ops = [op.Clone() for op in Inst1.GetUses()]
                inst = Instruction(
                    id=f"{Inst1.id}_phi64",
                    opcodes=["PHI64"],
                    operands=[dest_op] + src_ops,
                    inst_content=f"PHI64 {dest_op.Name}, {', '.join(op.Name for op in src_ops)}",
                    parentBB=Inst1.parent
                )

                InsertInsts[Inst1] = inst

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                for i in range(len(Inst1.GetUses())):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[i]], Inst2.ReachingDefs[Inst2.GetUses()[i]]))

            elif Inst1.opcodes[0] == "SHL" and Inst2.opcodes[0] == "SHF" and Inst2.opcodes[1] == "L":
                # (SHR R13, R2.reuse, 0x1f, SHL R12, R2.reuse, 0x2, SHF.L.U64 R13, R2, 0x2, R13) => (SHL64 R12, R2, 0x2)
                # 0x1f indicates sign bit extraction
                # This is a sign extended 64bit shift left that takes a 32bit input 
                Inst3 = Inst2.ReachingDefs[Inst2.GetUses()[2]]  # SHR instruction

                dest_op = Inst1.GetDef().Clone()
                src_op = Inst1.GetUses()[0].Clone()
                src_op2 = Inst1.GetUses()[1].Clone()
                inst = Instruction(
                    id=f"{Inst1.id}_shl64",
                    opcodes=["SHL64"],
                    operands=[dest_op, src_op, src_op2],
                    inst_content=f"SHL64 {dest_op.Name}, {src_op.Name}, {src_op2.Name}",
                    parentBB=Inst1.parent
                )

                InsertInsts[Inst3] = inst

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)
                RemoveInsts.add(Inst3)

                # Special case: Instruction pairs should end at this point
                # This pattern always start from a 32-bit register
                Cast64Inserts.add(Inst3)
                continue

            elif Inst1.opcodes[0] == "ISCADD" and Inst2.opcodes[0] == "IADD":
                # (SHR R3 R2.reuse 0x1e, ISCADD R2.CC R2 c[0x0][0x150] 0x2, IADD.X R3 R3 c[0x0][0x154]) => (IMAD64 R2, R2, c[0x0][0x150], 0x2)
                # Three instructions involved
                Inst3 = Inst2.ReachingDefs[Inst2.GetUses()[0]] # SHR instruction

                if Inst3.opcodes[0] != "SHR":
                    raise ValueError("Expected SHR instruction as the third instruction in the pair")
                
                dest_op = Inst1.GetDef().Clone()
                src_op = Inst1.GetUses()[0].Clone() 
                src_op2 = Inst1.GetUses()[1].Clone()
                offset = Inst1.GetUses()[2].ImmediateValue
                val = 1 << offset
                imm_op = Operand(str(val), None, None, -1, False, False, False, True, val)
                inst = Instruction(
                    id=f"{Inst1.id}_imad64",
                    opcodes=["IMAD64"],
                    operands=[dest_op, src_op, imm_op, src_op2],
                    inst_content=f"IMAD64 {dest_op.Name}, {src_op.Name}, {imm_op.Name}, {src_op2.Name}",
                    parentBB=Inst1.parent
                )

                InsertInsts[Inst3] = inst

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)
                RemoveInsts.add(Inst3)

                # Special case: Instruction pairs should end at this point
                # This pattern always start from a 32-bit register
                Cast64Inserts.add(Inst3)
                continue

            else:
                # Current pair not matching known patterns
                raise ValueError(f"Unhandled instruction pair: {Inst1} and {Inst2}")

            print(f"Processing pair: {Inst1} and {Inst2}")
            for InstPair in NextInstPairs:
                print(f"\tNext pair: {InstPair[0]} and {InstPair[1]}")

            for InstPair in NextInstPairs:
                # If merge point reached, insert CAST64 before the starting instruction pair
                if InstPair[0] == InstPair[1]:
                    Cast64Inserts.add(Inst1)
                else:
                    Queue.append(InstPair)


    def ApplyChanges(self, func, Pack64Insts, Cast64Inserts, InsertInsts, RemoveInsts):

        # Remove Pack64Insts and handle register dependencies
        for Inst in Pack64Insts:
            mergeOp = Inst.GetUses()[0]
            for _, UseOp in Inst.Users:
                UseOp.SetReg(mergeOp.Name)

            RemoveInsts.add(Inst)

        for bb in func.blocks:

            new_insts = []

            for inst in bb.instructions:

                if inst in Cast64Inserts:
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
                        inst_content=inst_content,
                        parentBB=inst.parent
                    )
                    new_insts.append(cast_inst)

                    # Update the instruction to use the new register
                    InsertInsts[inst].GetUses()[0].SetReg(dest_op_name)

                if inst in InsertInsts:
                    insertInst = InsertInsts[inst]
                    new_insts.append(insertInst)

                
                if inst not in RemoveInsts:
                    new_insts.append(inst)

            bb.instructions = new_insts
