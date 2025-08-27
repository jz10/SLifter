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
            Cast64Inserts = {}
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
                # Case 1
                # (SHL R6, R0.reuse, 0x2 ; SHR R0, R0, 0x1e ;) => SHL64 R6, R0, 0x2
                # Case 2
                # (SHL R6, R0.reuse, 0x2 ; SHR.U32 R0, R0, 0x1e ;) => SHL64.U32 R6, R0, 0x2

                unsigned = "U32" in Inst2.opcodes
                opcodes = ["SHL64"]
                if unsigned:
                    opcodes.append("U32")

                dest_op = Inst1.GetDef().Clone()
                src_op = Inst2.GetUses()[0].Clone()
                imm_op = Inst1.GetUses()[1].Clone()
                inst = Instruction(
                    id=f"{Inst1.id}_shl64",
                    opcodes=opcodes,
                    operands=[dest_op, src_op, imm_op],
                    inst_content=f"SHL64 {dest_op.Name}, {src_op.Name}, {imm_op.Name}",
                    parentBB=Inst1.parent
                )
                InsertInsts[Inst1] = inst

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                Cast64Inserts.setdefault(Inst1, []).append(0)


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
                Cast64Inserts.setdefault(Inst3, []).append(0)

            elif Inst1.opcodes[0] == "ISCADD" and Inst2.opcodes[0] == "IADD":
                # (SHR R3 R2.reuse 0x1e, ISCADD R2.CC R2 c[0x0][0x150] 0x2, IADD.X R3 R3 c[0x0][0x154]) => (IMAD64 R2, 0x4, c[0x0][0x150])
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
                Cast64Inserts.setdefault(Inst3, []).append(0)

            elif Inst1 == Inst2 and Inst1.opcodes[0] == "IMAD" and Inst1.opcodes[1] == "WIDE":
                # IMAD.WIDE R6, R7 = R4, R5, c[0x0][0x168] => IMAD64 R6 = R4, R5, c[0x0][0x168]

                dest_op = Inst1.GetDefs()[0].Clone()
                src_ops = [op.Clone() for op in Inst1.GetUses()]

                inst = Instruction(
                    id=f"{Inst1.id}_imad64",
                    opcodes=["IMAD64"],
                    operands=[dest_op] + src_ops,
                    inst_content=f"IMAD64 {dest_op.Name}, {', '.join(op.Name for op in src_ops)}",
                    parentBB=Inst1.parent
                )

                InsertInsts[Inst1] = inst
                RemoveInsts.add(Inst1)

                # Special case: Instruction pairs should end at this point
                # This pattern always start from two 32-bit register
                for i in range(len(Inst1.GetUses())):
                    if Inst1.GetUses()[i].IsReg:
                        Cast64Inserts.setdefault(Inst1, []).append(i)

            elif Inst1.opcodes[0] == "IADD3" and Inst2.opcodes[0] == "IMAD": 
                if Inst1.GetDefs()[1].IsPredicateReg and Inst2.opcodes[1] == "X":
                    dest_op = Inst1.GetDefs()[0].Clone()
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

                if (Inst1.GetUses()[0].IsReg and Inst2.GetUses()[0].IsReg):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]]))
                if (Inst1.GetUses()[1].IsReg and Inst2.GetUses()[2].IsReg):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[1]], Inst2.ReachingDefs[Inst2.GetUses()[2]]))

            elif Inst1.opcodes[0] == "LEA" and Inst2.opcodes[0] == "LEA":
                # (LEA R62, P0 = R58, R52, 0x2, LEA.HI.X R64 = R58, R53, R59, 0x2, P0) => (IMAD64 R62 = R52 0x4 R58)
                dest_op = Inst1.GetDefs()[0].Clone()
                src_op1 = Inst1.GetUses()[0].Clone()
                src_op2 = Inst1.GetUses()[1].Clone()
                src_op3 = Inst1.GetUses()[2].Clone()

                inst = Instruction(
                    id=f"{Inst1.id}_lea64",
                    opcodes=["LEA64"],
                    operands=[dest_op, src_op1, src_op2, src_op3],
                    inst_content=f"LEA64 {dest_op.Name}, {src_op1.Name}, {src_op2.Name}, {src_op3.Name}",
                    parentBB=Inst1.parent
                )

                InsertInsts[Inst1] = inst

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                if (Inst1.GetUses()[0].IsReg and Inst2.GetUses()[2].IsReg):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[2]]))
                if (Inst1.GetUses()[1].IsReg and Inst2.GetUses()[1].IsReg):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[1]], Inst2.ReachingDefs[Inst2.GetUses()[1]]))

            elif Inst1.opcodes[0] == "IADD3" and Inst2.opcodes[0] == "IADD3":
                # (IADD3 R38, P1 = R36.reuse, c[0x0][0x168], RZ, IADD3.X R40 = R37.reuse, c[0x0][0x16c], RZ, P1)
                # => (IADD64 R38 = R36, c[0x0][0x168], RZ)
                dest_op = Inst1.GetDefs()[0].Clone()
                src_op1 = Inst1.GetUses()[0].Clone()
                src_op2 = Inst1.GetUses()[1].Clone()
                src_op3 = Inst1.GetUses()[2].Clone()

                inst = Instruction(
                    id=f"{Inst1.id}_iadd64",
                    opcodes=["IADD64"],
                    operands=[dest_op, src_op1, src_op2, src_op3],
                    inst_content=f"IADD64 {dest_op.Name}, {src_op1.Name}, {src_op2.Name}, {src_op3.Name}",
                    parentBB=Inst1.parent
                )

                InsertInsts[Inst1] = inst

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                if (Inst1.GetUses()[0].IsReg and Inst2.GetUses()[0].IsReg):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]]))
                if (Inst1.GetUses()[1].IsReg and Inst2.GetUses()[1].IsReg):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[1]], Inst2.ReachingDefs[Inst2.GetUses()[1]]))
                if (Inst1.GetUses()[2].IsReg and Inst2.GetUses()[2].IsReg):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[2]], Inst2.ReachingDefs[Inst2.GetUses()[2]]))

            elif Inst1.opcodes[0] == "IADD3" and Inst2.opcodes[0] == "LEA":
                # (IADD3 R95, P0 = R49.reuse, R93, RZ and LEA.HI.X.SX32 R102 = R49, R94, 0x1, P0) => (IADD64 R95 = R49 R93
                # Note R49 is 32bit, wherase R94:R93 is a 64bit value

                dest_op = Inst1.GetDefs()[0].Clone()
                src_op1 = Inst1.GetUses()[0].Clone()
                src_op2 = Inst1.GetUses()[1].Clone()

                inst = Instruction(
                    id=f"{Inst1.id}_iadd64",
                    opcodes=["IADD64"],
                    operands=[dest_op, src_op1, src_op2],
                    inst_content=f"IADD64 {dest_op.Name}, {src_op1.Name}, {src_op2.Name}",
                    parentBB=Inst1.parent
                )

                InsertInsts[Inst1] = inst

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                if (Inst1.GetUses()[0].IsReg and Inst2.GetUses()[0].IsReg):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]]))
                if (Inst1.GetUses()[1].IsReg and Inst2.GetUses()[1].IsReg):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[1]], Inst2.ReachingDefs[Inst2.GetUses()[1]]))

            elif Inst2.opcodes[0] == "SHF" and any(Inst2 == user for user, _ in Inst1.Users):
                # (IMAD R93 = R55, c[0x0][0x17c], RZ and SHF.R.S32.HI R94 = RZ, 0x1f, R93)
                # => (IMAD R93 = R55, c[0x0][0x17c], RZ and CAST64 R94 = R93)
                # Quick sanity check: second Inst uses first inst
                assert any(Inst2 == user for user, _ in Inst1.Users)

                dest_op = Inst1.GetDef().Clone()
                src_op = Inst1.GetUses()[0].Clone()
                src_op.SetReg(dest_op.Name + "_int32")

                Inst1.GetDef().SetReg(src_op.Name)

                inst = Instruction(
                    id=f"{Inst1.id}_cast64",
                    opcodes=["CAST64"],
                    operands=[dest_op, src_op],
                    inst_content=f"CAST64 {dest_op.Name}, {src_op.Name}",
                    parentBB=Inst1.parent
                )

                InsertInsts[Inst2] = inst
                RemoveInsts.add(Inst2)

            else:
                # Current pair not matching known patterns
                raise ValueError(f"Unhandled instruction pair: {Inst1} and {Inst2}")

            print(f"Processing pair: {Inst1} and {Inst2}")

            for InstPair in NextInstPairs:
                if InstPair[0] == InstPair[1] and "WIDE" not in InstPair[0].opcodes:
                    print(f"\tCast64 inserts: {InstPair[0]}")
                    Cast64Inserts.setdefault(Inst1, []).append(0)
                else:
                    print(f"\tNext pair: {InstPair[0]} and {InstPair[1]}")
                    Queue.append(InstPair)


    def ApplyChanges(self, func, Pack64Insts, Cast64Inserts, InsertInsts, RemoveInsts):

        # Remove Pack64Insts and handle register dependencies
        for Inst in Pack64Insts:
            mergeOp = Inst.GetUses()[0]
            for _, UseOp in Inst.Users:
                UseOp.SetReg(mergeOp.Name)

            RemoveInsts.add(Inst)

        vals64 = set()
        for inst in InsertInsts.values():
            for op in inst.GetDefs():
                vals64.add(op.Reg)

        for bb in func.blocks:

            new_insts = []

            for inst in bb.instructions:

                if inst in Cast64Inserts:
                    # Create the CAST64 instruction
                    use_indices = Cast64Inserts[inst]

                    for use_index in use_indices:
                        src_op_name = inst.GetUses()[use_index].Name
                        src_op = Operand(src_op_name, src_op_name, None, -1, True, False, False)


                        # Skip if the current reg will be a 64 value
                        if src_op_name in vals64:
                            continue

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
                        for UseOp in InsertInsts[inst].GetUses():
                            if UseOp.Reg == src_op_name:
                                UseOp.SetReg(dest_op_name)

                if inst in InsertInsts:
                    insertInst = InsertInsts[inst]
                    new_insts.append(insertInst)

                
                if inst not in RemoveInsts:
                    new_insts.append(inst)

            bb.instructions = new_insts
