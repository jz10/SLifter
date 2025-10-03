from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand
from collections import deque

class OperAggregate(SaSSTransform):
    # Apply operator aggregation on module 
    def apply(self, module):
        print("=== Start of Operator Aggregation Transformation ===")

        totalInsert = 0
        totalRemove = 0
        totalCast64 = 0

        for func in module.functions:

            InsertInsts = {}
            RemoveInsts = set()
            Cast64Inserts = {}
            Pack64Insts = self.GetPack64Instructions(func)
            Vals64 = set()
            
            for Inst in Pack64Insts:
                self.MergeInstructionPairs(Inst, Cast64Inserts, InsertInsts, RemoveInsts, Vals64)

            self.ApplyChanges(func, Pack64Insts, Cast64Inserts, InsertInsts, RemoveInsts)

            print(f"Function {func.name}:")
            print(f"OperAggregate Cast64 Inserts: {len(Cast64Inserts)}")
            print(f"OperAggregate Insert Instructions: {len(InsertInsts)}")
            print(f"OperAggregate Remove Instructions: {len(RemoveInsts)}")

            totalCast64 += len(Cast64Inserts)
            totalInsert += len(InsertInsts)
            totalRemove += len(RemoveInsts)

        print(f"Total OperAggregate Cast64 Inserts: {totalCast64}")
        print(f"Total OperAggregate Insert Instructions: {totalInsert}")
        print(f"Total OperAggregate Remove Instructions: {totalRemove}")
        
        print("=== End of Operator Aggregation Transformation ===")
            
    def GetPack64Instructions(self, func):
        Pack64Insts = []
        for bb in func.blocks:
            for inst in bb.instructions:
                if len(inst.opcodes) > 0 and inst.opcodes[0] == "PACK64":
                    Pack64Insts.append(inst)

        return Pack64Insts
    
    def AddCast64Insert(self, InsertBefore, AffectedInst, Op, Cast64Inserts, Vals64):

        src_op_name = Op.Name
        src_op = Op.Clone()

        # Skip if the current reg will be a 64 value
        if src_op_name in Vals64:
            return

        dest_op_name = src_op_name + "_int64"
        dest_op = Operand(dest_op_name, dest_op_name, None, None, True, False, False)

        inst_content = f"CAST64 {dest_op.Name}, {src_op.Name}"
        op_id = Op.Reg if Op.IsReg else Op.Name
        cast_inst = Instruction(
            id=f"{InsertBefore.id}_{op_id}cast64",
            opcodes=["CAST64"],
            operands=[dest_op, src_op],
            inst_content=inst_content,
            parentBB=InsertBefore.parent
        )

        # Update the instruction to use the new register
        for UseOp in AffectedInst.GetUses():
            if UseOp.Name == src_op_name:
                UseOp.Replace(dest_op)

        Cast64Inserts.setdefault(InsertBefore, []).append(cast_inst)

        print(f"\tCast64 insert for {AffectedInst} due to use of {Op}")

    def AddInsertInst(self, InsertBefore, NewInst, InsertInsts, Cast64Inserts, Vals64):
        print(f"\tInsert instruction {NewInst} before {InsertBefore}")
        
        skip = False
        if "IMAD" in InsertBefore.opcodes and "WIDE" in InsertBefore.opcodes:
            # IMAD.WIDE takes a 64-bit value in one of its use operands
            # For now we just assume a const memory operand will never need a cast64
            # TODO: need to generalize to considering 32/64 value operands
            skip = True

        if not skip:
            for op in NewInst.GetUses():
                if op.IsConstMem:
                    self.AddCast64Insert(InsertBefore, NewInst, op, Cast64Inserts, Vals64)
        InsertInsts[InsertBefore] = NewInst
        for op in NewInst.GetDefs():
            Vals64.add(op.Name)

        # Return the inserted instruction so callers can track it
        return NewInst

    def _insert_pack64_fallback(self, pack64_src_pair, last_insert_before, last_64_inst,
                                 Cast64Inserts, Vals64):

        if last_insert_before is None or last_64_inst is None:
            # Nothing to do; fallback only makes sense after at least one 64-bit
            # instruction has been inserted in this chain.
            print("\tNo prior 64-bit instruction to attach fallback PACK64; skipping.")
            return

        inst_lo, inst_hi = pack64_src_pair
        lo_def = inst_lo.GetDef().Clone()
        hi_def = inst_hi.GetDef().Clone()

        # Identify which operand of the last 64-bit instruction this pack belongs to.
        # Prefer matching the low part's register name.
        base_use = None
        for u in last_64_inst.GetUses():
            if u.Name == lo_def.Name or u.Name == hi_def.Name:
                base_use = u
                break
        # Fallback: if we cannot find a matching use, just take the first use.
        if base_use is None and len(last_64_inst.GetUses()) > 0:
            base_use = last_64_inst.GetUses()[0]

        if base_use is None:
            print("\tCould not determine target operand for fallback PACK64; skipping.")
            return

        # Create a canonical 64-bit pseudo-reg name and replace the use in place
        dest_name = base_use.Name + "_int64"
        dest_op = Operand(dest_name, dest_name, None, None, True, False, False)

        # Replace the operand in the last 64-bit instruction to consume the packed value
        base_use.Replace(dest_op)

        # Insert the PACK64 before the last inserted 64-bit instruction
        pack_inst = Instruction(
            id=f"{last_insert_before.id}_pack64_restore",
            opcodes=["PACK64"],
            operands=[dest_op, lo_def, hi_def],
            inst_content=f"PACK64 {dest_op.Name}, {lo_def.Name} {hi_def.Name}",
            parentBB=last_insert_before.parent
        )

        Cast64Inserts.setdefault(last_insert_before, []).append(pack_inst)
        Vals64.add(dest_name)
        print(f"\tInserted fallback PACK64 {pack_inst} before {last_insert_before}")


    def MergeInstructionPairs(self, Pack64Inst, Cast64Inserts, InsertInsts, RemoveInsts, Vals64):

        UseOps = Pack64Inst.GetUses()

        if not UseOps[0].IsWritableReg or not UseOps[1].IsWritableReg:
            return
        
        InstPair = (Pack64Inst.ReachingDefs[UseOps[0]], Pack64Inst.ReachingDefs[UseOps[1]])

        Queue = deque([InstPair])

        # Track the last successfully inserted 64-bit instruction and its position.
        last_insert_before = None
        last_inserted_64 = None

        while Queue:
            CurrPair = Queue.popleft()
            Inst1, Inst2 = CurrPair

            if Inst1 in RemoveInsts:
                # Already processed this instruction pair
                continue

            NextInstPairs = []

            if Inst1 == Inst2:
                print(f"Processing single: {Inst1}")
            else:
                print(f"Processing pair: {Inst1} and {Inst2}")

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


                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)
                
                # Cast64Inserts.setdefault(Inst1, []).append(0)
                self.AddCast64Insert(Inst1, inst, Inst1.GetUses()[0], Cast64Inserts, Vals64)

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1


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

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                for i in range(len(Inst1.GetUses())):
                    if Inst1.GetUses()[i].IsReg and not Inst1.GetUses()[i].IsRZ:
                        NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[i]], Inst2.ReachingDefs[Inst2.GetUses()[i]]))

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1


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

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[1]]))

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

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

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]]))

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

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

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                for i in range(len(Inst1.GetUses())):
                    if Inst1.GetUses()[i].IsReg and not Inst1.GetUses()[i].IsRZ and Inst2.GetUses()[i].IsReg and not Inst2.GetUses()[i].IsRZ:
                        NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[i]], Inst2.ReachingDefs[Inst2.GetUses()[i]]))

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

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

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)
                RemoveInsts.add(Inst3)

                # Special case: Instruction pairs should end at this point
                # This pattern always start from a 32-bit register
                # Cast64Inserts.setdefault(Inst3, []).append(0)
                self.AddCast64Insert(Inst3, inst, Inst3.GetUses()[0], Cast64Inserts, Vals64)

                last_inserted_64 = self.AddInsertInst(Inst3, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst3

            elif Inst1.opcodes[0] == "IMAD" and Inst2.opcodes[0] == "SHF":
                # Sign extension to 64 and shift left
                # (SHF.R.S32.HI R35 = RZ, 0x1f, R4
                # IMAD.SHL.U32 R36 = R4, 0x4, RZ
                # SHF.L.U64.HI R37 = R4, 0x2, R35)
                # => SHL64 R36 = R4, 0x2
                # Note: SHL is a hint to use bit shift to perform x4
                # It does not mean shift left by 4 bit
                Inst3 = Inst2.ReachingDefs[Inst2.GetUses()[2]]  # SHF.R.S32.HI instruction

                dest_op = Inst1.GetDef().Clone()
                src_op = Inst1.GetUses()[0].Clone()
                src_op2 = Inst2.GetUses()[1].Clone()
                inst = Instruction(
                    id=f"{Inst1.id}_shl64",
                    opcodes=["SHL64"],
                    operands=[dest_op, src_op, src_op2],
                    inst_content=f"SHL64 {dest_op.Name}, {src_op.Name}, {src_op2.Name}",
                    parentBB=Inst1.parent
                )

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)
                RemoveInsts.add(Inst3)

                # Special case: Instruction pairs should end at this point
                # This pattern always start from a 32-bit register
                # Cast64Inserts.setdefault(Inst1, []).append(0)
                self.AddCast64Insert(Inst1, inst, Inst1.GetUses()[0], Cast64Inserts, Vals64)

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

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
                imm_op = Operand(str(val), None, None, None, False, False, False, True, val)
                inst = Instruction(
                    id=f"{Inst1.id}_imad64",
                    opcodes=["IMAD64"],
                    operands=[dest_op, src_op, imm_op, src_op2],
                    inst_content=f"IMAD64 {dest_op.Name}, {src_op.Name}, {imm_op.Name}, {src_op2.Name}",
                    parentBB=Inst1.parent
                )

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)
                RemoveInsts.add(Inst3)

                # Special case: Instruction pairs should end at this point
                # This pattern always start from a 32-bit register
                self.AddCast64Insert(Inst3, inst, Inst3.GetUses()[0], Cast64Inserts, Vals64)

                last_inserted_64 = self.AddInsertInst(Inst3, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst3

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
                RemoveInsts.add(Inst1)

                # Special case: Instruction pairs should end at this point
                # This pattern always start from two 32-bit register
                for i in range(0, len(Inst1.GetUses())):
                    if Inst1.GetUses()[i].IsReg and not Inst1.GetUses()[i].IsRZ:
                        self.AddCast64Insert(Inst1, inst, Inst1.GetUses()[i], Cast64Inserts, Vals64)

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

                continue

            elif Inst1 == Inst2 and Inst1.opcodes[0] == "ULDC":
                # ULDC R4, R5= c[0x0][0x168] => ULDC64 R4 = c[0x0][0x168]

                dest_op = Inst1.GetDefs()[0].Clone()
                src_op = Inst1.GetUses()[0].Clone()

                inst = Instruction(
                    id=f"{Inst1.id}_uldc64",
                    opcodes=["ULDC64"],
                    operands=[dest_op, src_op],
                    inst_content=f"ULDC64 {dest_op.Name}, {src_op.Name}",
                    parentBB=Inst1.parent
                )
                RemoveInsts.add(Inst1)

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

                continue

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

                    RemoveInsts.add(Inst1)
                    RemoveInsts.add(Inst2)

                    if (Inst1.GetUses()[0].IsReg and not Inst1.GetUses()[0].IsRZ and Inst2.GetUses()[0].IsReg and not Inst2.GetUses()[0].IsRZ):
                        NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]]))
                    if (Inst1.GetUses()[1].IsReg and not Inst1.GetUses()[1].IsRZ and Inst2.GetUses()[2].IsReg and not Inst2.GetUses()[2].IsRZ):
                        NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[1]], Inst2.ReachingDefs[Inst2.GetUses()[2]]))

                    last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                    last_insert_before = Inst1

            elif Inst1.opcodes[0] == "LEA" and Inst2.opcodes[0] == "LEA":
                # Two patterns
                # (LEA R26, P0 = R14.reuse, c[0x0][0x168], 0x2, LEA.HI.X R28 = R14, c[0x0][0x16c], 0x2, P0) => (LEA64 R26 = R14, c[0x0][0x168], 0x2)
                # (LEA R62, P0 = R58, R52, 0x2, LEA.HI.X R64 = R58, R53, R59, 0x2, P0) => (LEA64 R62 = R52 0x2 R58)
                # Pattern 2 variant:
                # (LEA R26, P0 = R14.reuse, c[0x0][0x168], 0x2 and LEA.HI.X R29 = R14, c[0x0][0x16c], RZ, 0x2, P0)
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

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)


                if (Inst1.GetUses()[1].IsReg and not Inst1.GetUses()[1].IsRZ and Inst2.GetUses()[1].IsReg and not Inst2.GetUses()[1].IsRZ):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[1]], Inst2.ReachingDefs[Inst2.GetUses()[1]]))

                if (len(Inst2.GetUses()) > 4): # pattern 2
                    if (Inst1.GetUses()[0].IsReg and not Inst1.GetUses()[0].IsRZ and Inst2.GetUses()[2].IsReg and not Inst2.GetUses()[2].IsRZ):
                        NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[2]]))
                    else: # It is possible that Inst2.GetUses()[2] is RZ, effectively cast64 needed for Inst1.GetUses()[0]
                        if (Inst1.GetUses()[0].IsReg and not Inst1.GetUses()[0].IsRZ):
                            self.AddCast64Insert(Inst1, inst, Inst1.GetUses()[0], Cast64Inserts, Vals64)
                else: # pattern 1
                    if (Inst1.GetUses()[0].IsReg and not Inst1.GetUses()[0].IsRZ):
                        self.AddCast64Insert(Inst1, inst, Inst1.GetUses()[0], Cast64Inserts, Vals64)

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

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

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                if (Inst1.GetUses()[0].IsReg and not Inst1.GetUses()[0].IsRZ and Inst2.GetUses()[0].IsReg and not Inst2.GetUses()[0].IsRZ):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]]))
                if (Inst1.GetUses()[1].IsReg and not Inst1.GetUses()[1].IsRZ and Inst2.GetUses()[1].IsReg and not Inst2.GetUses()[1].IsRZ):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[1]], Inst2.ReachingDefs[Inst2.GetUses()[1]]))
                if (Inst1.GetUses()[2].IsReg and not Inst1.GetUses()[2].IsRZ and Inst2.GetUses()[2].IsReg and not Inst2.GetUses()[2].IsRZ):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[2]], Inst2.ReachingDefs[Inst2.GetUses()[2]]))

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

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

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                if (Inst1.GetUses()[0].IsReg and not Inst1.GetUses()[0].IsRZ and Inst2.GetUses()[0].IsReg and not Inst2.GetUses()[0].IsRZ):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[0]]))
                if (Inst1.GetUses()[1].IsReg and not Inst1.GetUses()[1].IsRZ and Inst2.GetUses()[1].IsReg and not Inst2.GetUses()[1].IsRZ):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[1]], Inst2.ReachingDefs[Inst2.GetUses()[1]]))

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1
          
            elif Inst1.opcodes[0] == "IMAD" and "WIDE" in Inst1.opcodes and Inst2.opcodes[0] == "IMAD" and "WIDE" not in Inst2.opcodes:
                # Handle 64-bit multiply-add
                # Pattern:
                #   Inst1: IMAD.WIDE.U32 Rlo, Rtmp_hi = R_mult32, R_mulcnd_lo, R_add32
                #   Inst2: IMAD           Rhi          = R_mult32, R_mulcnd_hi, Rtmp_hi
                # Becomes:
                #   IMAD64 Rlo = R_mult32, R_mulcnd_64, R_add32
                if Inst1.GetDefs()[1].Reg != Inst2.GetUses()[2].Reg:
                    raise ValueError(f"IMAD.WIDE pattern mismatch: tmp_hi register does not match between {Inst1} and {Inst2}")
                if Inst1.GetUses()[0].Reg != Inst2.GetUses()[0].Reg:
                    raise ValueError(f"IMAD.WIDE pattern mismatch: multiplier register does not match between {Inst1} and {Inst2}")

                dest_op = Inst1.GetDefs()[0].Clone()
              
                multiplier_op = Inst1.GetUses()[0].Clone()
                multiplicand_op = Inst1.GetUses()[1].Clone()
                addend_op = Inst1.GetUses()[2].Clone()

                inst = Instruction(
                    id=f"{Inst1.id}_imad64",
                    opcodes=["IMAD64"],
                    operands=[dest_op, multiplier_op, multiplicand_op, addend_op],
                    inst_content=f"IMAD64 {dest_op.Name}, {multiplier_op.Name}, {multiplicand_op.Name}, {addend_op.Name}",
                    parentBB=Inst1.parent
                )
              
                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                # The multiplicand is a 64-bit value composed of a low and high part.
                # If it's registers, trace them back. If a constant, AddInsertInst handles it.
                multiplicand_op_hi = Inst2.GetUses()[1]
                if multiplicand_op.IsReg and not multiplicand_op.IsRZ and multiplicand_op_hi.IsReg and not multiplicand_op_hi.IsRZ:
                    NextInstPairs.append((Inst1.ReachingDefs.get(multiplicand_op), Inst2.ReachingDefs.get(multiplicand_op_hi)))

                # The multiplier and addend are 32-bit values. They need to be cast to 64-bit.
                if multiplier_op.IsReg and not multiplier_op.IsRZ:
                    self.AddCast64Insert(Inst1, inst, multiplier_op, Cast64Inserts, Vals64)
              
                if addend_op.IsReg and not addend_op.IsRZ:
                    self.AddCast64Insert(Inst1, inst, addend_op, Cast64Inserts, Vals64)
              
                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

            elif Inst2.opcodes[0] == "SHF" and any(Inst2 == user for users in Inst1.Users.values() for user, _ in users):
                # (IMAD R93 = R55, c[0x0][0x17c], RZ and SHF.R.S32.HI R94 = RZ, 0x1f, R93)
                # => (IMAD R93 = R55, c[0x0][0x17c], RZ and CAST64 R94 = R93)
                # Quick sanity check: second Inst uses first inst
                assert any(Inst2 == user for users in Inst1.Users.values() for user, _ in users)

                dest_op = None
                inst1DefOp = None
                for defOp, useInsts in Inst1.Users.items():
                    for useInst, _ in useInsts:
                        if useInst == Inst2:
                            inst1DefOp = defOp
                            dest_op = defOp.Clone()
                            break
                    if dest_op:
                        break


                src_op = Inst1.GetUses()[0].Clone()
                src_op.SetReg(dest_op.Name + "_int32")

                inst1DefOp.SetReg(src_op.Name)

                inst = Instruction(
                    id=f"{Inst1.id}_cast64",
                    opcodes=["CAST64"],
                    operands=[dest_op, src_op],
                    inst_content=f"CAST64 {dest_op.Name}, {src_op.Name}",
                    parentBB=Inst1.parent
                )
                RemoveInsts.add(Inst2)

                InsertInsts[Inst2] = inst

            elif Inst1.opcodes[0] == "SHF" and Inst2.opcodes[0] == "SHF":
            # (SHF.L.U32 R8 = R666, 0x2, RZ, SHF.L.U64.HI R621 = R666.reuse, 0x2, R718) 
            # => SHL64 R8 = R666, 0x2
                if Inst1.opcodes[1] != Inst2.opcodes[1]:
                    raise ValueError(f"SHF pattern mismatch: shift direction does not match between {Inst1} and {Inst2}")
                
                dest_op = Inst1.GetDef().Clone()
                src_op = Inst1.GetUses()[0].Clone()
                imm_op = Inst1.GetUses()[1].Clone()

                if Inst1.opcodes[1] == "L":
                    opcodes = "SHL64"
                elif Inst1.opcodes[1] == "R":
                    opcodes = "SHR64"

                inst = Instruction(
                    id=f"{Inst1.id}_{opcodes.lower()}",
                    opcodes=[opcodes],
                    operands=[dest_op, src_op, imm_op],
                    inst_content=f"{opcodes} {dest_op.Name}, {src_op.Name}, {imm_op.Name}",
                    parentBB=Inst1.parent
                )

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)

                if (Inst1.GetUses()[0].IsReg and not Inst1.GetUses()[0].IsRZ and Inst2.GetUses()[2].IsReg and not Inst2.GetUses()[2].IsRZ):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[0]], Inst2.ReachingDefs[Inst2.GetUses()[2]]))

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

            elif Inst1.opcodes[0] == "LEA" and Inst2.opcodes[0] == "IMAD" and "X" in Inst2.opcodes:
            # (SHF.L.U64.HI R37 = R2, 0x2, R3,
            # LEA R6, P0 = R2, R6, 0x2 and IMAD.X R7 = R7, 0x1, R37, P7)
            # => (LEA64 R6 = R2, R6, 0x2)

                dest_op = Inst1.GetDefs()[0].Clone()
                src_op1 = Inst1.GetUses()[0].Clone()
                src_op2 = Inst1.GetUses()[1].Clone()
                imm_op = Inst1.GetUses()[2].Clone()

                Inst3 = Inst2.ReachingDefs[Inst2.GetUses()[2]]  # SHF.L.U64.HI instruction
                if Inst3.opcodes[0] != "SHF" or Inst3.opcodes[1] != "L" or Inst3.opcodes[2] != "U64":
                    raise ValueError("Expected SHF.L.U64 instruction as the third instruction in the pair")

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)
                RemoveInsts.add(Inst3)

                inst = Instruction(
                    id=f"{Inst1.id}_lea64",
                    opcodes=["LEA64"],
                    operands=[dest_op, src_op1, src_op2, imm_op],
                    inst_content=f"LEA64 {dest_op.Name} = {src_op1.Name}, {src_op2.Name}, {imm_op.Name}",
                    parentBB=Inst1.parent
                )

                if (Inst1.GetUses()[1].IsReg and not Inst1.GetUses()[1].IsRZ and Inst2.GetUses()[0].IsReg and not Inst2.GetUses()[0].IsRZ):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[1]], Inst2.ReachingDefs[Inst2.GetUses()[0]]))

                if (Inst3.GetUses()[0].IsReg and not Inst3.GetUses()[0].IsRZ and Inst3.GetUses()[2].IsReg and not Inst3.GetUses()[2].IsRZ):
                    NextInstPairs.append((Inst3.ReachingDefs[Inst3.GetUses()[0]], Inst3.ReachingDefs[Inst3.GetUses()[2]]))

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

            elif Inst1.opcodes[0] == "LEA" and Inst2.opcodes[0] == "IADD3" and "X" in Inst2.opcodes:
            # (SHF.L.U64.HI R37 = R2, 0x2, R3,
            # LEA R6, P0 = R2, R6, 0x2 and IADD3.X R7 = R7, R37, RZ, P7)
            # => (LEA64 R6 = R2, R6, 0x2)

                dest_op = Inst1.GetDefs()[0].Clone()
                src_op1 = Inst1.GetUses()[0].Clone()
                src_op2 = Inst1.GetUses()[1].Clone()
                imm_op = Inst1.GetUses()[2].Clone()

                Inst3 = Inst2.ReachingDefs[Inst2.GetUses()[1]]  # SHF.L.U64.HI instruction
                if Inst3.opcodes[0] != "SHF" or Inst3.opcodes[1] != "L" or Inst3.opcodes[2] != "U64":
                    raise ValueError("Expected SHF.L.U64 instruction as the third instruction in the pair")

                RemoveInsts.add(Inst1)
                RemoveInsts.add(Inst2)
                RemoveInsts.add(Inst3)

                inst = Instruction(
                    id=f"{Inst1.id}_lea64",
                    opcodes=["LEA64"],
                    operands=[dest_op, src_op1, src_op2, imm_op],
                    inst_content=f"LEA64 {dest_op.Name} = {src_op1.Name}, {src_op2.Name}, {imm_op.Name}",
                    parentBB=Inst1.parent
                )

                if (Inst1.GetUses()[1].IsReg and not Inst1.GetUses()[1].IsRZ and Inst2.GetUses()[0].IsReg and not Inst2.GetUses()[0].IsRZ):
                    NextInstPairs.append((Inst1.ReachingDefs[Inst1.GetUses()[1]], Inst2.ReachingDefs[Inst2.GetUses()[0]]))

                if (Inst3.GetUses()[0].IsReg and not Inst3.GetUses()[0].IsRZ and Inst3.GetUses()[2].IsReg and not Inst3.GetUses()[2].IsRZ):
                    NextInstPairs.append((Inst3.ReachingDefs[Inst3.GetUses()[0]], Inst3.ReachingDefs[Inst3.GetUses()[2]]))

                last_inserted_64 = self.AddInsertInst(Inst1, inst, InsertInsts, Cast64Inserts, Vals64)
                last_insert_before = Inst1

            else:
                # Current pair not matching known patterns. Instead of failing,
                # re-introduce a PACK64 right before the last successfully inserted
                # 64-bit instruction to preserve correctness.
                self._insert_pack64_fallback((Inst1, Inst2), last_insert_before, last_inserted_64,
                                             Cast64Inserts, Vals64)
                # Stop processing this PACK64 chain after fallback insertion.
                return

            for InstPair in NextInstPairs:
                if InstPair[0] == InstPair[1] and "WIDE" not in InstPair[0].opcodes and InstPair[0].opcodes[0] != "ULDC":
                    # Cast64Inserts.setdefault(Inst1, []).append(0)
                    self.AddCast64Insert(InstPair[0], InstPair[0], InstPair[0].GetUses()[0], Cast64Inserts, Vals64)
                else:
                    print(f"\tNext pair: {InstPair[0]} and {InstPair[1]}")
                    Queue.append(InstPair)


    def ApplyChanges(self, func, Pack64Insts, Cast64Inserts, InsertInsts, RemoveInsts):

        # Remove Pack64Insts and handle register dependencies
        for Inst in Pack64Insts:
            mergeOp = Inst.GetUses()[0]
            for users in Inst.Users.values():
                for _, UseOp in users:
                    UseOp.SetReg(mergeOp.Name)

            RemoveInsts.add(Inst)

        for bb in func.blocks:

            new_insts = []

            for inst in bb.instructions:

                if inst in Cast64Inserts:
                    new_insts.extend(Cast64Inserts[inst])

                if inst in InsertInsts:
                    insertInst = InsertInsts[inst]
                    new_insts.append(insertInst)

                
                if inst not in RemoveInsts:
                    new_insts.append(inst)

            bb.instructions = new_insts
