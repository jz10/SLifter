from transform.transform import SaSSTransform
from sir.operand import Operand
from collections import deque
from sir.instruction import Instruction

class TypeAnalysis(SaSSTransform):
    def __init__(self, name):
        super().__init__(name)

        # operands with PROP must have the same type
        # operands with PROP_PTR must have the same type but with _PTR suffix
        # if no propagate type is found, default pointer type to PTR
        # operands with ANY can have any type
        self.instructionTypeTable = {
            "FADD": ["Float32", "Float32", "Float32", "NA", "NA", "NA"],
            "FFMA": ["Float32", "Float32", "Float32", "Float32", "NA", "NA"],
            "FMUL": ["Float32", "Float32", "Float32", "NA", "NA", "NA"],
            "FSETP": ["Int1", "PROP", "PROP", "PROP", "PROP", "NA"],
            "MUFU": ["Float32", "Float32", "Float32", "NA", "NA", "NA"],
            "S2R": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "IMAD": ["Int32", "Int32", "Int32", "Int32", "NA", "NA"],
            "IADD3": ["Int32", "Int32", "Int32", "Int32", "NA", "NA"],
            "XMAD": ["Int32", "Int32", "Int32", "Int32", "NA", "NA"],
            "IADD32I": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "MOV": ["Int32", "Int32", "NA", "NA", "NA", "NA"],
            "IADD": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "ISETP": ["Int1", "PROP", "PROP", "PROP", "PROP", "NA"],
            "AND": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "OR": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "XOR": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "NOT": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "LEA": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "SHL": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "SHR": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "LOP3": ["PROP", "PROP", "PROP", "PROP", "PROP", "PROP"],
            "LDG": ["PROP", "PROP_PTR", "NA", "NA", "NA", "NA"],
            "LD": ["PROP", "PROP_PTR", "NA", "NA", "NA", "NA"],
            "SULD": ["PROP", "PROP_PTR", "NA", "NA", "NA", "NA"],
            "STG": ["PROP_PTR", "PROP", "NA", "NA", "NA", "NA"],
            "ST" : ["PROP_PTR", "PROP", "NA", "NA", "NA", "NA"],
            "SUST": ["PROP_PTR", "PROP", "NA", "NA", "NA", "NA"],
            "F2I": ["Int32", "Float32", "NA", "NA", "NA", "NA"],
            "I2F": ["Float32", "Int32", "NA", "NA", "NA", "NA"],
            "NOP": ["NA", "NA", "NA", "NA", "NA", "NA"],
            "BRA": ["NA", "NA", "NA", "NA", "NA", "NA"],
            "EXIT": ["NA", "NA", "NA", "NA", "NA", "NA"],
            "RET": ["NA", "NA", "NA", "NA", "NA", "NA"],
            "SYNC": ["NA", "NA", "NA", "NA", "NA", "NA"],
            "BAR": ["NA", "NA", "NA", "NA", "NA", "NA"],
            "SSY": ["NA", "NA", "NA", "NA", "NA", "NA"],
            "SHF": ["Int32", "Int32", "Int32", "Int32", "NA", "NA"],
            "DEPBAR": ["NA", "NA", "NA", "NA", "NA", "NA"],
            "LOP32I": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "ISCADD": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "MOV32I": ["Int32", "Int32", "NA", "NA", "NA", "NA"],
            "IABS": ["Int32", "Int32", "NA", "NA", "NA", "NA"],
            "ULDC": ["PROP", "PROP", "NA", "NA", "NA", "NA"],
            "DMUL": ["Float64", "Float64", "Float64", "NA", "NA", "NA"],
            "DFMA": ["Float64", "Float64", "Float64", "Float64", "NA", "NA"],
            "LDS": ["PROP", "Int32", "NA", "NA", "NA", "NA"],
            "STS": ["Int32", "PROP", "NA", "NA", "NA", "NA"],
            "MATCH": ["Int32", "Int32", "NA", "NA", "NA"],
            "BREV" : ["Int32", "Int32", "NA", "NA", "NA", "NA"],
            "FLO": ["Int32", "Int32", "NA", "NA", "NA", "NA"],
            "POPC": ["Int32", "Int32", "NA", "NA", "NA", "NA"],
            "RED": ["Int64", "Int32", "NA", "NA", "NA", "NA"],
            "IMNMX": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "PRMT": ["Int32", "Int32", "Int32", "Int32", "NA", "NA"],
            "PLOP3": ["PROP", "PROP", "PROP", "PROP", "PROP", "PROP"],
            "HMMA" : ["Float32"] * 12,
            "MOVM" : ["PROP", "PROP", "NA", "NA", "NA", "NA"],


            # Uniform variants
            "USHF": ["Int32", "Int32", "Int32", "Int32", "NA", "NA"],
            "ULEA": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "ULOP3": ["PROP", "PROP", "PROP", "PROP", "PROP", "PROP"],
            "UIADD3": ["Int32", "Int32", "Int32", "Int32", "NA", "NA"],
            "UMOV": ["Int32", "Int32", "NA", "NA", "NA", "NA"],

            # Dummy instruction types
            "PHI": ["PROP", "PROP", "PROP", "PROP", "NA", "NA"],
            "INTTOPTR": ["PROP_PTR", "Int64", "NA", "NA", "NA", "NA"],
            "PACK64": ["Int64", "Int32", "Int32", "NA", "NA", "NA"],
            "UNPACK64": ["Int32", "Int64", "NA", "NA", "NA", "NA"],
            "CAST64": ["Int64", "Int32", "NA", "NA", "NA", "NA"],
            "IADD64": ["Int64", "Int64", "Int64", "NA", "NA", "NA"],
            "IMAD64": ["Int64", "Int32", "Int32", "Int64", "NA", "NA"],
            "SHL64": ["Int64", "Int64", "Int64", "NA", "NA", "NA"],
            "MOV64": ["Int64", "Int64", "NA", "NA", "NA", "NA"],
            "IADD32I64": ["Int64", "Int64", "Int64", "NA", "NA", "NA"],
            "PHI64": ["Int64", "Int64", "Int64", "Int64", "Int64", "NA"],
            "BITCAST": ["ANY", "ANY", "NA", "NA", "NA", "NA"],
            "PBRA": ["Int1", "NA", "NA", "NA", "NA", "NA"],
            "LEA64": ["Int64", "Int64", "Int64", "Int64", "NA", "NA"],
            "SETZERO": ["PROP", "NA", "NA", "NA", "NA", "NA"],
            "ULDC64": ["PROP", "PROP", "NA", "NA", "NA", "NA"],
            "LDG64": ["Int64", "Int64_PTR", "NA", "NA", "NA", "NA"],
            "SHR64": ["Int64", "Int64", "Int64", "NA", "NA", "NA"],
            "ISETP64": ["Int1", "PROP", "PROP", "PROP", "PROP", "NA"],
            "IADD364": ["Int64", "Int64", "Int64", "Int64", "NA", "NA"],
        }

        self.modifierOverrideTable = {
            "MATCH": {
                "U32": ["ANY", "Int32"],
                "U64": ["ANY", "Int64"]
            },
            "IMNMX": {
                "U32": ["Int32", "Int32", "Int32"],
                "U64": ["Int64", "Int64", "Int64"]
            },
            "HMMA": {
                "F32": ["Float32"] * 12,
                "F16": ["Float16"] * 12,
            }
        }


    def apply(self, module):
        for func in module.functions:
            self.ProcessFunc(func)


    def ProcessFunc(self, function):
        WorkList = self.TraverseCFG(function)

        OpTypes = {}
        print("=== Start of TypeAnalysis ===")

        Changed = True

        Iteration = 0

        self.Conflicts = {}
        # Track registers involved in type conflicts and synthetic bitcast temps
        self.ConflictedOriginalRegs = set()
        self.BitcastRegs = set()
        
        while Changed:

            # for BB in WorkList:
            #     for Inst in BB.instructions:
            #         print(str(Inst)+" => ", end="")
            #         for Operand in Inst.operands:
            #             if Operand in OpTypes:
            #                 TypeDesc = OpTypes[Operand]
            #             elif Operand.Reg in OpTypes:
            #                 TypeDesc = OpTypes[Operand.Reg]
            #             else:
            #                 TypeDesc = "NOTYPE"
            #             print(TypeDesc+", ",end="")
            #         print("")

            # print("-----next iteration-----")

            Changed = False

            for BB in WorkList:
                Changed |= self.ProcessBB(BB, OpTypes, False)
            
            for BB in reversed(WorkList):
                Changed |= self.ProcessBB(BB, OpTypes, True)

            # Resolve conflict by inserting bitcast
            for BB in WorkList:
                NewInstructions = []
                for Inst in BB.instructions:
                    if Inst in self.Conflicts:
                        op, OldType, NewType = self.Conflicts[Inst]
                        print(f"Warning: Inserting BITCAST to resolve type conflict for {op} in {Inst}: {OldType} vs {NewType}")
                        # Insert bitcast before Inst
                        orig_reg = op.Reg
                        NewRegName = f"{orig_reg}_bitcast"

                        BitcastReg = Operand.fromReg(NewRegName, NewRegName)

                        BitcastInst = Instruction(
                            id=f"{Inst.id}_type_resolve", 
                            opcodes=["BITCAST"],
                            operands=[BitcastReg, op],
                            inst_content=f"BITCAST {BitcastReg}, {orig_reg}",
                            parentBB=BB
                        )
                        NewInstructions.append(BitcastInst)

                        # Book-keeping for statistics
                        self.ConflictedOriginalRegs.add(orig_reg)
                        self.BitcastRegs.add(NewRegName)

                        OpTypes[BitcastReg.Reg] = NewType
                        OpTypes[orig_reg] = OldType
                        op.SetReg(BitcastReg.Reg)

                        del self.Conflicts[Inst]

                    NewInstructions.append(Inst)

                BB.instructions = NewInstructions

            Iteration += 1
            if Iteration > 5:
                print("Warning: TypeAnalysis exceeds 5 iterations, stopping")
                break

        # Apply types to instructions
        for BB in WorkList:
            for Inst in BB.instructions:
                for op in Inst.operands:
                    if op in OpTypes:
                        op._TypeDesc = OpTypes[op]
                    elif op.Reg in OpTypes:
                        op._TypeDesc = OpTypes[op.Reg]
                    else:
                        op._TypeDesc = "NOTYPE"


        for BB in WorkList:
            for Inst in BB.instructions:
                print(str(Inst)+" => ", end="")
                for op in Inst.operands:
                    print(op._TypeDesc+", ",end="")
                print("")

        # Statistics
        # Build AllRegs from the function CFG, then classify:
        # - If reg in ConflictedOriginalRegs -> Conflicted
        # - Else if reg in BitcastRegs -> skip (synthetic)
        # - Else if reg has a type in OpTypes -> count by that type
        # - Else -> Unresolved
        AllRegs = set()
        for BB in WorkList:
            for Inst in BB.instructions:
                for op in Inst.operands:
                    if op.IsWritableReg:
                        AllRegs.add(op.Reg)

        TypeCount = {}
        for reg in AllRegs:
            # Skip synthetic bitcast temps entirely
            if reg in self.BitcastRegs:
                continue

            if reg in self.ConflictedOriginalRegs:
                TypeCount["Conflicted"] = TypeCount.get("Conflicted", 0) + 1
                continue

            if reg in OpTypes and not isinstance(reg, Operand):
                tdesc = OpTypes[reg]
                TypeCount[tdesc] = TypeCount.get(tdesc, 0) + 1
            else:
                print(f"Warning: Unresolved type for register {reg}")
                TypeCount["Unresolved"] = TypeCount.get("Unresolved", 0) + 1

        print("Type analysis statistics")
        for type_desc, count in TypeCount.items():
            print(f"Type counts {type_desc}: {count} registers")

        print("done")
        print("=== End of TypeAnalysis ===")


    def TraverseCFG(self, function):
        EntryBB = function.blocks[0]
        Visited = set()
        Queue = deque([EntryBB])
        WorkList = []

        while Queue:
            CurrBB = Queue.popleft()
            
            if CurrBB in Visited:
                continue

            Visited.add(CurrBB)
            WorkList.append(CurrBB)

            for SuccBB in CurrBB._succs:
                if SuccBB not in Visited:
                    Queue.append(SuccBB)

        return WorkList
    

    def printTypes(self, types):
        print("{")
        for reg, type_desc in types.items():
            print(f"{reg}: {type_desc}")

        print("}")
        

    def ProcessBB(self, BB, OpTypes, Reversed):
        OldOpTypes = OpTypes.copy()

        if Reversed:
            Instructions = reversed(BB.instructions)
        else:
            Instructions = BB.instructions

        for Inst in Instructions:
            self.PropagateTypes(Inst, OpTypes)
        
        Changed = (OldOpTypes != OpTypes)
        return Changed

    def SetOptype(self, OpTypes, Operand, TypeDesc):
        # Do not overwrite e.g. Int32_PTR with generic PTR
        existing = self.GetOptype(OpTypes, Operand)
        if TypeDesc == "PTR" and existing and existing.endswith("_PTR"):
            return

        if Operand.IsWritableReg:
            OpTypes[Operand.Reg] = TypeDesc
        else:
            # Store operand itself as key for non-register values
            # E.g. 0 in IADD vs 0 in FADD have different types
            OpTypes[Operand] = TypeDesc

    def GetOptype(self, OpTypes, Operand):
        if Operand in OpTypes:
            return OpTypes[Operand]
        elif Operand.Reg in OpTypes:
            return OpTypes[Operand.Reg]
        else:
            return "NOTYPE"
        
    def TypeConflict(self, TypeA, TypeB):
        if TypeA == TypeB:
            return False

        if TypeA == "NOTYPE" or TypeB == "NOTYPE":
            return False

        if TypeA == "ANY" or TypeB == "ANY":
            return False

        if TypeA == "NA" or TypeB == "NA":
            return False

        if TypeA.endswith("PTR") and TypeB.endswith("PTR"):
            baseA = TypeA[:-3]
            baseB = TypeB[:-3]
            if baseA == baseB:
                return False
            if baseA == "" or baseB == "":
                return False

        return True

    def ModifierOverride(self, Inst, i):
        op = Inst.opcodes[0]

        if op not in self.modifierOverrideTable:
            return None

        modifierTable = self.modifierOverrideTable[op]

        for mod in Inst.opcodes[1:]:
            if mod in modifierTable:
                typeArr = modifierTable[mod]
                if i < len(typeArr):
                    return typeArr[i]

        return None

        
    def ResolveType(self, Inst, i):
        op = Inst._opcodes[0]

        # Special case for the predicate register in instructions with multiple defs
        # E.g. LEA R12, P0 = R6.reuse, R4, 0x2, because of P0, decrease index by 1 to get correct entry in table
        typeDesc = "NA"
        if i < len(self.instructionTypeTable[op]):
            if len(Inst.GetDefs()) > 1 and i > 1:
                typeDesc = self.instructionTypeTable[op][i-1]
            else:
                typeDesc = self.instructionTypeTable[op][i]

        # Modifier overrides
        modRewrite = self.ModifierOverride(Inst, i)
        if modRewrite:
            typeDesc = modRewrite
            
        # Default PROP for second def operands
        # TODO: make the table work with variable def operands instead of just 1
        if i == 1 and len(Inst.GetDefs()) == 2:
            typeDesc = "PROP"

        # Operand overrides
        if Inst.operands[i].IsPredicateReg or Inst.operands[i].IsPT:
            typeDesc = "Int1"

        return typeDesc

    def PropagateTypes(self, Inst, OpTypes):
        # Get Inst opcode
        op = Inst._opcodes[0]

        if op not in self.instructionTypeTable:
            print(f"Warning: Unhandled opcode {op} in {Inst}")
            return
        
        # Static resolve types
        propType = "NOTYPE"
        for i, operand in enumerate(Inst.operands):
            typeDesc = self.ResolveType(Inst, i)

            if typeDesc != "NA" and "PROP" not in typeDesc and typeDesc != "ANY":

                if operand.Reg in OpTypes and self.TypeConflict(OpTypes[operand.Reg], typeDesc):
                    print(f"Warning: Type mismatch for {operand.Reg} in {Inst}: {OpTypes[operand.Reg]} vs {typeDesc}")
                    self.Conflicts[Inst] = (operand, OpTypes[operand.Reg], typeDesc)
                    return

                self.SetOptype(OpTypes, operand, typeDesc)

        # Find propagate type
        for i, operand in enumerate(Inst.operands):
            typeDesc = self.ResolveType(Inst, i)
            if typeDesc == "PROP":
                existing = self.GetOptype(OpTypes, operand)
                if existing != "NOTYPE":
                    if propType != "NOTYPE" and self.TypeConflict(existing, propType):
                        print(f"Warning: Propagation type mismatch for {operand} in {Inst}: {existing} vs {propType}")
                        self.Conflicts[Inst] = (operand, existing, propType)
                        return

                    propType = existing

        # Propagate types
        for i, operand in enumerate(Inst.operands):
            typeDesc = self.ResolveType(Inst, i)

            if typeDesc == "PROP" and propType != "NOTYPE":
                self.SetOptype(OpTypes, operand, propType)
            elif typeDesc == "PROP_PTR":
                if propType == "NOTYPE" or propType == "ANY" or propType == "NA":
                    self.SetOptype(OpTypes, operand, "PTR")
                else:
                    self.SetOptype(OpTypes, operand, propType + "_PTR")
