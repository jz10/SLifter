from transform.transform import SaSSTransform
from sir.operand import Operand
from collections import deque
from sir.instruction import Instruction

class TypeAnalysis(SaSSTransform):
    def __init__(self):
        super().__init__()

        # operands with PROP must have the same type
        # operands with PROP_PTR must have the same type but with _PTR suffix(unused for now)
        # address operands are represented as Int64 and cast to pointers in the lifter
        # operands with ANY can have any type
        self.instructionTypeTable = {
            "FADD": [["Float32"], ["Float32", "Float32"]],
            "FFMA": [["Float32"], ["Float32", "Float32", "Float32"]],
            "FMUL": [["Float32"], ["Float32", "Float32"]],
            "FSETP": [["Int1", "Int1"], ["Float32", "Float32", "Int1"]],
            "FSEL": [["Float32"], ["Float32", "Float32", "Int1"]],
            "MUFU": [["Float32"], ["Float32"]],
            "S2R": [["Int32"], ["Int32", "Int32"]],
            "IMAD": [["Int32"], ["Int32", "Int32", "Int32"]],
            "IADD3": [["Int32"], ["Int32", "Int32", "Int32"]],
            "XMAD": [["Int32"], ["Int32", "Int32", "Int32"]],
            "IADD32I": [["Int32"], ["Int32", "Int32"]],
            "MOV": [["PROP"], ["PROP"]],
            "IADD": [["Int32"], ["Int32", "Int32"]],
            "ISETP": [["Int1"], ["PROP", "PROP", "PROP", "PROP"]],
            "AND": [["PROP"], ["PROP", "PROP"]],
            "OR": [["PROP"], ["PROP", "PROP"]],
            "XOR": [["PROP"], ["PROP", "PROP"]],
            "NOT": [["PROP"], ["PROP", "PROP"]],
            "LEA": [["Int32"], ["Int32", "Int32"]],
            "SHL": [["PROP"], ["PROP", "PROP"]],
            "SHR": [["PROP"], ["PROP", "PROP"]],
            "LOP3": [["PROP"], ["PROP", "PROP", "PROP", "PROP", "PROP"]],
            "LDG": [["PROP"], ["Int64"]],
            "LD": [["PROP"], ["Int64"]],
            "SULD": [["PROP"], ["Int64"]],
            "STG": [[], ["Int64", "PROP"]],
            "ST": [[], ["Int64", "PROP"]],
            "SUST": [[], ["Int64", "PROP"]],
            "F2I": [["Int32"], ["Float32"]],
            "I2F": [["Float32"], ["Int32"]],
            "SEL": [["Int32"], ["Int32", "Int32", "Int1"]],
            "NOP": [[], []],
            "BRA": [[], []],
            "EXIT": [[], []],
            "RET": [[], []],
            "SYNC": [[], []],
            "BAR": [[], []],
            "SSY": [[], []],
            "SHF": [["Int32"], ["Int32", "Int32", "Int32"]],
            "SHFL": [["Int1", "Int32"], ["Int32", "Int32", "Int32"]],
            "DEPBAR": [[], []],
            "LOP32I": [["Int32"], ["Int32", "Int32"]],
            "ISCADD": [["Int32"], ["Int32", "Int32"]],
            "MOV32I": [["Int32"], ["Int32"]],
            "IABS": [["Int32"], ["Int32"]],
            "ULDC": [["PROP"], ["PROP"]],
            "DMUL": [["Float64"], ["Float64", "Float64"]],
            "DFMA": [["Float64"], ["Float64", "Float64", "Float64"]],
            "LDS": [["PROP"], ["Int32"]],
            "STS": [[], ["Int32", "PROP"]],
            "VOTE": [["Int32"], ["Int1", "Int1"]],
            "VOTEU": [["Int32"], ["Int1", "Int1"]],
            "MATCH": [["Int32"], ["Int32"]],
            "BREV": [["Int32"], ["Int32"]],
            "FLO": [["Int32"], ["Int32"]],
            "POPC": [["Int32"], ["Int32"]],
            "RED": [[], ["Int64", "Int32"]],
            "IMNMX": [["Int32"], ["Int32", "Int32"]],
            "PRMT": [["Int32"], ["Int32", "Int32", "Int32"]],
            "PLOP3": [["PROP"], ["PROP", "PROP", "PROP", "PROP", "PROP"]],
            "HMMA": [["Float32"] * 4, ["Float32"] * 8],
            "MOVM": [["PROP"], ["PROP"]],

            # Uniform variants
            "USHF": [["Int32"], ["Int32", "Int32", "Int32"]],
            "ULEA": [["Int32"], ["Int32", "Int32"]],
            "ULOP3": [["PROP"], ["PROP", "PROP", "PROP", "PROP", "PROP"]],
            "UIADD3": [["Int32"], ["Int32", "Int32", "Int32"]],
            "UMOV": [["Int32"], ["Int32"]],

            # Dummy instruction types
            "PHI": [["PROP"], ["PROP", "PROP", "PROP"]],
            "PACK64": [["Int64"], ["Int32", "Int32"]],
            "UNPACK64": [["Int32"], ["Int64"]],
            "CAST64": [["Int64"], ["Int32"]],
            "IADD64": [["Int64"], ["Int64", "Int64"]],
            "IMAD64": [["Int64"], ["Int32", "Int32", "Int64"]],
            "SHL64": [["Int64"], ["Int64", "Int64"]],
            "MOV64": [["Int64"], ["Int64"]],
            "IADD32I64": [["Int64"], ["Int64", "Int64"]],
            "PHI64": [["Int64"], ["Int64", "Int64", "Int64", "Int64"]],
            "BITCAST": [["ANY"], ["ANY"]],
            "PBRA": [[], ["Int1"]],
            "LEA64": [["Int64"], ["Int64", "Int64", "Int64"]],
            "SETZERO": [["PROP"], []],
            "ULDC64": [["PROP"], ["PROP"]],
            "LDG64": [["Int64"], ["Int64"]],
            "SHR64": [["Int64"], ["Int64", "Int64"]],
            "ISETP64": [["Int1"], ["PROP", "PROP", "PROP", "PROP"]],
            "IADD364": [["Int64"], ["Int64", "Int64", "Int64"]],
        }

        self.modifierOverrideTable = {
            "MATCH": {
                "U32": [["ANY"], ["Int32"]],
                "U64": [["ANY"], ["Int64"]],
            },
            "IMNMX": {
                "U32": [["Int32"], ["Int32", "Int32"]],
                "U64": [["Int64"], ["Int64", "Int64"]],
            },
            "HMMA": {
                "F32": [["Float32"] * 4, ["Float32"] * 8],
                "F16": [["Float16"] * 4, ["Float16"] * 8],
            },
        }


    def apply(self, module):
        super().apply(module)
        print("=== Start of TypeAnalysis ===")
        for func in module.functions:
            print(f"Processing function: {func.name}")
            self.ProcessFunc(func)
        print("=== End of TypeAnalysis ===")


    def ProcessFunc(self, function):
        WorkList = self.TraverseCFG(function)

        OpTypes = {}

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

                        SrcReg = op.Clone()
                        BitcastReg = Operand.fromReg(NewRegName, NewRegName)

                        BitcastInst = Instruction(
                            id=f"{Inst.id}_type_resolve", 
                            opcodes=["BITCAST"],
                            operands=[BitcastReg, SrcReg],
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
                        op.TypeDesc = OpTypes[op]
                    elif op.Reg in OpTypes:
                        op.TypeDesc = OpTypes[op.Reg]
                    else:
                        op.TypeDesc = "NOTYPE"


        for BB in WorkList:
            for Inst in BB.instructions:
                print(str(Inst)+" => ", end="")
                for op in Inst.operands:
                    print(op.TypeDesc+", ",end="")
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

    def ModifierOverride(self, Inst, idx, is_def):
        op = Inst.opcodes[0]

        if op not in self.modifierOverrideTable:
            return None

        modifierTable = self.modifierOverrideTable[op]

        for mod in Inst.opcodes[1:]:
            if mod in modifierTable:
                def_override, use_override = modifierTable[mod]
                if is_def:
                    if idx < len(def_override):
                        return def_override[idx]
                else:
                    if idx < len(use_override):
                        return use_override[idx]

        return None

        
    def ResolveType(self, Inst, operand, idx, defs_len, is_def):
        op = Inst._opcodes[0]
        def_types, use_types = self.instructionTypeTable[op]
        type_list = def_types if is_def else use_types

        typeDesc = "NA"
        if idx < len(type_list):
            typeDesc = type_list[idx]

        # Modifier overrides
        modRewrite = self.ModifierOverride(Inst, idx, is_def)
        if modRewrite:
            typeDesc = modRewrite

        # Operand overrides
        if operand.IsPredicateReg or operand.IsPT:
            typeDesc = "Int1"

        return typeDesc

    def PropagateTypes(self, Inst, OpTypes):
        # Get Inst opcode
        op = Inst._opcodes[0]

        if op not in self.instructionTypeTable:
            print(f"Warning: Unhandled opcode {op} in {Inst}")
            return

        defs = Inst.GetDefs()
        uses = Inst.GetUses()
        defs_len = len(defs)

        resolved_def_types = [
            self.ResolveType(Inst, operand, idx, defs_len, True)
            for idx, operand in enumerate(defs)
        ]
        resolved_use_types = [
            self.ResolveType(Inst, operand, idx, defs_len, False)
            for idx, operand in enumerate(uses)
        ]

        def_pairs = list(zip(defs, resolved_def_types))
        use_pairs = list(zip(uses, resolved_use_types))

        # Static resolve types
        for operand, typeDesc in def_pairs + use_pairs:
            if typeDesc != "NA" and "PROP" not in typeDesc and typeDesc != "ANY":
                if not Inst.IsPhi(): # Propagate conflicts to somewhere else, not in PHI
                    if operand.Reg in OpTypes and self.TypeConflict(OpTypes[operand.Reg], typeDesc):
                        print(f"Warning: Type mismatch for {operand.Reg} in {Inst}: {OpTypes[operand.Reg]} vs {typeDesc}")
                        self.Conflicts[Inst] = (operand, OpTypes[operand.Reg], typeDesc)
                        return

                self.SetOptype(OpTypes, operand, typeDesc)

        # Find propagate type
        propType = "NOTYPE"
        for operand, typeDesc in def_pairs + use_pairs:
            if typeDesc == "PROP":
                existing = self.GetOptype(OpTypes, operand)
                if existing != "NOTYPE":
                    if not Inst.IsPhi(): # Propagate conflicts to somewhere else, not in PHI
                        if propType != "NOTYPE" and self.TypeConflict(existing, propType):
                            print(f"Warning: Propagation type mismatch for {operand} in {Inst}: {existing} vs {propType}")
                            self.Conflicts[Inst] = (operand, existing, propType)
                            return

                    propType = existing

        # Propagate types
        for operand, typeDesc in def_pairs + use_pairs:
            if typeDesc == "PROP" and propType != "NOTYPE":
                self.SetOptype(OpTypes, operand, propType)
            elif typeDesc == "PROP_PTR":
                if propType in ["NOTYPE", "ANY"]:
                    self.SetOptype(OpTypes, operand, "PTR")
                else:
                    self.SetOptype(OpTypes, operand, propType + "_PTR")
