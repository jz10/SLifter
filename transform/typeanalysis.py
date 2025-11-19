from transform.transform import SaSSTransform
from sir.operand import Operand
from collections import deque
from sir.instruction import Instruction
from transform.defuse_analysis import DefUseAnalysis

        
BOOL = {"Bool"}
HALF2 = {"Half2"}
FLOAT16 = {"Float16"}
FLOAT32 = {"Float32"}
FLOAT64 = {"Float64"}
INT32 = {"Int32"}
INT64 = {"Int64"}
NUM1 = BOOL
NUM32 = INT32 | FLOAT32 | HALF2
NUM64 = INT64 | FLOAT64
TOP = NUM32 | NUM1 | NUM64
BOTTOM = set()

class TypeAnalysis(SaSSTransform):
        
    def __init__(self):
        super().__init__()
        
        self.instructionTypeTable = {
            "S2R": [[INT32], [INT32, INT32]],
            "IMAD": [[INT32], [INT32, INT32, INT32]],
            "IADD3": [[INT32], [INT32, INT32, INT32]],
            "XMAD": [[INT32], [INT32, INT32, INT32]],
            "IADD32I": [[INT32], [INT32, INT32]],
            "IADD": [[INT32], [INT32, INT32]],
            "ISETP": [[BOOL, BOOL], [INT32, INT32, BOOL, BOOL]],
            "LEA": [[INT32, BOOL], [INT32, INT32, INT32]],
            "LOP3": [[INT32, INT32], [INT32, INT32, INT32, INT32, INT32, BOOL]],
            "LDG": [[NUM32], [INT64]],
            "LD": [[NUM32], [INT64]],
            "SULD": [[NUM32], [INT64]],
            "STG": [[], [INT64, NUM32]],
            "ST": [[], [INT64, NUM32]],
            "SUST": [[], [INT64, NUM32]],
            "F2I": [[INT32], [FLOAT32]],
            "I2F": [[FLOAT32], [INT32]],
            "SEL": [[INT32], [INT32, INT32, BOOL]],
            "NOP": [[], []],
            "BRA": [[], [TOP, INT32]],
            "EXIT": [[], []],
            "RET": [[], []],
            "SYNC": [[], []],
            "BAR": [[], [INT32]],
            "SSY": [[], []],
            "SHF": [[INT32], [INT32, INT32, INT32]],
            "DEPBAR": [[], []],
            "LOP32I": [[INT32], [INT32, INT32]],
            "ISCADD": [[INT32], [INT32, INT32]],
            "MOV32I": [[INT32], [INT32]],
            "IABS": [[INT32], [INT32]],
            "ULDC": [[NUM32], [INT32]],
            "MATCH": [[INT32], [INT32]],
            "BREV": [[INT32], [INT32]],
            "FLO": [[INT32], [INT32]],
            "POPC": [[INT32], [INT32]],
            "RED": [[], [INT64, INT32]],
            "IMNMX": [[INT32], [INT32, INT32, BOOL]],
            "PRMT": [[INT32], [NUM32, NUM32, NUM32]],
            "HMMA": [[FLOAT32] * 4, [FLOAT32] * 8],
            "MOV": [[NUM32], [NUM32]],
            "SHL": [[INT32], [INT32, INT32]],
            
            # Predicate instructions
            "PLOP3": [[BOOL, BOOL], [BOOL, BOOL, BOOL, INT32, BOOL]],
            
            # Shared memory
            "LDS": [[NUM32], [INT32]],
            "STS": [[], [INT32, NUM32]],
            
            # Warp-level primitives
            "SHFL": [[BOOL, NUM32], [NUM32, INT32, INT32]],
            "VOTE": [[INT32], [BOOL, BOOL]],
            "VOTEU": [[INT32], [BOOL, BOOL]],
            
            "MOV64": [[NUM64], [NUM64]],
            "PHI64": [[NUM64], [NUM64, NUM64, NUM64, NUM64, NUM64, NUM64]],
            "PHI": [[NUM32], [NUM32, NUM32, NUM32, NUM32, NUM32, NUM32]],
            
            # Float instruction types
            "FADD": [[FLOAT32], [FLOAT32, FLOAT32]],
            "FFMA": [[FLOAT32], [FLOAT32, FLOAT32, FLOAT32]],
            "FMUL": [[FLOAT32], [FLOAT32, FLOAT32]],
            "FSETP": [[BOOL, BOOL], [FLOAT32, FLOAT32, BOOL]],
            "FSEL": [[FLOAT32], [FLOAT32, FLOAT32, BOOL]],
            "MUFU": [[FLOAT32], [FLOAT32]],
            "FCHK": [[BOOL], [FLOAT32, INT32]],
            "FMNMX": [[FLOAT32], [FLOAT32, FLOAT32, FLOAT32, BOOL]],
            
            # Double instruction types
            "DADD": [[FLOAT64], [FLOAT64, FLOAT64]],
            "DMUL": [[FLOAT64], [FLOAT64, FLOAT64]],
            "DFMA": [[FLOAT64], [FLOAT64, FLOAT64, FLOAT64]],
            "DSETP": [[BOOL, BOOL], [FLOAT64, FLOAT64, BOOL]],

            # Uniform variants
            "USHF": [[INT32], [INT32, INT32, INT32]],
            "ULEA": [[INT32], [INT32, INT32, INT32, INT32]],
            "ULOP3": [[INT32], [INT32, INT32, INT32, INT32, INT32, BOOL]],
            "UIADD3": [[INT32], [INT32, INT32, INT32]],
            "UMOV": [[INT32], [INT32]],
            "UISETP": [[BOOL, BOOL], [INT32, INT32, BOOL]],
            "USEL": [[INT32], [INT32, INT32, BOOL]],

            # Dummy instruction types
            "PACK64": [[NUM64], [NUM32, NUM32]],
            "UNPACK64": [[NUM32, NUM32], [NUM64]],
            "CAST64": [[NUM64], [NUM32]],
            "IADD64": [[INT64], [INT64, INT64]],
            "IMAD64": [[INT64], [INT32, INT32, INT64]],
            "SHL64": [[INT64], [INT64, INT64]],
            "IADD32I64": [[INT64], [INT64, INT32]],
            "BITCAST": [[TOP], [TOP]],
            "PBRA": [[], [BOOL, INT32, INT32]],
            "LEA64": [[INT64], [INT64, INT64, INT64]],
            "SETZERO": [[TOP], []],
            "ULDC64": [[NUM64], [INT32]],
            "LDG64": [[NUM64], [INT64]],
            "SHR64": [[INT64], [INT64, INT64]],
            "ISETP64":  [[BOOL, BOOL], [INT64, INT64, BOOL]],
            "IADD364": [[INT64], [INT64, INT64, INT64]],
        }

        self.modifierOverrideTable = {
            "MATCH": {
                "U32": [[TOP], [INT32]],
                "U64": [[TOP], [INT64]],
            },
            "IMNMX": {
                "U32": [[INT32], [INT32, INT32]],
                "U64": [[INT64], [INT64, INT64]],
            },
            "HMMA": {
                "F32": [[FLOAT32] * 4, [FLOAT32] * 8],
                "F16": [[FLOAT16] * 4, [FLOAT16] * 8],
            },
            "IMAD": {
                "WIDE": [[INT64], [INT32, INT32, INT64]],
            },
            "STG": {
                "64": [[], [INT64, NUM64]],
            },
        }
        
        self.propagateTable = {
            "MOV": [["A"], ["A"]],
            "AND": [["A"], ["A", "A"]],
            "OR": [["A"], ["A", "A"]],
            "XOR": [["A"], ["A", "A"]],
            "NOT": [["A"], ["A", "A"]],
            "SHL": [["A"], ["A", "A"]],
            "SHR": [["A"], ["A", "A"]],
            "MOVM": [["A"], ["A"]],
            
            "PHI": [["A"], ["A", "A", "A", "A", "A", "A"]],
            "PACK64": [["B"], ["A", "A"]],
            "UNPACK64": [["A", "A"], ["B"]],
            "PHI64": [["A"], ["A", "A", "A", "A", "A", "A"]],
            "MOV64": [["A"], ["A"]],
        }


    def apply(self, module):
        super().apply(module)
        self.defuse = DefUseAnalysis()
        print("=== Start of TypeAnalysis ===")
        for func in module.functions:
            print(f"Processing function: {func.name}")
            self.ProcessFunc(func)
        print("=== End of TypeAnalysis ===")


    def ProcessFunc(self, function):
        WorkList = self.TraverseCFG(function)

        OpTypes = {}
        self.DefUse = DefUseAnalysis()
        
        # Static type resolution
        for BB in WorkList:
            for Inst in BB.instructions:
                self.ResolveTypes(Inst, OpTypes)

        Changed = True
        Iteration = 0
        
        while Changed:

            for BB in WorkList:
                for Inst in BB.instructions:
                    print(str(Inst)+" => ", end="")
                    for Operand in Inst.operands:
                        if Operand in OpTypes:
                            TypeDesc = str(OpTypes[Operand])
                        elif Operand.Reg in OpTypes:
                            TypeDesc = str(OpTypes[Operand.Reg])
                        else:
                            TypeDesc = "NOTYPE"
                        print(TypeDesc+", ",end="")
                    print("")

            print("-----next iteration-----")

            Changed = False
            self.Conflicts = {}
            self.PhiConflicts = {}
            
            for BB in WorkList:
                Changed |= self.ProcessBB(BB, OpTypes)

            # If there is any conflict, insert bitcast instructions
            # After that, re-run defuse, resolve types again
            if len(self.Conflicts) > 0 or len(self.PhiConflicts) > 0:
                for BB in WorkList:
                    NewInstructions = []
                    for Inst in BB.instructions:
                        if Inst in self.Conflicts:
                            self.InsertBitcastBefore(Inst, BB, NewInstructions, OpTypes)
                        NewInstructions.append(Inst)
                        if Inst in self.PhiConflicts:
                            self.InsertBitcastAfter(Inst, BB, NewInstructions, OpTypes)
                    BB.instructions = NewInstructions
                    
                self.defuse.BuildDefUse(function.blocks)
                
                OpTypes = {}
                for BB in WorkList:
                    for Inst in BB.instructions:
                        self.ResolveTypes(Inst, OpTypes)
                
                

            Iteration += 1
            if Iteration > 3:
                print("Warning: TypeAnalysis exceeds 3 iterations, stopping")
                break

        # Apply types to instructions
        for BB in WorkList:
            for Inst in BB.instructions:
                for op in Inst.operands:
                    op.TypeDesc = self.GetTypeDesc(op, OpTypes)

        for BB in WorkList:
            for Inst in BB.instructions:
                print(str(Inst)+" => ", end="")
                for op in Inst.operands:
                    print(op.TypeDesc+", ",end="")
                print("")

        # # Statistics
        # # Build AllRegs from the function CFG, then classify:
        # # - If reg in ConflictedOriginalRegs -> Conflicted
        # # - Else if reg in BitcastRegs -> skip (synthetic)
        # # - Else if reg has a type in OpTypes -> count by that type
        # # - Else -> Unresolved
        # AllRegs = set()
        # for BB in WorkList:
        #     for Inst in BB.instructions:
        #         for op in Inst.operands:
        #             if op.IsWritableReg:
        #                 AllRegs.add(op.Reg)

        # TypeCount = {}
        # for reg in AllRegs:
        #     # Skip synthetic bitcast temps entirely
        #     if reg in self.BitcastRegs:
        #         continue

        #     if reg in self.ConflictedOriginalRegs:
        #         TypeCount["Conflicted"] = TypeCount.get("Conflicted", 0) + 1
        #         continue

        #     if reg in OpTypes and not isinstance(reg, Operand):
        #         tdesc = OpTypes[reg]
        #         TypeCount[tdesc] = TypeCount.get(tdesc, 0) + 1
        #     else:
        #         print(f"Warning: Unresolved type for register {reg}")
        #         TypeCount["Unresolved"] = TypeCount.get("Unresolved", 0) + 1

        # print("Type analysis statistics")
        # for type_desc, count in TypeCount.items():
        #     print(f"Type counts {type_desc}: {count} registers")

        print("done")
        print("=== End of TypeAnalysis ===")
        
    def InsertBitcastBefore(self, Inst, BB, NewInstructions, OpTypes):
        op, OldType, NewType = self.Conflicts[Inst]
        print(f"Warning: Inserting BITCAST to resolve type conflict for {op} before {Inst}: {OldType} vs {NewType}")
        # Insert bitcast before Inst
        SrcReg = Operand.fromReg(op.Reg, op.Reg)
        NewRegName = f"{SrcReg.Reg}_bitcast"
        DestReg = Operand.fromReg(NewRegName, NewRegName)

        BitcastInst = Instruction(
            id=f"{Inst.id}_type_resolve", 
            opcodes=["BITCAST"],
            operands=[DestReg, SrcReg],
            parentBB=BB
        )
        NewInstructions.append(BitcastInst)

        # # Book-keeping for statistics
        # self.ConflictedOriginalRegs.add(orig_reg)
        # self.BitcastRegs.add(NewRegName)

        OpTypes[DestReg.Reg] = OldType
        OpTypes[SrcReg.Reg] = NewType
        op.SetReg(DestReg.Reg)
        
    def InsertBitcastAfter(self, Inst, BB, NewInstructions, OpTypes):
        op, OldType, NewType, PhiDefOp = self.PhiConflicts[Inst]
        print(f"Warning: Inserting BITCAST to resolve type conflict for {op} after {Inst}: {OldType} vs {NewType}")
        # Insert bitcast before Inst
        SrcReg = Operand.fromReg(op.Reg, op.Reg)
        NewRegName = f"{SrcReg.Reg}_bitcast"
        DestReg = Operand.fromReg(NewRegName, NewRegName)

        BitcastInst = Instruction(
            id=f"{Inst.id}_type_resolve", 
            opcodes=["BITCAST"],
            operands=[DestReg, SrcReg],
            parentBB=BB
        )
        NewInstructions.append(BitcastInst)

        OpTypes[SrcReg.Reg] = OldType
        OpTypes[DestReg.Reg] = NewType
        
        PhiDefOp.SetReg(NewRegName)
        
    def GetTypeDesc(self, Operand, OpTypes):
        if Operand in OpTypes:
            types = OpTypes[Operand]
        elif Operand.Reg in OpTypes:
            types = OpTypes[Operand.Reg]
        else:
            raise ValueError(f"Type not found for operand {Operand} / {Operand.Reg}")
        
        if len(types) > 1:
            print(f"Warning: Multiple possible types for {Operand} / {Operand.Reg}: {types}")
            
        if len(INT32 & types) > 0:
            return list(INT32)[0]
        elif len(INT64 & types) > 0:
            return list(INT64)[0]
        else:
            return list(types)[0]
        
    def ResolveTypes(self, Inst, OpTypes):
        
        # Static type table
        opcode = Inst.opcodes[0]
        typeTable = self.instructionTypeTable.get(opcode, None)
        if not typeTable:
            print(f"Warning: Unknown opcode {opcode} in StaticResolve")
            
            # Default to TOP
            for op in Inst.operands:
                self.SetOpType(OpTypes, op, TOP, Inst)
            return
        
        optypeMap = {}
    
        for i, defOp in enumerate(Inst.GetDefs()):
            if typeTable[0][i] != "PROP":
                optypeMap[defOp] = typeTable[0][i]
            else:
                optypeMap[defOp] = TOP
                
        for i, useOp in enumerate(Inst.GetUses()):
            if typeTable[1][i] != "PROP":
                optypeMap[useOp] = typeTable[1][i]
            else:
                optypeMap[useOp] = TOP
                
        # Modifier table
        modifierTable = self.modifierOverrideTable.get(opcode, None)
        if modifierTable:
            for mod in modifierTable:
                if mod not in Inst.opcodes[1:]:
                    continue
                def_overrides, use_overrides = modifierTable[mod]
                
                for i, defOp in enumerate(Inst.GetDefs()):
                    if i < len(def_overrides):
                        optypeMap[defOp] = def_overrides[i]
                        
                for i, useOp in enumerate(Inst.GetUses()):
                    if i < len(use_overrides):
                        optypeMap[useOp] = use_overrides[i]
                        
        # Apply types
        for operand, typeDesc in optypeMap.items():
            if operand.IsPredicateReg or operand.IsPT:
                self.SetOpType(OpTypes, operand, BOOL, Inst)
            else:
                self.SetOpType(OpTypes, operand, typeDesc, Inst)
                    
        # Special case for pred?
        # if operand.IsPredicateReg or operand.IsPT:
        #     typeDesc = BOOL
            

    def GetOptype(self, OpTypes, Operand):
        if Operand in OpTypes:
            return OpTypes[Operand]
        elif Operand.Reg in OpTypes:
            return OpTypes[Operand.Reg]
        else:
            return TOP

    def SetOpType(self, OpTypes, Operand, Type, Inst):
        
        previousType = OpTypes.get(Operand.Reg, TOP)
        
        # Record type conflict
        if len(previousType & Type) == 0:
            print(f"Warning: Type mismatch for {Operand.Reg} in {Inst}: prevType: {previousType} vs optype: {Type}")
            
            if Inst.opcodes[0] != "PHI":
                self.Conflicts[Inst] = (Operand, Type, previousType)
            else:
                # Cannot insert bitcast before phi,
                # need to search and insert after non-phi def instructions
                self.AddPhiConflicts(OpTypes, Operand, Type, previousType, Inst)
            return
            
        if Operand.IsWritableReg:
            OpTypes[Operand.Reg] = previousType & Type
        else:
            # Store operand itself as key for non-register values
            # E.g. 0 in IADD vs 0 in FADD have different types
            OpTypes[Operand] = previousType & Type
            
    def AddPhiConflicts(self, OpTypes, Operand, NewType, OldType, Inst):
        queue = deque()
        prevQueue = deque()
        visited = set()
        
        queue.extend(list(Inst.ReachingDefsSet[Operand]))
        prevQueue.append(Operand)
        
        while queue:
            currInst, currDefOp = queue.popleft()
            phiUseOp = prevQueue.popleft()
            
            if currInst in visited:
                continue
            visited.add(currInst)
            
            if currInst.opcodes[0] != "PHI":
                if len(self.GetOptype(OpTypes, currDefOp) & NewType) == 0:
                    self.PhiConflicts[currInst] = (currDefOp, self.GetOptype(OpTypes, currDefOp), NewType, phiUseOp)
                continue
            
            # Give PHI another chance to repropagate after the bitcasts are inserted
            OpTypes[currDefOp.Reg] = TOP 
            
            for useOp, defInstPair in currInst.ReachingDefsSet.items():
                queue.extend(list(defInstPair))
                prevQueue.append(useOp)

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

    def ProcessBB(self, BB, OpTypes):
        OldOpTypes = OpTypes.copy()

        for Inst in BB.instructions:
            self.PropagateTypes(Inst, OpTypes)
        
        return (OldOpTypes != OpTypes)

    def PropagateTypes(self, Inst, OpTypes):
        opcode = Inst.opcodes[0]
        
        if opcode not in self.propagateTable:
            return
        
        propTable = self.propagateTable[opcode]
        
        propOps = {}
        for i, defOp in enumerate(Inst.GetDefs()):
            propOps.setdefault(propTable[0][i], []).append(defOp)
                
        for i, useOp in enumerate(Inst.GetUses()):
            propOps.setdefault(propTable[1][i], []).append(useOp)

        for propKey in propOps:
            propOpsList = propOps[propKey]

            propType = TOP
            for propOp in propOpsList:
                opType = self.GetOptype(OpTypes, propOp)
                
                if len(propType & opType) != 0:
                    propType = propType & opType
                
            for propOp in propOpsList:
                self.SetOpType(OpTypes, propOp, propType, Inst)