from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand
from collections import deque
from bisect import bisect_right

from transform.opaggregate_patterns import PatternTable

class OperAggregate(SaSSTransform):
    def apply(self, module):
        print("=== Start of Operator Aggregation Transformation ===")

        totalInsert = 0
        totalRemove = 0

        for func in module.functions:

            self.InsertInsts = {}
            self.RemoveInsts = set()
            self.HandledInsts = set()
            self.KnownRegPairs = {}
            self.ProcessedPatterns = []

            SeedPatterns = self.FindSeedPatterns(func)

            self.TrackPatterns(SeedPatterns)
            
            self.FixRegisterDependencies()

            self.ApplyChanges(func)

            print(f"Function {func.name}:")
            print(f"OperAggregate Insert Instructions: {len(self.InsertInsts)}")
            print(f"OperAggregate Remove Instructions: {len(self.RemoveInsts)}")

            totalInsert += len(self.InsertInsts)
            totalRemove += len(self.RemoveInsts)

        print(f"Total OperAggregate Insert Instructions: {totalInsert}")
        print(f"Total OperAggregate Remove Instructions: {totalRemove}")
        
        print("=== End of Operator Aggregation Transformation ===")
        
    # def GetLDG64Instructions(self, func):
    #     LDG64Insts = []
    #     for bb in func.blocks:
    #         for inst in bb.instructions:
    #             if len(inst.opcodes) > 0 and inst.opcodes[0] == "LDG" and "64" in inst.opcodes:
    #                 LDG64Insts.append(inst)

    #     return LDG64Insts
            
    # def GetPack64Instructions(self, func):
    #     Pack64Insts = []
    #     for bb in func.blocks:
    #         for inst in bb.instructions:
    #             if len(inst.opcodes) > 0 and inst.opcodes[0] == "PACK64":
    #                 Pack64Insts.append(inst)

    #     return Pack64Insts
    
    def IterateOpcodes(self, OpcodeInstDict, OpcodeSeq, OrderIndex):
        """
        ai.
        Yield instruction tuples (i0, i1, ..., in) where i_k has opcode OpcodeSeq[k],
        and their program order is strictly increasing within the basic block.
        """
        InstLists = []
        for op in OpcodeSeq:
            lst = OpcodeInstDict.get(op)
            if not lst:
                return
            InstLists.append(lst)

        def dfs(k, prev_idx, acc):
            if k == len(InstLists):
                yield tuple(acc)
                return
            for inst in InstLists[k]:
                idx = OrderIndex.get(inst)
                if idx is None:
                    continue
                if prev_idx is None or idx > prev_idx:
                    acc.append(inst)
                    yield from dfs(k + 1, idx, acc)
                    acc.pop()

        yield from dfs(0, None, [])

    
    def FindSeedPatterns(self, func):
        SeedPatterns = []
        
        OpcodePatterns = []
        
        for patternKey in PatternTable.keys():
            # phi and mov not matched because they can be trivially paired
            if not patternKey or patternKey[0] == "PHI" or patternKey[0] == "MOV":
                continue
            # # ldg not matched because ldg.64 can be for float64 or int64
            # # for now, just avoid this complexity
            # if patternKey[0] == "LDG":
            #     continue
            OpcodePatterns.append(patternKey)
    
        for bb in func.blocks:
            # Cache instruction by opcode for quick lookup
            OpcodeInstDict = {}
            OrderIndex = {}
            for idx, inst in enumerate(bb.instructions):
                if len(inst.opcodes) == 0:
                    continue
                OrderIndex[inst] = idx
                OpcodeInstDict.setdefault(inst.opcodes[0], []).append(inst)
                
            print(f"Processing basic block...")
            
            # Find seed instruction pairs
            for OpcodeSeq in OpcodePatterns:
                # quick skip if any opcode is absent in this block
                if any(op not in OpcodeInstDict for op in OpcodeSeq):
                    continue

                for InstTuple in self.IterateOpcodes(OpcodeInstDict, OpcodeSeq, OrderIndex):
                    # print(f"Trying instruction tuple: ({', '.join(f'<{inst}>' for inst in InstTuple)})")
                    
                    status, _, _ = self.Match(list(InstTuple), CheckOnly=True)
                    if status:
                        SeedPatterns.append(list(InstTuple))
                        print(f"Found seed instruction tuple: ({', '.join(f'<{inst}>' for inst in InstTuple)})")

        return SeedPatterns
    
    

    def TrackPatterns(self, SeedPatterns):
        # reg3:reg2 -> (reg2,reg3) dictionary
        # all regs are SSA so no conflict
        Queue = deque()

        for SeedPattern in SeedPatterns:
            Queue.append(SeedPattern)

        while len(Queue) > 0:
            PatternInsts = Queue.popleft()
            
            print(f"({', '.join(f'<{inst}>' for inst in PatternInsts)})")
                    
            if len(PatternInsts) == 0:
                continue

            # Identify pattern, get replace instruction, next reg pairs candidates
            Status, MergeInsts, RegPairs = self.Match(PatternInsts)
                
                
            # Skip if any inst in patternInsts is already handled
            if (any(inst in self.HandledInsts for inst in PatternInsts)):
                # If not all of them being handled, print a warning
                if not all(inst in self.HandledInsts for inst in PatternInsts):
                    print(f"\tWarning: not all instructions handled for pattern {PatternInsts}")
                print("=> Already handled, skipping\n")
                continue
            
            # Check match status
            if not Status:
                print(f"\tChain ended for this pair")
                print("=> No pattern match\n")
                continue
                
            # Mark all instructions in pattern as handled
            for inst in PatternInsts:
                self.HandledInsts.add(inst)

            EndOfChain = (len(RegPairs) == 0)
            
            # Convert operands to instructions
            NextPatterns = []
            for RegPair in RegPairs: 
                NextPatterns.extend(self.GetMatchingPairs(RegPair))
                
            
                # For each reg pair, make sure all def instructions are covered
                # Reg1Defs = set()
                # Reg1Defs.add((RegPair[0][0].Parent, RegPair[0][0]))
                    
                # Reg2Defs = set()
                # Reg2Defs.add((RegPair[1][0].Parent, RegPair[1][0]))

                # if not self.AllDefsCovered(MatchingPairs, Reg1Defs, Reg2Defs):
                #     InsertInsts[Inst1].append(self.CreatePack64(Inst1, RegPair[0], RegPair[1]))
                #     EndOfChain = True
                #     print(f"\tPack64 inserted for operand ({RegPair[0][1]}, {RegPair[1][1]})")
                
            # Add to processed patterns
            self.ProcessedPatterns.append([PatternInsts, MergeInsts])

            if not EndOfChain:
                Queue.extend(NextPatterns)
                for NextPattern in NextPatterns:
                    print(f"\tNext pair(use->def): (", end="")
                    for NextPatternInst in NextPattern:
                        print(f"{NextPatternInst},", end=" ")
                    print(")")
            else:
                print(f"\tChain ended for this pair")

            print(f"=> {MergeInsts}\n")

    # CheckOnly: won't generate new instructions
    def Match(self, PatternInsts, CheckOnly=False, ResolveDefine=True):

        def IsVariable(token):
            prefixes = ("reg", "pred", "const", "op", "imm")
            return any(token.startswith(prefix) for prefix in prefixes)

        def VariableTypeMatches(var_name, value):
            prefix_end = 0
            for idx, ch in enumerate(var_name):
                if ch.isalpha():
                    prefix_end = idx + 1
                else:
                    break
            prefix = var_name[:prefix_end]

            if prefix == "reg":
                return isinstance(value, Operand) and value.IsReg
            if prefix == "pred":
                return isinstance(value, Operand) and value.IsPredicateReg
            if prefix == "const":
                return isinstance(value, Operand) and value.IsConstMem
            if prefix == "imm":
                return isinstance(value, Operand) and value.IsImmediate
            if prefix == "op":
                return True
            return True
        
        def MatchOpcodes(PatternOpcodes, InstOpcodes, Variables):
            if len(PatternOpcodes) > len(InstOpcodes):
                return False
            
            for patternOpcode, opcode in zip(PatternOpcodes, InstOpcodes):
                if "op" in patternOpcode:
                    if patternOpcode in Variables and Variables[patternOpcode] != opcode:
                        return False
                    else:
                        Variables[patternOpcode] = opcode
                else:
                    if patternOpcode != opcode:
                        return False
            return True
        
        def MatchOperands(PatternOperands, InstOperands, Variables):
            if len(PatternOperands) == 1 and "[*]" in PatternOperands[0]:
                PatternOperands = [PatternOperands[0].replace("[*]", str(i)) for i in range(len(InstOperands))]
                pack_length = len(InstOperands)
                if "pack_length" in Variables:
                    # This prevents two phi having different number of operands to be matched.
                    if Variables["pack_length"] != pack_length:
                        return False
                else:
                    Variables["pack_length"] = pack_length

            if len(PatternOperands) > len(InstOperands):
                return False
            
            for patternOperand, operand in zip(PatternOperands, InstOperands):
                if IsVariable(patternOperand):
                    if not VariableTypeMatches(patternOperand, operand):
                        return False
                    if patternOperand in Variables and Variables[patternOperand].Name != operand.Name:
                        return False
                    else:
                        Variables[patternOperand] = operand
                else:
                    if patternOperand != operand.Name:
                        return False
            return True
        
        def GenArray(PatternArray, Variables, Operand=Operand.Parse):
            Array = []
            for item in PatternArray:
                if IsVariable(item):
                    if item in Variables:
                        Array.append(Variables[item])
                    else:
                        Array.append(Operand(item))
                else:
                    Array.append(Operand.Parse(item))
            return Array
        
        def ResolveDefined(PatternDefined, Variables):
            def FindOperandInDefInst(regOpInUseInst):
                if len(regOpInUseInst.DefiningInsts) > 1:
                    raise Exception("Multiple defining instructions found for operand")
                
                for defInst in regOpInUseInst.DefiningInsts:
                    for defOp in defInst.GetDefs():
                        if defOp.Reg == regOpInUseInst.Reg:
                            return defOp
                return None
            
            if len(PatternDefined) == 0:
                return True
            
            for patternDef in PatternDefined:
                regs = patternDef.split(":")
                if len(regs) != 2:
                    raise Exception("Defined reg pair must have two regs")
                
                reg2, reg1 = regs
                
                # Find the corresponding operands in defining instructions
                if reg1 in Variables:
                    reg1DefOp = FindOperandInDefInst(Variables[reg1])
                if reg2 in Variables:
                    reg2DefOp = FindOperandInDefInst(Variables[reg2])
                    
                
                if reg1 in Variables and reg1DefOp in self.KnownRegPairs:
                    Variables[reg2] = self.KnownRegPairs[reg1DefOp][0]
                elif reg2 in Variables and reg2DefOp in self.KnownRegPairs:
                    Variables[reg1] = self.KnownRegPairs[reg2DefOp][1]
                else:
                    return False
                
            return True
        
        def UpdateDefToKnownRegPairs(PatternOperands, Variables):
            if len(PatternOperands) == 1 and "[*]" in PatternOperands[0]:
                OperandsLength = Variables["pack_length"]
                PatternOperands = [PatternOperands[0].replace("[*]", str(i)) for i in range(OperandsLength)]

            for PatternOperand in PatternOperands:
                p = PatternOperand.split(":")
                for i in range(len(p)):
                    if IsVariable(p[i]):
                        var = Variables[p[i]]
                    else:
                        var = p[i]
                    p[i] = var

                if len(p) == 2:
                    print(f"\tKnown reg pair: ({p[0]}, {p[1]})")
                    self.KnownRegPairs[p[0]] = (p[0], p[1])
                    self.KnownRegPairs[p[1]] = (p[0], p[1])
                        
        def GenOpcodes(PatternArray, Variables):
            Opcodes = []
            for p in PatternArray:
                if IsVariable(p):
                    Opcodes.append(Variables[p])
                else:
                    Opcodes.append(p)
            return Opcodes
        
        def GenNextArray(PatternArray, Variables):
            if len(PatternArray) == 1 and "[*]" in PatternArray[0][0]:
                OperandsLength = Variables["pack_length"]
                PatternArray = [(PatternArray[0][0].replace("[*]", str(i)), PatternArray[0][1].replace("[*]", str(i))) for i in range(OperandsLength)]

            RegPairs = []
            for patternPair in PatternArray:
                if len(patternPair) != 2:
                    raise Exception("Next reg pair must have two regs")
                
                reg1, reg2 = patternPair
                if IsVariable(reg1):
                    reg1Op = Variables[reg1]
                else:
                    reg1Op = Operand.Parse(reg1)
                    
                if IsVariable(reg2):
                    reg2Op = Variables[reg2]
                else:
                    reg2Op = Operand.Parse(reg2)
                
                RegPairs.append((reg1Op, reg2Op))
                
            return RegPairs

        def GenOperands(PatternOperands, Variables):
            if len(PatternOperands) == 1 and "[*]" in PatternOperands[0]:
                OperandsLength = Variables["pack_length"]
                PatternOperands = [PatternOperands[0].replace("[*]", str(i)) for i in range(OperandsLength)]

            Array = []
            for PatternOperand in PatternOperands:
                p = PatternOperand.split(":")
                for i in range(len(p)):
                    if IsVariable(p[i]):
                        name = Variables[p[i]].Name
                    else:
                        name = p[i]
                    p[i] = name
                Array.append(Operand.Parse(":".join(p)))

            return Array
        
        def HandleRZ(UseOperands, ParentInst, OutInsts):
            for useOp in UseOperands:
                if "RZ" in useOp.Name and ":" in useOp.Name:
                    regs = useOp.Name.split(":")

                    operands = [useOp.Clone(), Operand.Parse(regs[0]), Operand.Parse(regs[1])]

                    pack64Inst = Instruction(
                        id=f"{ParentInst.id}_pack64_{useOp.Name}",
                        opcodes=["PACK64"],
                        operands=operands,
                        parentBB=ParentInst.parent
                    )
                    OutInsts.append(pack64Inst)
                    print(f"\tPack64 inserted for operand {useOp}")

        # Match pattern
        opcodesPattern = tuple([inst.opcodes[0] for inst in PatternInsts])
        if opcodesPattern not in PatternTable:
            return False, None, None
        currentPatterns = PatternTable[opcodesPattern]
        
        for pattern in currentPatterns:
            inInstsPattern = pattern["in"]
            outInstsPattern = pattern["out"]
            
            variables = {"RZ": Operand.Parse("RZ"), "PT": Operand.Parse("PT")}
            matched = True
            
            if len(inInstsPattern) != len(PatternInsts):
                continue

            for idx, inInstPattern in enumerate(inInstsPattern):
                inInst = PatternInsts[idx]

                # Match opcodes
                pattenInInstOpcodes = inInstPattern["opcodes"]
                if not MatchOpcodes(pattenInInstOpcodes, inInst.opcodes, variables):
                    matched = False
                    break

                # Match def operands
                patternInInstDefs = inInstPattern["def"]
                if not MatchOperands(patternInInstDefs, inInst.GetDefs(), variables):
                    matched = False
                    break
                
                # Match use operands
                patternInInstUses = inInstPattern["use"]
                if not MatchOperands(patternInInstUses, inInst.GetUses(), variables):
                    matched = False
                    break
                
                # Resolve defined reg pairs 
                if ResolveDefine:
                    patternDefined = pattern.get("defined", [])
                    if not ResolveDefined(patternDefined, variables):
                        matched = False
                        break

            if not matched:
                continue
            

            # Skip early if CheckOnly
            if CheckOnly:
                return True, None, None

            # Generate output instructions
            outInsts = []
            for outInstPattern in outInstsPattern:
                
                # Add its defining reg pairs to known reg pairs
                UpdateDefToKnownRegPairs(outInstPattern["def"], variables)
                
                regPairs = GenNextArray(pattern.get("next", []), variables)

                id = f"{inInst.id}_x64"
                opcodes = GenOpcodes(outInstPattern["opcodes"], variables)
                defs = GenOperands(outInstPattern["def"], variables)
                uses = GenOperands(outInstPattern["use"], variables)
                operands = defs + uses
                parent = inInst.parent
                
                outInst = Instruction(
                    id=id,
                    opcodes=opcodes,
                    operands=operands,
                    parentBB=parent
                )
                
                # Handle RZ. For example, if an use operand is RZ:R2, 
                # insert PACK64 RZ:R2=RZ,R2 before outInst 
                HandleRZ(uses, outInst, outInsts)
                
                outInsts.append(outInst)

                return True, outInsts, regPairs

        return False, None, None

    def GetMatchingPairs(self, RegPairs):
        Reg1Op = RegPairs[0]
        Reg2Op = RegPairs[1]
        Reg1Insts = Reg1Op.DefiningInsts
        Reg2Insts = Reg2Op.DefiningInsts
        MatchingPairs = []
        
        # Pattern could be all combinations: R1 only, R2 only, R1 and R2
        if len(Reg1Insts) == 0:
            MatchR1 = False
            MatchR2 = True
            MatchR1AndR2 = False
        elif len(Reg2Insts) == 0:
            MatchR2 = False
            MatchR1 = True
            MatchR1AndR2 = False
        elif Reg1Insts == Reg2Insts:
            MatchR1 = True
            MatchR2 = False
            MatchR1AndR2 = False
        else:
            MatchR1 = True
            MatchR2 = True
            MatchR1AndR2 = True
        
        if MatchR1AndR2:
            for Reg1DefInst in Reg1Insts:
                for Reg2DefInst in Reg2Insts:
                    if self.ControlEquivalent(Reg1DefInst, Reg2DefInst):
                        Status, _, _ = self.Match([Reg1DefInst, Reg2DefInst], CheckOnly=True, ResolveDefine=False)
                        if Status:
                            MatchingPairs.append((Reg1DefInst, Reg2DefInst))
        if MatchR2:
            for RegDefInst in Reg2Insts:
                Status, _, _ = self.Match([RegDefInst], CheckOnly=True, ResolveDefine=False)
                if Status:
                    MatchingPairs.append((RegDefInst,))
        if MatchR1:
            for RegDefInst in Reg1Insts:
                Status, _, _ = self.Match([RegDefInst], CheckOnly=True, ResolveDefine=False)
                if Status:
                    MatchingPairs.append((RegDefInst,))
        
        return MatchingPairs
    
    def ControlEquivalent(self, Inst1, Inst2):
        # TODO: Inst1.dominate(Inst2) and Inst2.postdominate(Inst1)
        return True
    
    def AllDefsCovered(self, MatchingPairs, Reg1Defs, Reg2Defs):
        CoveredReg1 = set()
        CoveredReg2 = set()
        for r1Def, r2Def in MatchingPairs:
            CoveredReg1.add(r1Def)
            CoveredReg2.add(r2Def)

        return CoveredReg1 == Reg1Defs and CoveredReg2 == Reg2Defs

    # def CreatePack64(self, InsertBefore, UseOp1, UseOp2):
        
    #     Inst = Instruction(
    #         id=f"{InsertBefore.id}_pack64_restore",
    #         opcodes=["PACK64"],
    #         operands=[dest_op, lo_def, hi_def],
    #         inst_content=f"PACK64 {dest_op.Name}, {lo_def.Name} {hi_def.Name}",
    #         parentBB=InsertBefore.parent
    #     )
    #     newDefOp = Operand()
    #     newDefOp.SetReg(f"pack64_{UseOp1.Name}_{UseOp2.Name}")
    #     newDefOp.Type = "U64"
        
    #     newInst = Instruction()
    #     newInst.opcodes = ["PACK64"]
    #     newInst.operands = [newDefOp, UseOp1, UseOp2]
        
    #     return newInst
    
    def FixRegisterDependencies(self):

        for Pattern in self.ProcessedPatterns:
            InInsts = Pattern[0]
            OutInsts = Pattern[1]

            # go over user instruction of every InInsts to make sure every user is converted
            AllConverted = True
            if len(InInsts) > 1 or (len(InInsts) == 1 and len(InInsts[0].GetDefs()) > 1):
                for Inst in InInsts:
                    for User in Inst.Users.values():
                        for UserInst, UserOp in User:
                            if UserInst not in self.HandledInsts:
                                AllConverted = False

            # Remove pattern instructions
            for Inst in InInsts:
                self.RemoveInsts.add(Inst)
                
            # Insert output instructions after the last pattern instruction    
            self.InsertInsts[InInsts[-1]] = OutInsts
            
            # If not all users converted, insert UNPACK64
            if not AllConverted:
                print(f"\tNot all users converted, inserting UNPACK64 after {InInsts[-1]}")
                self.InsertInsts[InInsts[-1]].extend(self.CreateUnpack64(OutInsts))
            
    def CreateUnpack64(self, OutInsts):
        Unpack64Insts = []
        
        for OutInst in OutInsts:
            for DefOp in OutInst.GetDefs():
                if ":" not in DefOp.Name:
                    continue
                
                regs = DefOp.Name.split(":")
                operands = [Operand.Parse(regs[0]), Operand.Parse(regs[1]), DefOp.Clone()]
                
                unpack64Inst = Instruction(
                    id=f"{OutInst.id}_unpack64_{operands[0].Name}_{operands[1].Name}",
                    opcodes=["UNPACK64"],
                    operands=operands,
                    parentBB=OutInst.parent
                )
                Unpack64Insts.append(unpack64Inst)
                print(f"\tUnpack64 inserted for operand ({operands[0].Name}, {operands[1].Name}) from {DefOp.Name}")

        return Unpack64Insts
            

    def ApplyChanges(self, func):
        # # Remove Pack64Insts and handle register dependencies
        # for Inst in Pack64Insts:
        #     mergeOp = Inst.GetUses()[0]
        #     for users in Inst.Users.values():
        #         for _, UseOp in users:
        #             UseOp.SetReg(mergeOp.Name)

        #     RemoveInsts.add(Inst)

        for bb in func.blocks:

            new_insts = []

            for inst in bb.instructions:

                if inst in self.InsertInsts:
                    insertInst = self.InsertInsts[inst]
                    new_insts.extend(insertInst)

                
                if inst not in self.RemoveInsts:
                    new_insts.append(inst)

            bb.instructions = new_insts
