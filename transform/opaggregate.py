from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand
from collections import deque


PatternTable = {
    ("PACK64","PACK64"): [
        {
            "in": [
                {
                    "opcodes": ["PACK64"],
                    "def": ["reg1"],
                    "use": ["reg2", "reg3"],
                },
                {
                    "opcodes": ["PACK64"],
                    "def": ["reg1"],
                    "use": ["reg2", "reg3"],
                },
            ],
            "out": [
                {
                    "opcodes": ["MOV64"],
                    "def": ["reg1"],
                    "use": ["reg3:reg2"],
                }
            ],
        }
    ],

    ("ISETP", "ISETP"): [
        {
            "in": [
                {
                    "opcodes": ["ISETP", "op1", "U32", "AND"],
                    "def": ["pred1"],
                    "use": ["PT", "reg1", "imm1", "PT"],
                },
                {
                    "opcodes": ["ISETP", "op1", "AND", "EX"],
                    "def": ["pred2"],
                    "use": ["PT", "reg2", "imm2", "PT", "pred1"],
                },
            ],
            "out": [
                {
                    "opcodes": ["ISETP64", "op1", "AND"],
                    "def": ["pred2"],
                    "use": ["reg2:reg1", "imm2:imm1", "PT"],
                }
            ],
        },
        {
            "in": [
                {
                    "opcodes": ["ISETP", "op1", "U32", "AND"],
                    "def": ["pred1"],
                    "use": ["PT", "reg1", "imm1", "PT"],
                },
                {
                    "opcodes": ["ISETP", "op1", "U32", "AND", "EX"],
                    "def": ["pred2"],
                    "use": ["PT", "reg2", "imm2", "PT", "pred1"],
                },
            ],
            "out": [
                {
                    "opcodes": ["ISETP64", "op1", "AND"],
                    "def": ["pred2"],
                    "use": ["reg2:reg1", "imm2:imm1", "PT"],
                }
            ],
        }
    ],
    ("LDG", "LDG"): [
        {
            "in": [
                {
                    "opcodes": ["LDG", "E", "64", "SYS"],
                    "def": ["reg1", "reg2"],
                    "use": ["reg3"],
                },
                {
                    "opcodes": ["LDG", "E", "64", "SYS"],
                    "def": ["reg1", "reg2"],
                    "use": ["reg3"],
                },
            ],
            "out": [
                {
                    "opcodes": ["LDG64", "E", "SYS"],
                    "def": ["reg2:reg1"],
                    "use": ["reg3"],
                }
            ],
        }
    ],
    ("LEA", "LEA"): [
        {
            "in": [
                {
                    "opcodes": ["LEA"],
                    "def": ["reg1", "pred1"],
                    "use": ["reg2", "op1", "imm1"],
                },
                {
                    "opcodes": ["LEA", "HI", "X"],
                    "def": ["reg3"],
                    "use": ["reg2", "op2", "reg4", "imm1", "pred1"],
                },
            ],
            "out": [
                {
                    "opcodes": ["LEA64"],
                    "def": ["reg3:reg1"],
                    "use": ["reg4:reg2", "op2:op1", "imm1"],
                }
            ],
        }
    ],
    ("PHI", "PHI"): [
        {
            "in": [
                {
                    "opcodes": ["PHI"],
                    "def": ["reg1"],
                    "use": "pack_low",
                },
                {
                    "opcodes": ["PHI"],
                    "def": ["reg2"],
                    "use": "pack_high",
                },
            ],
            "out": [
                {
                    "opcodes": ["PHI64"],
                    "def": ["reg2:reg1"],
                    "use": "pack",
                }
            ],
        }
    ],

    ("IMAD","IMAD"): [
        {
            "in": [
                {
                    "opcodes": ["IMAD", "WIDE"],
                    "def": ["reg1", "reg2"],
                    "use": ["reg3", "op1"],
                },
                {
                    "opcodes": ["IMAD", "WIDE"],
                    "def": ["reg1", "reg2"],
                    "use": ["reg3", "op1"],
                }
            ],
            "out": [
                {
                    "opcodes": ["IMAD64"],
                    "def": ["reg2:reg1"],
                    "use": ["reg3", "op1"],
                }
            ],
        },
    ],
    ("IMAD", "ANY"): [
        {
            "in": [
                {
                    "opcodes": ["IMAD", "MOV", "U32"],
                    "def": ["reg1"],
                    "use": ["RZ", "RZ", "reg2"],
                },
                {
                    "opcodes": [],
                    "def": ["reg3"],
                    "use": [],
                    "keep": True,
                },
            ],
            "out": [
                {
                    "opcodes": ["MOV64"],
                    "def": ["reg3:reg1"],
                    "use": ["reg3:reg2"],
                }
            ],
        },
    ],
    ("NONE", "IMAD"): [
        {
            "in": [
                {
                    "opcodes": ["IMAD", "MOV", "U32"],
                    "def": ["reg1"],
                    "use": ["RZ", "RZ", "reg2"],
                },
            ],
            "out": [
                {
                    "opcodes": ["AND64"],
                    "def": ["RZ:reg1"],
                    "use": ["reg3:reg2", "0x0000ffff"],
                },
            ],
        },
    ],
    ("ANY", "SHF"): [
        {
            "in": [
                {
                  "opcodes": [],
                  "def": [],
                  "use": [],
                  "keep": True,
                },
                {
                    "opcodes": ["SHF", "R", "S32", "HI"],
                    "def": ["reg3"],
                    "use": ["RZ", "0x1f", "reg1"],
                }
            ],
            "out": [
                {
                    "opcodes": ["SHR64"],
                    "def": ["reg3:reg1"],
                    "use": ["reg1:reg2", "0x20"],
                },
                # {
                #     "opcodes": ["CAST64"],
                #     "def": ["reg2:reg3"],
                #     "use": ["reg3"],
                # }
            ],
        }
    ],
    ("IADD3", "IADD3"): [
        { # weird pattern observed from wyllie
            "in": [
                {
                    "opcodes": ["IADD3"],
                    "def": ["reg1", "pred1", "pred2"],
                    "use": ["RZ", "reg3", "reg4"],
                },
                {
                    "opcodes": ["IADD3", "X"],
                    "def": ["reg5"],
                    "use": ["reg6", "RZ", "RZ", "pred1", "pred2"],
                },
            ],
            "out": [
                {
                    "opcodes": ["IADD364"],
                    "def": ["reg5:reg1"],
                    "use": ["RZ", "reg6:reg3", "RZ:reg4"],
                }
            ],
        },
        {
            "in": [
                {
                    "opcodes": ["IADD3"],
                    "def": ["reg1", "pred1", "pred2"],
                    "use": ["reg2", "reg3", "reg4"],
                },
                {
                    "opcodes": ["IADD3", "X"],
                    "def": ["reg5"],
                    "use": ["reg6", "reg7", "reg8", "pred1", "pred2"],
                },
            ],
            "out": [
                {
                    "opcodes": ["IADD364"],
                    "def": ["reg5:reg1"],
                    "use": ["reg6:reg2", "reg7:reg3", "reg8:reg4"],
                }
            ],
        },
        {
            "in": [
                {
                    "opcodes": ["IADD3"],
                    "def": ["reg1", "pred1"],
                    "use": ["reg2", "reg3", "reg4"],
                },
                {
                    "opcodes": ["IADD3", "X"],
                    "def": ["reg5"],
                    "use": ["reg6", "reg7", "reg8", "pred1", "!PT"],
                },
            ],
            "out": [
                {
                    "opcodes": ["IADD364"],
                    "def": ["reg5:reg1"],
                    "use": ["reg6:reg2", "reg7:reg3", "reg8:reg4"],
                }
            ],
        },
    ],
}

class OperAggregate(SaSSTransform):
    def apply(self, module):
        print("=== Start of Operator Aggregation Transformation ===")

        totalInsert = 0
        totalRemove = 0

        for func in module.functions:

            # Pack64Insts = self.GetPack64Instructions(func)
            
            # LDG64Insts = self.GetLDG64Instructions(func)
            
            SeedInstPairs = self.FindSeedInstructions(func)

            InsertInsts, RemoveInsts = self.TrackPatterns(SeedInstPairs)

            self.ApplyChanges(func, InsertInsts, RemoveInsts)

            print(f"Function {func.name}:")
            print(f"OperAggregate Insert Instructions: {len(InsertInsts)}")
            print(f"OperAggregate Remove Instructions: {len(RemoveInsts)}")

            totalInsert += len(InsertInsts)
            totalRemove += len(RemoveInsts)

        print(f"Total OperAggregate Insert Instructions: {totalInsert}")
        print(f"Total OperAggregate Remove Instructions: {totalRemove}")
        
        print("=== End of Operator Aggregation Transformation ===")
        
    def GetLDG64Instructions(self, func):
        LDG64Insts = []
        for bb in func.blocks:
            for inst in bb.instructions:
                if len(inst.opcodes) > 0 and inst.opcodes[0] == "LDG" and "64" in inst.opcodes:
                    LDG64Insts.append(inst)

        return LDG64Insts
            
    def GetPack64Instructions(self, func):
        Pack64Insts = []
        for bb in func.blocks:
            for inst in bb.instructions:
                if len(inst.opcodes) > 0 and inst.opcodes[0] == "PACK64":
                    Pack64Insts.append(inst)

        return Pack64Insts
    
    def FindSeedInstructions(self, func):
        SeedInsts = []
        
        OpcodeMatchDict = {}
        
        for patternKey in PatternTable.keys():
            op1, op2 = patternKey
            if op1 != "ANY" and op2 != "ANY" and op1 != "PHI" and op2 != "PHI":
                OpcodeMatchDict.setdefault(op1, set()).add(op2)
        
        for bb in func.blocks:
            # Cache instruction by opcode for quick lookup
            OpcodeInstDict = {}
            for inst in bb.instructions:
                if len(inst.opcodes) == 0:
                    continue
                OpcodeInstDict.setdefault(inst.opcodes[0], []).append(inst)
                
            print(f"Processing basic block...")
            
            # Find seed instruction pairs
            for op1 in OpcodeMatchDict:
                if op1 not in OpcodeInstDict:
                    continue
                for inst1 in OpcodeInstDict[op1]:
                    for op2 in OpcodeMatchDict[op1]:
                        if op2 not in OpcodeInstDict:
                            continue
                        for inst2 in OpcodeInstDict[op2]:
                            status, _, _, _, _ = self.Match(inst1, inst2, {}, CheckOnly=True, MatchAny=False)
                            if status:
                                SeedInsts.append((inst1, inst2))
                                print(f"Found seed instruction pair: (<{inst1}>, <{inst2}>)")
        return SeedInsts

    def TrackPatterns(self, SeedInstPairs):

        InsertInsts = {}
        RemoveInsts = set()
        HandledInsts = set()
        
        # reg3:reg2 -> (reg2,reg3) dictionary
        # all regs are SSA so no conflict
        KnownRegPairs = {} 
        Queue = deque()

        for SeedInst1, SeedInst2 in SeedInstPairs:
            Queue.append((SeedInst1, SeedInst2))

        while len(Queue) > 0:
            (Inst1, Inst2) = Queue.popleft()

            print(f"\n\n(<{Inst1}>, <{Inst2}>)")

            # Identify pattern, get replace instruction, next reg pairs candidates
            Status, MergeInst, RegPairs, KeepIn1, KeepIn2 = self.Match(Inst1, Inst2, KnownRegPairs)
            if not Status:
                raise Exception("Should match here")
            else:
                print(f" => {MergeInst}")
                
            if (not KeepIn1 and Inst1 in HandledInsts) or (not KeepIn2 and Inst2 in HandledInsts):
                print("\tPair already handled, skipping")
                continue

            if not KeepIn1:
                HandledInsts.add(Inst1)
            if not KeepIn2:
                HandledInsts.add(Inst2)            

            EndOfChain = False
            if len(RegPairs) == 0:
                EndOfChain = True
            
            # For each reg pair, make sure all def instructions are covered
            for RegPair in RegPairs: 
                MatchingPairs = self.GetMatchingPairs(RegPair, KnownRegPairs)
                
                # If an input is kept, we do not advance its def-use chain
                if not KeepIn1:
                    Reg1Defs = Inst1.ReachingDefsSet.get(RegPair[0][0])
                else:
                    Reg1Defs = set()
                    Reg1Defs.add((RegPair[0][0].Parent, RegPair[0][0]))
                    
                if not KeepIn2:
                    Reg2Defs = Inst2.ReachingDefsSet.get(RegPair[1][0])
                else:
                    Reg2Defs = set()
                    Reg2Defs.add((RegPair[1][0].Parent, RegPair[1][0]))


                if not self.AllDefsCovered(MatchingPairs, Reg1Defs, Reg2Defs):
                    InsertInsts[Inst1].append(self.CreatePack64(Inst1, RegPair[0], RegPair[1]))
                    EndOfChain = True
                    print(f"\tPack64 inserted for operand ({RegPair[0][1]}, {RegPair[1][1]})")
                    
            # # Heuristic: explore users of the current instruction pair
            # # Algorithm should generally go upward(use->def), but sometimes going downward helps(def->use)
            # # The reason is not all chain of patterns end up in pack64/ldg64.
            # if not KeepIn1 and not KeepIn2:
            #     Inst1Users = Inst1.Users[DefReg1]
            #     Inst2Users = Inst2.Users[DefReg2]
            #     if len(Inst1Users) == len(Inst2Users):
            #         for i in range(len(Inst1Users)):
            #             user1, useOp1 = list(Inst1Users)[i]
            #             user2, useOp2 = list(Inst2Users)[i]
            #             if self.ControlEquivalent(user1, user2):
            #                 status, _, _, _, _ = self.Match(user1, user2, UnresolvedRegPairs, CheckOnly=True)
            #                 if status and user1 not in HandledInsts and user2 not in HandledInsts:
            #                     print(f"\tNext pair(def->use): ({user1}, {user2})")
            #                     Queue.append(((user1, user1.GetDefs()[0]), (user2, user2.GetDefs()[0])))
            #     else:
            #         print(f"\tWarning: users length mismatch between {Inst1} and {Inst2}, skip exploring users")

            if not KeepIn1:
                RemoveInsts.add(Inst1)
            if not KeepIn2:
                RemoveInsts.add(Inst2)
                
            InsertInsts[Inst1] = MergeInst

            if not EndOfChain:
                Queue.extend(MatchingPairs)
                for RegPair in MatchingPairs:
                    print(f"\tNext pair(use->def): ({RegPair[0][0]}, {RegPair[1][0]})")
            else:
                print(f"\tChain ended for this pair")
            
        return InsertInsts, RemoveInsts

    # CheckOnly: won't generate new instructions
    # MatchAny: match ANY opcode patterns
    def Match(self, Inst1, Inst2, KnownRegPairs, CheckOnly=False, MatchAny=True):

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

        def MatchArray(PatternArray, InstArray, Variables):
            if PatternArray == "pack_low":
                PatternArray = [f"reg_pack_low_{i}" for i in range(len(InstArray))]
            elif PatternArray == "pack_high":
                PatternArray = [f"reg_pack_high_{i}" for i in range(len(InstArray))]
            
            if len(PatternArray) > len(InstArray):
                return False

            for p, i in zip(PatternArray, InstArray):
                if IsVariable(p):
                    if not VariableTypeMatches(p, i):
                        return False
                    if p in Variables:
                        if isinstance(i, Operand):
                            if Variables[p].Name != i.Name:
                                return False
                        else:
                            if Variables[p] != i:
                                return False
                    else:
                        Variables[p] = i
                else:
                    if isinstance(i, Operand):
                        if p != i.Name:
                            return False
                    else:
                        if p != i:
                            return False
            return True
        
        def GenArray(PatternArray, Variables, PostFunc=None):
            if PatternArray == "pack":
                PatternArray = []
                for var in Variables:
                    if var.startswith("reg_pack_low_"):
                        PatternArray.append(var)
                PatternArray = sorted(PatternArray)
            
            Array = []
            
            for i, p in enumerate(PatternArray):
                if ":" in p: # reg2:reg1 => use reg1 to represent x64 value
                    p = p.split(":")[1]
                if IsVariable(p):
                    value = Variables[p]
                    if hasattr(value, "Name"):
                        Array.append(value.Name)
                    else:
                        Array.append(value)
                else:
                    Array.append(p)
                    
            for i, p in enumerate(Array):
                if PostFunc is not None:
                    Array[i] = PostFunc(p)
                    
            return Array
        
        # Try to match one or two instructions
        currentPatterns = []
        isOneInstPattern = False
        if MatchAny and ("ANY", Inst2.opcodes[0]) in PatternTable:
            currentPatterns.extend(PatternTable[("ANY", Inst2.opcodes[0])])
            matchInsts = [Inst1, Inst2]
            isOneInstPattern = True
        if MatchAny and (Inst1.opcodes[0], "ANY") in PatternTable:
            currentPatterns.extend(PatternTable[(Inst1.opcodes[0], "ANY")])
            matchInsts = [Inst1, Inst2]
            isOneInstPattern = True
        if (Inst1.opcodes[0], Inst2.opcodes[0]) in PatternTable:
            currentPatterns.extend(PatternTable[(Inst1.opcodes[0], Inst2.opcodes[0])])
            matchInsts = [Inst1, Inst2]
        
        if len(currentPatterns) == 0:
            return False, None, None, None, None
        
        for pattern in currentPatterns:
            inInstsPattern = pattern["in"]
            outInstsPattern = pattern["out"]
            
            variables = {"RZ": Operand.Parse("RZ"), "PT": Operand.Parse("PT")}
            matched = True


            for idx, inInstPattern in enumerate(inInstsPattern):
                inInst = matchInsts[idx]

                # Match opcodes
                inInstOpcodes = inInstPattern["opcodes"]
                if not MatchArray(inInstOpcodes, inInst.opcodes, variables):
                    matched = False
                    break

                # Match defs and uses
                inInstDefs = inInstPattern["def"]
                inInstUses = inInstPattern["use"]
                if not MatchArray(inInstDefs, inInst.GetDefs(), variables):
                    matched = False
                    break
                if not MatchArray(inInstUses, inInst.GetUses(), variables):
                    matched = False
                    break

            if not matched:
                continue
                
            if CheckOnly:
                return True, None, None, None, None
            
            keepIn1 = inInstsPattern[0].get("keep", False)
            keepIn2 = inInstsPattern[1].get("keep", False)


            outInsts = []
            for outInstPattern in outInstsPattern:                  
                # Find regpairs from merged instructions
                usePatternArray = outInstPattern["use"]
                if usePatternArray == "pack":
                    regPairs = {}
                    for var in variables:
                        if var.startswith("reg_pack_low_") or var.startswith("reg_pack_high_"):
                            idx = 0 if "high" in var else 1
                            regPairs.setdefault(var.split("_")[-1], [None, None])[idx] = var
                    regPairs = sorted(regPairs.values())
                    usePatternArray =[f"{v[0]}:{v[1]}" for v in regPairs]
                usePatternArray = [v for v in usePatternArray if ":" in v]
                
                # Collect what pairs to jump to next via use-def chain
                regPairs = []
                for item in usePatternArray:
                    regs = item.split(":")
                    # Rn+1:Rn => (Rn, Rn+1)
                    regOp1 = variables.get(regs[1])
                    regOp2 = variables.get(regs[0])
                    
                    if regOp1 == None and regOp2 == None:
                        raise Exception("Either one should be not none")
                    
                    # If either not known, skip this pair
                    if not regOp1:
                        # TODO: generalize this to multiple defs
                        assert(len(Inst2.ReachingDefsSet.get(regOp2)) == 1) 
                        defInsts2 = list(Inst2.ReachingDefsSet.get(regOp2))[0][0]
                        resolveRegPairs = KnownRegPairs.get(defInsts2)
                        if not resolveRegPairs:
                            raise Exception("Reg pair not known yet")
                        
                        if len(resolveRegPairs) > 1:
                            raise Exception(f"{regOp2} found multiple pair resolution")
                        
                        resolveRegPair = list(resolveRegPairs)[0]
                        
                        variables[regs[1]] = resolveRegPair[0]
                        print(f"\tSkip ({regOp1}, {regOp2}), reg1 resolved to ({resolveRegPair[0]}, {resolveRegPair[1]})")
                        continue
                    if not regOp2:
                        # Not sure if this case is ever needed
                        raise Exception("Not implemented")
                        


                    # Handle cases such as RZ:reg1
                    if not regOp1.IsWritableReg xor not regOp2.IsWritableReg:
                        print(f"\tSkip ({regOp1}, {regOp2}) because pair is not writable")
                        continue
                    
                    # If an input is kept, we do not advance its def-use chain
                    if not keepIn1:
                        defInsts1 = Inst1.ReachingDefsSet.get(regOp1)
                    else:
                        defInsts1 = set()
                        defInsts1.add((regOp1.Parent, regOp1))

                    if not keepIn2:
                        defInsts2 = Inst2.ReachingDefsSet.get(regOp2)
                    else:
                        defInsts2 = set()
                        defInsts2.add((regOp2.Parent, regOp2))
                        
                    regPairs.append(((regOp1, defInsts1), (regOp2, defInsts2)))
                                        
                # Record def pairs to known reg pairs
                defPatternArray = outInstPattern["def"]
                for item in defPatternArray:
                    if ":" in item:
                        regs = item.split(":")
                        # Rn+1:Rn => (Rn, Rn+1)
                        regOp1 = variables.get(regs[1])
                        regOp2 = variables.get(regs[0])
                        
                        if regOp1 == None or regOp2 == None:
                            raise Exception("Def operand must have reg pair fully resolved")

                        # Impossible to have RZ:reg1 in def
                        if not regOp1.IsWritableReg or not regOp2.IsWritableReg:
                            raise Exception("Def operand must have writtable regs")
                        
                        # Add to known reg pairs
                        # For now, not including one-inst pattern 
                        # because they cause same reg to be shared by multiple pairs
                        # We cannot handle that yet
                        if not isOneInstPattern:
                            print(f"\tAdd known reg pair: ({regOp1}, {regOp2})")
                            KnownRegPairs.setdefault(Inst1, set()).add((regOp1.Reg, regOp2.Reg))
                            KnownRegPairs.setdefault(Inst2, set()).add((regOp1.Reg, regOp2.Reg))

                # Create the output instructions
                id = f"{Inst1.id}_x64"
                opcodes = GenArray(outInstPattern["opcodes"], variables)
                defs = GenArray(outInstPattern["def"], variables, Operand.Parse)
                uses = GenArray(outInstPattern["use"], variables, Operand.Parse)
                operands = defs + uses
                parent = Inst1.parent
                
                outInst = Instruction(
                    id=id,
                    opcodes=opcodes,
                    operands=operands,
                    parentBB=parent
                )
                outInsts.append(outInst)

                return True, outInsts, regPairs, keepIn1, keepIn2

        return False, None, None, None, None

    def GetMatchingPairs(self, RegPairs, KnownRegPairs):
        Reg1Defs = RegPairs[0][1]
        Reg2Defs = RegPairs[1][1]
        MatchingPairs = []
        for r1DefInst, r1DefOp in Reg1Defs:
            for r2DefInst, r2DefOp in Reg2Defs:
                if self.ControlEquivalent(r1DefInst, r2DefInst):
                    Status, _, _, _, _ = self.Match(r1DefInst, r2DefInst, KnownRegPairs, True)
                    if Status:
                        MatchingPairs.append(((r1DefInst, r1DefOp), (r2DefInst, r2DefOp)))
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

    def CreatePack64(self, InsertBefore, UseOp1, UseOp2):
        
        Inst = Instruction(
            id=f"{InsertBefore.id}_pack64_restore",
            opcodes=["PACK64"],
            operands=[dest_op, lo_def, hi_def],
            inst_content=f"PACK64 {dest_op.Name}, {lo_def.Name} {hi_def.Name}",
            parentBB=InsertBefore.parent
        )
        newDefOp = Operand()
        newDefOp.SetReg(f"pack64_{UseOp1.Name}_{UseOp2.Name}")
        newDefOp.Type = "U64"
        
        newInst = Instruction()
        newInst.opcodes = ["PACK64"]
        newInst.operands = [newDefOp, UseOp1, UseOp2]
        
        return newInst
            

    def ApplyChanges(self, func, InsertInsts, RemoveInsts):

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

                if inst in InsertInsts:
                    insertInst = InsertInsts[inst]
                    new_insts.extend(insertInst)

                
                if inst not in RemoveInsts:
                    new_insts.append(inst)

            bb.instructions = new_insts
