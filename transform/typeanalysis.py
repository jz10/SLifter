from transform.transform import SaSSTransform
from collections import deque

class TypeAnalysis(SaSSTransform):
    def apply(self, module):
        for func in module.functions:
            self.ProcessFunc(func)


    def ProcessFunc(self, function):
        WorkList = self.TraverseCFG(function)

        RegTypes = {}
        print("=== Start of TypeAnalysis ===")

        Changed = True

        Iteration = 0
        
        while Changed:

            # for BB in WorkList:
            #     for Inst in BB.instructions:
            #         print(Inst._InstContent+" => ", end="")
            #         for Operand in Inst.operands:
            #             print(Operand._TypeDesc+", ",end="")
            #         print("")

            # print(".")

            Changed = False

            for BB in WorkList:
                Changed |= self.ProcessBB(BB, RegTypes, False)
            
            for BB in reversed(WorkList):
                Changed |= self.ProcessBB(BB, RegTypes, True)

            Iteration += 1
            if Iteration > 3:
                print("Warning: TypeAnalysis exceeds 3 iterations, stopping")
                break

        for BB in WorkList:
            for Inst in BB.instructions:
                print(Inst._InstContent+" => ", end="")
                for Operand in Inst.operands:
                    print(Operand._TypeDesc+", ",end="")
                print("")

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
        

    def ProcessBB(self, BB, RegTypes, Reversed):
        CurrentState = RegTypes.copy()

        if Reversed:
            Instructions = reversed(BB.instructions)
        else:
            Instructions = BB.instructions

        for Inst in Instructions:
            self.PropagateTypes(Inst, CurrentState)
        
        # self.printTypes(CurrentState)
        Changed = (CurrentState != RegTypes)
        RegTypes.update(CurrentState)          

        return Changed

    def PropagateTypes(self, Inst, RegTypes):
        # Get Inst opcode       
        idx = 0
        if  Inst.IsPredicateReg(Inst._opcodes[idx]):
            idx += 1
        op = Inst._opcodes[idx]

        FLOAT32_OPS = {
            "FADD"
        }

        INT32_OPS = {
            "S2R", "IMAD", "IADD3", "XMAD", "IADD32I",
            "MOV" # TODO: assume MOV is always int32?
        }

        # int32 or int64 depending on the operand
        INT_OPS = {
            "IADD", "ISETP",
            "AND", "OR", "XOR", "NOT", "LOP", "LOP3",
            "SHL", "SHR", 
        }

        LD_OPS = {
            "LDG", "SULD"
        }

        ST_OPS = {
            "STG", "SUST"
        }

        CAST_OPS = {
            "F2I", "I2F", "INTTOPTR"
        }

        SKIP_OPS = {
            "NOP", "EXIT", "RET", "SYNC", "SSY"
        }

        # Force all operands to be float32
        if op in FLOAT32_OPS:
            for operand in Inst.operands:
                operand.SetTypeDesc("Float32")
                RegTypes[operand.Name] = operand.GetTypeDesc()

        # Make operands have the same type, except for predicate registers(for ISETP)
        elif op in INT_OPS:
            type = None
            for operand in Inst.operands:
                if operand.GetTypeDesc() != "NOTYPE" and not Inst.IsPredicateReg(operand.Name):
                    if type is None:
                        if type is not None and type != operand.GetTypeDesc():
                            print(f"Warning: Type mismatch in {Inst._InstContent}: {operand.Name} type overwriting from {operand.GetTypeDesc()} to {type}")
                        type = operand.GetTypeDesc()

            if type is not None:
                for operand in Inst.operands:
                    if not Inst.IsPredicateReg(operand.Name):
                        operand.SetTypeDesc(type)
                        RegTypes[operand.Name] = operand.GetTypeDesc()
                    
            # Force all predicate registers to be int1          
            for operand in Inst.operands:
                if Inst.IsPredicateReg(operand.Name):
                    if operand.GetTypeDesc() != "NOTYPE" and operand.GetTypeDesc() != "Int1":
                        print(f"Warning: Type mismatch in {Inst._InstContent}: {operand.Name} type overwriting from {operand.GetTypeDesc()} to i1")
                    
                    operand.SetTypeDesc("Int1")
                    RegTypes[operand.Name] = operand.GetTypeDesc()

        # Force all operands to be int32, except for definition operand if wide flag is set
        elif op in INT32_OPS:
            for operand in Inst.operands:
                operand.SetTypeDesc("Int32")
                RegTypes[operand.Name] = operand.GetTypeDesc()

            Def = Inst.GetDef()
            if Def is not None and len(Inst.opcodes) > 1 and Inst.opcodes[1] == "WIDE":
                Def.SetTypeDesc("Int64")
                RegTypes[Def.Name] = Def.GetTypeDesc()
        
        # Guess ptr type from value type
        elif op in LD_OPS:
            val_type = Inst.operands[0].GetTypeDesc()
            ptr_type = Inst.operands[1].GetTypeDesc()

            if val_type != "NOTYPE":

                if ptr_type != "NOTYPE" and ptr_type != val_type + "_PTR":
                    print(f"Warning: Type mismatch in {Inst._InstContent}: ptr overwriting from {ptr_type} to {val_type}_PTR")

                Inst.operands[1].SetTypeDesc(val_type + "_PTR")
                RegTypes[Inst.operands[1].Name] = Inst.operands[1].GetTypeDesc()

        elif op in ST_OPS:
            ptr_type = Inst.operands[0].GetTypeDesc()
            val_type = Inst.operands[1].GetTypeDesc()

            if val_type != "NOTYPE":

                if ptr_type != "NOTYPE" and ptr_type != val_type + "_PTR":
                    print(f"Warning: Type mismatch in {Inst._InstContent}: ptr overwriting from {ptr_type} to {val_type}_PTR")

                Inst.operands[0].SetTypeDesc(val_type + "_PTR")
                RegTypes[Inst.operands[0].Name] = Inst.operands[0].GetTypeDesc()

        elif op == "PHI":
            # We want operands to have the same type
            type = None
            for operand in Inst.operands:
                if operand.GetTypeDesc() != "NOTYPE":
                    if type is None:
                        type = operand.GetTypeDesc()
                    elif type != operand.GetTypeDesc():
                        print(f"Warning: Type mismatch in {Inst._InstContent}: {operand.Name} type overwriting from {operand.GetTypeDesc()} to {type}")
            
            if type is not None:
                for operand in Inst.operands:
                    operand.SetTypeDesc(type)
                    RegTypes[operand.Name] = operand.GetTypeDesc()

        elif op in CAST_OPS:
            if op == "F2I":
                Inst.operands[0].SetTypeDesc("Float32")
            elif op == "I2F":
                Inst.operands[0].SetTypeDesc("Int32")
            elif op == "INTTOPTR":
                if Inst.operands[0].GetTypeDesc() != "NOTYPE" and Inst.operands[0].GetTypeDesc() != "Int32":
                    print(f"Warning: Type mismatch in {Inst._InstContent}: {Inst.operands[0].Name} type overwriting from {Inst.operands[0].GetTypeDesc()} to Int32")
                Inst.operands[0].SetTypeDesc("Int32")
                RegTypes[Inst.operands[0].Name] = Inst.operands[0].GetTypeDesc()
        else:
            if op not in SKIP_OPS:
                print(f"Warning: Unhandled opcode {op} in {Inst._InstContent}")
        
        for operand in Inst.operands:
            if operand.Name in RegTypes:
                if operand.GetTypeDesc() != "NOTYPE" and operand.GetTypeDesc() != RegTypes[operand.Name]:
                    print(f"Warning: Type mismatch in {Inst._InstContent}: {operand.Name} type overwriting from {operand.GetTypeDesc()} to {RegTypes[operand.Name]}")
                
                operand.SetTypeDesc(RegTypes[operand.Name])