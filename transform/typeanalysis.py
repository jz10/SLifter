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
        if self.IsPredicateReg(Inst._opcodes[idx]):
            idx += 1
        op = Inst._opcodes[idx]

        FLOAT32_OPS = {
            "FADD"
        }

        INT32_OPS = {
            "S2R", "IMAD"
        }

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

        SKIP_OPS = {
            "NOP", "EXIT", "RET"
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
                if operand.GetTypeDesc() != "NOTYPE" and not self.IsPredicateReg(operand.Name):
                    if type is None:
                        if type is not None and type != operand.GetTypeDesc():
                            print(f"Warning: Type mismatch in {Inst._InstContent}: {operand.Name} type overwriting from {operand.GetTypeDesc()} to {type}")
                        type = operand.GetTypeDesc()

            if type is not None:
                for operand in Inst.operands:
                    if not self.IsPredicateReg(operand.Name):
                        operand.SetTypeDesc(type)
                        RegTypes[operand.Name] = operand.GetTypeDesc()
                    
            # Force all predicate registers to be int1          
            for operand in Inst.operands:
                if self.IsPredicateReg(operand.Name):
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

        else:
            if op not in SKIP_OPS:
                print(f"Warning: Unhandled opcode {op} in {Inst._InstContent}")
        
        for operand in Inst.operands:
            if operand.Name in RegTypes:
                if operand.GetTypeDesc() != "NOTYPE" and operand.GetTypeDesc() != RegTypes[operand.Name]:
                    print(f"Warning: Type mismatch in {Inst._InstContent}: {operand.Name} type overwriting from {operand.GetTypeDesc()} to {RegTypes[operand.Name]}")
                
                operand.SetTypeDesc(RegTypes[operand.Name])

    def IsPredicateReg(self, opcode):
        if opcode[0] == 'P' and (opcode[1].isdigit() or opcode[1] == 'T'):
            return True
        if opcode[0] == '!' and opcode[1] == 'P' and (opcode[2].isdigit() or opcode[2] == 'T'):
            return True
        return False