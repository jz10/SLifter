from transform.transform import SaSSTransform
from collections import deque

class TypeAnalysis(SaSSTransform):
    def __init__(self, name):
        super().__init__(name)

        # PROP means it is the same type as the other operands
        # PROP_PTR means it is a pointer to the other operands
        self.instructionTypeTable = {
            "FADD": ["Float32", "Float32", "Float32", "NA", "NA"],
            "S2R": ["Int32", "Int32", "Int32", "NA", "NA"],
            "IMAD": ["Int32", "Int32", "Int32", "Int32", "NA"],
            "IADD3": ["Int32", "Int32", "Int32", "Int32", "NA"],
            "XMAD": ["Int32", "Int32", "Int32", "Int32", "NA"],
            "IADD32I": ["Int32", "Int32", "NA", "NA", "NA"],
            "MOV": ["Int32", "Int32", "NA", "NA", "NA"],
            "IADD": ["Int32", "Int32", "Int32", "NA", "NA"],
            "ISETP": ["Int1", "NA", "NA", "NA", "NA"],
            "AND": ["PROP", "PROP", "PROP", "NA", "NA"],
            "OR": ["PROP", "PROP", "PROP", "NA", "NA"],
            "XOR": ["PROP", "PROP", "PROP", "NA", "NA"],
            "NOT": ["PROP", "PROP", "PROP", "NA", "NA"],
            "LOP": ["PROP", "PROP", "PROP", "NA", "NA"],
            "SHL": ["PROP", "PROP", "PROP", "NA", "NA"],
            "SHR": ["PROP", "PROP", "PROP", "NA", "NA"],
            "LOP3": ["PROP", "PROP", "PROP", "PROP", "NA"],
            "LDG": ["PROP", "PROP_PTR", "NA", "NA", "NA"],
            "SULD": ["PROP", "PROP_PTR", "NA", "NA", "NA"],
            "STG": ["PROP_PTR", "PROP", "NA", "NA", "NA"],
            "SUST": ["PROP_PTR", "PROP", "NA", "NA", "NA"],
            "PHI": ["PROP", "PROP", "PROP", "PROP", "NA"],
            "F2I": ["Int32", "Float32", "NA", "NA", "NA"],
            "I2F": ["Float32", "Int32", "NA", "NA", "NA"],
            "INTTOPTR": ["PROP_PTR", "Int32", "NA", "NA", "NA"],
            "NOP": ["NA", "NA", "NA", "NA", "NA"],
            "EXIT": ["NA", "NA", "NA", "NA", "NA"],
            "RET": ["NA", "NA", "NA", "NA", "NA"],
            "SYNC": ["NA", "NA", "NA", "NA", "NA"],
            "SSY": ["NA", "NA", "NA", "NA", "NA"],
        }
        
        self.flagOverrideTable = {
            "WIDE": ["Int64", "NA", "NA", "NA", "NA"]
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
                Changed |= self.ProcessBB(BB, OpTypes, False)
            
            for BB in reversed(WorkList):
                Changed |= self.ProcessBB(BB, OpTypes, True)

            Iteration += 1
            if Iteration > 3:
                print("Warning: TypeAnalysis exceeds 3 iterations, stopping")
                break

        # Apply types to instructions
        for BB in WorkList:
            for Inst in BB.instructions:
                for Operand in Inst.operands:
                    if Operand.Name in OpTypes:
                        Operand._TypeDesc = OpTypes[Operand.Name]
                    else:
                        Operand._TypeDesc = "NOTYPE"


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
        

    def ProcessBB(self, BB, OpTypes, Reversed):
        CurrentState = OpTypes.copy()

        if Reversed:
            Instructions = reversed(BB.instructions)
        else:
            Instructions = BB.instructions

        for Inst in Instructions:
            self.PropagateTypes(Inst, CurrentState)
        
        # self.printTypes(CurrentState)
        Changed = (CurrentState != OpTypes)
        OpTypes.update(CurrentState)          

        return Changed

    def PropagateTypes(self, Inst, OpTypes):
        # Get Inst opcode
        idx = 0
        if Inst.IsPredicateReg(Inst._opcodes[idx]):
            idx += 1
        op = Inst._opcodes[idx]

        if op not in self.instructionTypeTable:
            print(f"Warning: Unhandled opcode {op} in {Inst._InstContent}")
            return
        
        # Static resolve types
        propType = "NOTYPE"
        for i, operand in enumerate(Inst.operands):
            typeDesc = self.instructionTypeTable[op][i]

            if typeDesc != "NA" and "PROP" not in typeDesc:

                if operand.Name in OpTypes and OpTypes[operand.Name] != typeDesc:
                    print(f"Warning: Type mismatch for {operand._Name} in {Inst._InstContent}: {OpTypes[operand.Name]} vs {typeDesc}")

                OpTypes[operand.Name] = typeDesc

        # Find propagate type
        for i, operand in enumerate(Inst.operands):
            typeDesc = self.instructionTypeTable[op][i]

            if typeDesc == "PROP":
                if operand.Name in OpTypes:
                    if propType != "NOTYPE" and OpTypes[operand.Name] != propType:
                        print(f"Warning: Propagation type mismatch for {operand._Name} in {Inst._InstContent}: {OpTypes[operand.Name]} vs {propType}")
                        
                    propType = OpTypes[operand.Name]

        # Propagate types
        if propType != "NOTYPE":
            for i, operand in enumerate(Inst.operands):
                typeDesc = self.instructionTypeTable[op][i]

                if typeDesc == "PROP":
                    OpTypes[operand.Name] = propType
                elif typeDesc == "PROP_PTR":
                    OpTypes[operand.Name] = propType + "_PTR"

        # Flag overrides
        for i, opcode in enumerate(Inst.opcodes):
            if opcode in self.flagOverrideTable:
                for j, operand in enumerate(Inst.operands):
                    if self.flagOverrideTable[opcode][j] != "NA":
                        OpTypes[operand.Name] = self.flagOverrideTable[opcode][j]

        # Operand overrides
        for i, operand in enumerate(Inst.operands):
            if operand.Name == "PT":
                OpTypes[operand.Name] = "Int1"
            elif operand.IsArg:
                OpTypes[operand.Name] = "Int32"