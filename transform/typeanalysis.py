from transform.transform import SaSSTransform
from collections import deque

class TypeAnalysis(SaSSTransform):
    def __init__(self, name):
        super().__init__(name)

        # operands with PROP must have the same type
        # operands with PROP_PTR must have the same type but with _PTR suffix
        # operands with ANY can have any type
        self.instructionTypeTable = {
            "FADD": ["Float32", "Float32", "Float32", "NA", "NA", "NA"],
            "FFMA": ["Float32", "Float32", "Float32", "Float32", "NA", "NA"],
            "MUFU": ["Float32", "Float32", "Float32", "NA", "NA", "NA"],
            "S2R": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "IMAD": ["Int32", "Int32", "Int32", "Int32", "NA", "NA"],
            "IADD3": ["Int32", "Int32", "Int32", "Int32", "NA", "NA"],
            "XMAD": ["Int32", "Int32", "Int32", "Int32", "NA", "NA"],
            "IADD32I": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "MOV": ["Int32", "Int32", "NA", "NA", "NA", "NA"],
            "IADD": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "ISETP": ["Int1", "Int32", "Int32", "Int32", "Int32", "NA"],
            "AND": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "OR": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "XOR": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "NOT": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "LEA": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "LOP": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "SHL": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "SHR": ["PROP", "PROP", "PROP", "NA", "NA", "NA"],
            "LOP3": ["PROP", "PROP", "PROP", "PROP", "NA", "NA"],
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
            "SSY": ["NA", "NA", "NA", "NA", "NA", "NA"],
            "SHF": ["Int32", "Int32", "Int32", "Int32", "NA", "NA"],
            "DEPBAR": ["NA", "NA", "NA", "NA", "NA", "NA"],
            "LOP32I": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "ISCADD": ["Int32", "Int32", "Int32", "NA", "NA", "NA"],
            "MOV32I": ["Int32", "Int32", "NA", "NA", "NA", "NA"],
            "IABS": ["Int32", "Int32", "NA", "NA", "NA", "NA"],

            # Dummy instruction types
            "PHI": ["PROP", "PROP", "PROP", "PROP", "NA", "NA"],
            "INTTOPTR": ["PROP_PTR", "Int64", "NA", "NA", "NA", "NA"],
            "PACK64": ["Int64", "Int32", "Int32", "NA", "NA", "NA"],
            "CAST64": ["Int64", "Int32", "NA", "NA", "NA", "NA"],
            "IADD64": ["Int64", "Int64", "Int64", "NA", "NA", "NA"],
            "IMAD64": ["Int64", "Int64", "Int64", "Int64", "NA", "NA"],
            "SHL64": ["Int64", "Int64", "Int64", "NA", "NA", "NA"],
            "MOV64": ["Int64", "Int64", "NA", "NA", "NA", "NA"],
            "IADD32I64": ["Int64", "Int64", "Int64", "NA", "NA", "NA"],
            "PHI64": ["Int64", "Int64", "Int64", "Int64", "Int64", "NA"],
            "BITCAST": ["ANY", "ANY", "NA", "NA", "NA", "NA"],
            "PBRA": ["Int1", "NA", "NA", "NA", "NA", "NA"],
            "LEA64": ["Int64", "Int64", "Int64", "Int64", "NA", "NA"],
            "SETZERO": ["PROP", "NA", "NA", "NA", "NA", "NA"],
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
            #         print(Inst+" => ", end="")
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
                    if Operand in OpTypes:
                        Operand._TypeDesc = OpTypes[Operand]
                    elif Operand.Reg in OpTypes:
                        Operand._TypeDesc = OpTypes[Operand.Reg]
                    else:
                        Operand._TypeDesc = "NOTYPE"


        for BB in WorkList:
            for Inst in BB.instructions:
                print(str(Inst)+" => ", end="")
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
    
    def ResolveType(self, Inst, i):
        op = Inst._opcodes[0]

        # Special case for the predicate register in instructions with multiple defs
        # E.g. LEA R12, P0 = R6.reuse, R4, 0x2, because of P0, decrease index by 1 to get correct entry in table
        if len(Inst.GetDefs()) > 1 and i > 1:
            typeDesc = self.instructionTypeTable[op][i-1]
        else:
            typeDesc = self.instructionTypeTable[op][i]

        # # Flag overrides
        # for j, flag in enumerate(Inst.opcodes[1:], start=1):
        #     if flag in self.flagOverrideTable and self.flagOverrideTable[flag][i] != "NA":
        #         typeDesc = self.flagOverrideTable[flag][i]
        #         break

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

                if operand.Reg in OpTypes and OpTypes[operand.Reg] != typeDesc:
                    print(f"Warning: Type mismatch for {operand.Reg} in {Inst}: {OpTypes[operand.Reg]} vs {typeDesc}")

                if operand.IsReg:
                    OpTypes[operand.Reg] = typeDesc
                else:
                    # Store operand itself as key for non-register values
                    # E.g. 0 in IADD vs 0 in FADD have different types
                    OpTypes[operand] = typeDesc 

        # Find propagate type
        for i, operand in enumerate(Inst.operands):
            typeDesc = self.ResolveType(Inst, i)

            if typeDesc == "PROP":
                if operand.Name in OpTypes:
                    if propType != "NOTYPE" and OpTypes[operand.Name] != propType:
                        print(f"Warning: Propagation type mismatch for {operand.Name} in {Inst}: {OpTypes[operand.Name]} vs {propType}")
                        
                    propType = OpTypes[operand.Name]

        # Propagate types
        if propType != "NOTYPE":
            for i, operand in enumerate(Inst.operands):
                typeDesc = self.ResolveType(Inst, i)

                if typeDesc == "PROP":
                    OpTypes[operand.Name] = propType
                elif typeDesc == "PROP_PTR":
                    OpTypes[operand.Name] = propType + "_PTR"
