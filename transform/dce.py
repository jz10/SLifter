from collections import deque
from transform.transform import SaSSTransform
from sir.operand import Operand
from sir.instruction import Instruction

class DCE(SaSSTransform):
    def apply(self, module):
        print("=== Start of DCE ===")

        for func in module.functions:
            print(f"Processing function: {func.name}")
            self.process(func)

        print("=== End of DCE ===")

    def process(self, func):
        side_effect_instructions = set([
            "STG",    
            "ST",     
            "STS",    
            "SUST",   
            "ATOM",   
            "RED",
            "BAR", 
            "CALL",   
            "RET",    
            "EXIT",   
            "BRA",    
            "PBRA",   
            "VOTE",   
            "SHFL",   
            "LDG",    
            "LD",     
            "LDS",    
            "SULD",
            "MATCH",
            "RED",
        ])

        live = []
        queue = deque()

        for block in func.blocks:
            for inst in block.instructions:
                if inst.opcodes and inst.opcodes[0] in side_effect_instructions:
                    live.append(inst)
                    queue.append(inst)
        
        while queue:
            inst = queue.popleft()
            for op, def_inst in inst.ReachingDefs.items():
                if def_inst not in live:
                    live.append(def_inst)
                    queue.append(def_inst)


        removedInsts = set()
        for block in func.blocks:
            new_instructions = []
            for inst in block.instructions:
                if inst not in live:
                    removedInsts.add(inst)
                else:
                    new_instructions.append(inst)

            # print out instructions
            for inst in block.instructions:
                if inst not in removedInsts:
                    print(f"{inst}")
                else:
                    print(f"{inst} => REMOVED")

            block.instructions = new_instructions
