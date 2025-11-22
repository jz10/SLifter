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
INT = INT32 | INT64
NUM1 = BOOL
NUM32 = INT32 | FLOAT32 | HALF2
NUM64 = INT64 | FLOAT64
TOP = NUM32 | NUM1 | NUM64
BOTTOM = set()

class TypeAnalysis(SaSSTransform):
        
    def __init__(self):
        super().__init__()
        
        self.instruction_type_table = {
            "S2R": [[INT], [INT, INT]],
            "IMAD": [[INT32], [INT32, INT32, INT32]],
            "IADD3": [[INT32], [INT32, INT32, INT32]],
            "XMAD": [[INT32], [INT32, INT32, INT32]],
            "IADD32I": [[INT32], [INT32, INT32]],
            "IADD": [[INT32], [INT32, INT32]],
            "ISETP": [[BOOL, BOOL], [INT32, INT32, BOOL, BOOL]],
            "LEA": [[INT32, BOOL], [INT32, INT32, INT32, INT32]],
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
            "SHL": [[INT32], [INT32, INT32, INT32]],
            
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
            "SHL64": [[INT64], [INT64, INT64, INT64]],
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

        self.modifier_override_table = {
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
        
        self.propagate_table = {
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
            self.process_func(func)
        print("=== End of TypeAnalysis ===")


    def process_func(self, function):
        work_list = self.traverse_cfg(function)

        optypes = {}
        
        # Static type resolution
        for bb in work_list:
            for inst in bb.instructions:
                self.resolve_types(inst, optypes)

        changed = True
        iteration = 0
        
        while changed:

            # for BB in work_list:
            #     for Inst in BB.instructions:
            #         print(str(Inst)+" => ", end="")
            #         for operand in Inst.operands:
            #             if operand in optypes:
            #                 TypeDesc = str(optypes[operand])
            #             elif operand.reg in optypes:
            #                 TypeDesc = str(optypes[operand.reg])
            #             else:
            #                 TypeDesc = "NOTYPE"
            #             print(TypeDesc+", ",end="")
            #         print("")

            # print("-----next iteration-----")

            changed = False
            # self.conflicts = {}
            # self.phi_conflicts = {}
            
            for bb in work_list:
                changed |= self.process_bb(bb, optypes)

            # # If there is any conflict, insert bitcast instructions
            # # After that, re-run defuse, resolve types again
            # if len(self.conflicts) > 0 or len(self.phi_conflicts) > 0:
            #     for bb in work_list:
            #         new_instructions = []
            #         for inst in bb.instructions:
            #             if inst in self.conflicts:
            #                 self.insert_bitcast_before(inst, bb, new_instructions, optypes)
            #             new_instructions.append(inst)
            #             if inst in self.phi_conflicts:
            #                 self.insert_bitcast_after(inst, bb, new_instructions, optypes)
            #         bb.instructions = new_instructions
                    
            #     self.defuse.build_def_use(function.blocks)
                
            #     optypes = {}
            #     for bb in work_list:
            #         for inst in bb.instructions:
            #             self.resolve_types(inst, optypes)
                
                
            iteration += 1
            if iteration > 3:
                print("Warning: TypeAnalysis exceeds 3 iterations, stopping")
                break

        # Apply types to instructions
        for bb in work_list:
            for inst in bb.instructions:
                for op in inst.operands:
                    op.type_desc = self.get_type_desc(op, optypes)

        for bb in work_list:
            for inst in bb.instructions:
                print(str(inst)+" => ", end="")
                for op in inst.operands:
                    print(op.type_desc+", ",end="")
                print("")

        # # Statistics
        # # Build AllRegs from the function CFG, then classify:
        # # - If reg in ConflictedOriginalRegs -> Conflicted
        # # - Else if reg in BitcastRegs -> skip (synthetic)
        # # - Else if reg has a type in optypes -> count by that type
        # # - Else -> Unresolved
        # AllRegs = set()
        # for BB in WorkList:
        #     for Inst in BB.instructions:
        #         for op in Inst.operands:
        #             if op.is_writable_reg:
        #                 AllRegs.add(op.reg)

        # TypeCount = {}
        # for reg in AllRegs:
        #     # Skip synthetic bitcast temps entirely
        #     if reg in self.BitcastRegs:
        #         continue

        #     if reg in self.ConflictedOriginalRegs:
        #         TypeCount["Conflicted"] = TypeCount.get("Conflicted", 0) + 1
        #         continue

        #     if reg in optypes and not isinstance(reg, Operand):
        #         tdesc = optypes[reg]
        #         TypeCount[tdesc] = TypeCount.get(tdesc, 0) + 1
        #     else:
        #         print(f"Warning: Unresolved type for register {reg}")
        #         TypeCount["Unresolved"] = TypeCount.get("Unresolved", 0) + 1

        # print("Type analysis statistics")
        # for type_desc, count in TypeCount.items():
        #     print(f"Type counts {type_desc}: {count} registers")

        print("done")
        print("=== End of TypeAnalysis ===")
        
    # def insert_bitcast_before(self, inst, bb, new_instructions, optypes):
    #     op, old_type, new_type = self.conflicts[inst]
    #     print(f"Warning: Inserting BITCAST to resolve type conflict for {op} before {inst}: {old_type} vs {new_type}")
    #     # Insert bitcast before Inst
    #     src_reg = Operand.from_reg(op.reg, op.reg)
    #     new_reg_name = f"{src_reg.reg}_bitcast"
    #     dest_reg = Operand.from_reg(new_reg_name, new_reg_name)

    #     bitcast_inst = Instruction(
    #         id=f"{inst.id}_type_resolve", 
    #         opcodes=["BITCAST"],
    #         operands=[dest_reg, src_reg],
    #         parentBB=bb
    #     )
    #     new_instructions.append(bitcast_inst)

    #     # # Book-keeping for statistics
    #     # self.ConflictedOriginalRegs.add(orig_reg)
    #     # self.BitcastRegs.add(NewRegName)

    #     optypes[dest_reg.reg] = old_type
    #     optypes[src_reg.reg] = new_type
    #     op.set_reg(dest_reg.reg)
        
    # def insert_bitcast_after(self, inst, bb, new_instructions, optypes):
    #     op, old_type, new_type, phi_def_op = self.phi_conflicts[inst]
    #     print(f"Warning: Inserting BITCAST to resolve type conflict for {op} after {inst}: {old_type} vs {new_type}")
    #     # Insert bitcast before Inst
    #     src_reg = Operand.from_reg(op.reg, op.reg)
    #     new_reg_name = f"{src_reg.reg}_bitcast"
    #     dest_reg = Operand.from_reg(new_reg_name, new_reg_name)

    #     bitcast_inst = Instruction(
    #         id=f"{inst.id}_type_resolve", 
    #         opcodes=["BITCAST"],
    #         operands=[dest_reg, src_reg],
    #         parentBB=bb
    #     )
    #     new_instructions.append(bitcast_inst)

    #     optypes[src_reg.reg] = old_type
    #     optypes[dest_reg.reg] = new_type
        
    #     phi_def_op.set_reg(new_reg_name)
        
    def get_type_desc(self, operand, optypes):
        if operand in optypes:
            types = optypes[operand]
        elif operand.reg in optypes:
            types = optypes[operand.reg]
        else:
            raise ValueError(f"Type not found for operand {operand} / {operand.reg}")
        
        if len(types) > 1:
            print(f"Warning: Multiple possible types for {operand} / {operand.reg}: {types}")
            
        if len(INT32 & types) > 0:
            return list(INT32)[0]
        elif len(INT64 & types) > 0:
            return list(INT64)[0]
        else:
            return list(types)[0]
        
    def resolve_types(self, inst, optypes):
        
        # Static type table
        opcode = inst.opcodes[0]
        type_table = self.instruction_type_table.get(opcode, None)
        if not type_table:
            print(f"Warning: Unknown opcode {opcode} in StaticResolve")
            
            # Default to TOP
            for op in inst.operands:
                self.set_op_type(optypes, op, TOP, inst)
            return
        
        optype_map = {}
    
        for i, def_op in enumerate(inst.get_defs()):
            if type_table[0][i] != "PROP":
                optype_map[def_op] = type_table[0][i]
            else:
                optype_map[def_op] = TOP
                
        for i, use_op in enumerate(inst.get_uses()):
            if type_table[1][i] != "PROP":
                optype_map[use_op] = type_table[1][i]
            else:
                optype_map[use_op] = TOP
                
        # Modifier table
        modifier_table = self.modifier_override_table.get(opcode, None)
        if modifier_table:
            for mod in modifier_table:
                if mod not in inst.opcodes[1:]:
                    continue
                def_overrides, use_overrides = modifier_table[mod]
                
                for i, def_op in enumerate(inst.get_defs()):
                    if i < len(def_overrides):
                        optype_map[def_op] = def_overrides[i]
                        
                for i, use_op in enumerate(inst.get_uses()):
                    if i < len(use_overrides):
                        optype_map[use_op] = use_overrides[i]
                        
        # Apply types
        for operand, type_desc in optype_map.items():
            if operand.is_predicate_reg or operand.is_pt:
                self.set_op_type(optypes, operand, BOOL, inst)
            else:
                self.set_op_type(optypes, operand, type_desc, inst)
                    
        # Special case for pred?
        # if operand.is_predicate_reg or operand.is_pt:
        #     typeDesc = BOOL
            

    def get_optype(self, optypes, operand):
        if operand in optypes:
            return optypes[operand]
        elif operand.reg in optypes:
            return optypes[operand.reg]
        else:
            return TOP

    def set_op_type(self, optypes, operand, type_set, inst):
        
        previous_type = optypes.get(operand.reg, TOP)
        
        # Record type conflict
        if len(previous_type & type_set) == 0:
            print(f"Warning: Type mismatch for {operand.reg} in {inst}: prevType: {previous_type} vs optype: {type_set}")
            
            # if inst.opcodes[0] != "PHI":
            #     self.conflicts[inst] = (operand, type_set, previous_type)
            # else:
            #     # Cannot insert bitcast before phi,
            #     # need to search and insert after non-phi def instructions
            #     self.add_phi_conflicts(optypes, operand, type_set, previous_type, inst)
            optypes[operand] = type_set
            return
            
        if operand.is_writable_reg:
            optypes[operand.reg] = previous_type & type_set
        else:
            # Store operand itself as key for non-register values
            # E.g. 0 in IADD vs 0 in FADD have different types
            optypes[operand] = previous_type & type_set
            
    # def add_phi_conflicts(self, optypes, operand, new_type, old_type, inst):
    #     queue = deque()
    #     prev_queue = deque()
    #     visited = set()
        
    #     queue.extend(list(inst.ReachingDefsSet[operand]))
    #     prev_queue.append(operand)
        
    #     while queue:
    #         curr_inst, curr_def_op = queue.popleft()
    #         phi_use_op = prev_queue.popleft()
            
    #         if curr_inst in visited:
    #             continue
    #         visited.add(curr_inst)
            
    #         if curr_inst.opcodes[0] != "PHI":
    #             if len(self.get_optype(optypes, curr_def_op) & new_type) == 0:
    #                 self.phi_conflicts[curr_inst] = (curr_def_op, self.get_optype(optypes, curr_def_op), new_type, phi_use_op)
    #             continue
            
    #         # Give PHI another chance to repropagate after the bitcasts are inserted
    #         optypes[curr_def_op.reg] = TOP 
            
    #         for use_op, def_inst_pair in curr_inst.ReachingDefsSet.items():
    #             queue.extend(list(def_inst_pair))
    #             prev_queue.append(use_op)

    def traverse_cfg(self, function):
        entry_bb = function.blocks[0]
        visited = set()
        queue = deque([entry_bb])
        work_list = []

        while queue:
            curr_bb = queue.popleft()
            
            if curr_bb in visited:
                continue

            visited.add(curr_bb)
            work_list.append(curr_bb)

            for succ_bb in curr_bb.succs:
                if succ_bb not in visited:
                    queue.append(succ_bb)

        return work_list

    def process_bb(self, bb, optypes):
        old_op_types = optypes.copy()

        for inst in bb.instructions:
            self.propagate_types(inst, optypes)
        
        return old_op_types != optypes

    def propagate_types(self, inst, optypes):
        opcode = inst.opcodes[0]
        
        if opcode not in self.propagate_table:
            return
        
        prop_table = self.propagate_table[opcode]
        
        prop_ops = {}
        for i, def_op in enumerate(inst.get_defs()):
            prop_ops.setdefault(prop_table[0][i], []).append(def_op)
                
        for i, use_op in enumerate(inst.get_uses()):
            prop_ops.setdefault(prop_table[1][i], []).append(use_op)

        for prop_key in prop_ops:
            prop_ops_list = prop_ops[prop_key]

            prop_type = TOP
            for prop_op in prop_ops_list:
                op_type = self.get_optype(optypes, prop_op)
                
                if len(prop_type & op_type) != 0:
                    prop_type = prop_type & op_type
                
            for prop_op in prop_ops_list:
                self.set_op_type(optypes, prop_op, prop_type, inst)
