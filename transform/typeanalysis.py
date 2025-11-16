from transform.transform import SaSSTransform
from sir.operand import Operand
from collections import deque
from sir.instruction import Instruction
from transform.defuse_analysis import DefUseAnalysis

# Type domains
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

    """
    Modified TypeAnalysis that:
    - does not introduce pointer types,
    - assumes conflicts only occur among same width (e.g., Int32 vs Float32; Int64 vs Float64),
    - defers conflicts and repairs them by inserting BITCASTs at use sites (or PHI edges),
    - chooses a canonical type per SSA def to ensure a single register type in the final IR.
    """

    def __init__(self):
        super().__init__()

        # Static instruction type table (unchanged; no pointer types)
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

        # Propagation for polymorphic instructions
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

        # State
        self.defuse = DefUseAnalysis()
        self.CanonicalType = {}  # reg -> single-type set (e.g., INT32 or FLOAT32)

    def apply(self, module):
        super().apply(module)
        print("=== Start of TypeAnalysis ===")
        for func in module.functions:
            print(f"Processing function: {func.name}")
            self.ProcessFunc(func)
        print("=== End of TypeAnalysis ===")

    def ProcessFunc(self, function):
        self.WorkList = self.TraverseCFG(function)
        self.currFunction = function

        # Build def-use upfront
        self.defuse.BuildDefUse(function.blocks)

        # 1) Static typing
        OpTypes = {}
        for BB in self.WorkList:
            for Inst in BB.instructions:
                self.ResolveTypes(Inst, OpTypes)

        # 2) Propagate to fixed point (soft meets, no immediate edits)
        Changed = True
        Iteration = 0
        while Changed and Iteration < 10:
            Changed = False
            for BB in self.WorkList:
                Changed |= self.ProcessBB(BB, OpTypes)
            Iteration += 1

        # 3) Choose canonical type per SSA def (single type per register)
        self.CanonicalType = self.build_canonical_types(OpTypes)

        # 4) Reconcile uses: insert BITCAST at uses (or PHI-edges) if needed
        #    under the assumption: conflicts only among same width.
        if self.reconcile_use_site_casts(function, OpTypes):
            # IR changed; rebuild def-use, re-run typing + propagation
            self.defuse.BuildDefUse(function.blocks)
            OpTypes = {}
            for BB in self.WorkList:
                for Inst in BB.instructions:
                    self.ResolveTypes(Inst, OpTypes)
            Changed = True
            Iteration = 0
            while Changed and Iteration < 5:
                Changed = False
                for BB in self.WorkList:
                    Changed |= self.ProcessBB(BB, OpTypes)
                Iteration += 1
            # rebuild canonical types after edits
            self.CanonicalType = self.build_canonical_types(OpTypes)

        # 5) Attach final TypeDesc to operands based on canonical types or resolved sets
        for BB in self.WorkList:
            for Inst in BB.instructions:
                for op in Inst.operands:
                    op.TypeDesc = self.GetTypeDesc(op, OpTypes)

        # Debug print (optional)
        for BB in self.WorkList:
            for Inst in BB.instructions:
                descs = []
                for op in Inst.operands:
                    descs.append(op.TypeDesc)
                print(f"{Inst} => {', '.join(descs)}")

        print("done")

    # --------------------
    # Core utilities
    # --------------------

    def GetOptype(self, OpTypes, Operand):
        if Operand in OpTypes:
            return OpTypes[Operand]
        elif Operand.Reg in OpTypes:
            return OpTypes[Operand.Reg]
        else:
            return TOP

    def SetOpType(self, OpTypes, Operand, Type, Inst):
        # Soft meet: if empty intersection, use union to keep both possibilities.
        prev = OpTypes.get(Operand.Reg, TOP)

        # Predicates always BOOL
        if Operand.IsPredicateReg or Operand.IsPT:
            Type = BOOL

        inter = prev & Type
        if inter:
            if Operand.IsWritableReg:
                OpTypes[Operand.Reg] = inter
            else:
                OpTypes[Operand] = inter
            return

        # Empty meet: do NOT edit IR here. Keep union to retain information.
        union = prev | Type
        if Operand.IsWritableReg:
            OpTypes[Operand.Reg] = union
        else:
            OpTypes[Operand] = union

    def ResolveTypes(self, Inst, OpTypes):
        opcode = Inst.opcodes[0]
        typeTable = self.instructionTypeTable.get(opcode, None)
        if not typeTable:
            # Unknown opcode -> TOP for everything
            for op in Inst.operands:
                self.SetOpType(OpTypes, op, TOP, Inst)
            return

        # Base types
        def_types, use_types = typeTable

        # Collect base mapping
        # Defs
        defs = Inst.GetDefs()
        for i, defOp in enumerate(defs):
            if i < len(def_types):
                t = def_types[i]
            else:
                t = TOP
            self.SetOpType(OpTypes, defOp, t, Inst)

        # Uses
        uses = Inst.GetUses()
        for i, useOp in enumerate(uses):
            if i < len(use_types):
                t = use_types[i]
            else:
                t = TOP
            self.SetOpType(OpTypes, useOp, t, Inst)

        # Apply modifier overrides
        modifierTable = self.modifierOverrideTable.get(opcode, None)
        if modifierTable:
            for mod in modifierTable:
                if mod not in Inst.opcodes[1:]:
                    continue
                def_overrides, use_overrides = modifierTable[mod]
                for i, defOp in enumerate(defs):
                    if i < len(def_overrides):
                        self.SetOpType(OpTypes, defOp, def_overrides[i], Inst)
                for i, useOp in enumerate(uses):
                    if i < len(use_overrides):
                        self.SetOpType(OpTypes, useOp, use_overrides[i], Inst)

    def ProcessBB(self, BB, OpTypes):
        old = dict(OpTypes)
        for Inst in BB.instructions:
            self.PropagateTypes(Inst, OpTypes)
        return old != OpTypes

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

        for propKey, propOpsList in propOps.items():
            propType = TOP
            for propOp in propOpsList:
                opType = self.GetOptype(OpTypes, propOp)
                if propType & opType:
                    propType = propType & opType
            for propOp in propOpsList:
                self.SetOpType(OpTypes, propOp, propType, Inst)

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

    # --------------------
    # Canonicalization and reconciliation
    # --------------------

    def build_canonical_types(self, OpTypes):
        Canon = {}
        for BB in self.WorkList:
            for Inst in BB.instructions:
                for defOp in Inst.GetDefs():
                    reg = defOp.Reg
                    tset = self.GetOptype(OpTypes, defOp)
                    Canon[reg] = self.choose_canonical_type(tset)
        return Canon

    def choose_canonical_type(self, tset):
        # Given a set of possible types, pick one single-type set deterministically.
        # Assumption: only same-width conflicts matter here.
        # Prefer exact leaf if singleton.
        if tset == INT32: return INT32
        if tset == FLOAT32: return FLOAT32
        if tset == INT64: return INT64
        if tset == FLOAT64: return FLOAT64
        if tset == BOOL: return BOOL
        if tset == HALF2: return HALF2

        # If it's a superset, pick a stable representative by width
        # For 32-bit family:
        if tset & NUM32:
            # Prefer Int32 over Float32 over Half2 over Bool to keep control ops simpler
            if tset & INT32: return INT32
            if tset & FLOAT32: return FLOAT32
            if tset & HALF2: return HALF2
            if tset & BOOL: return BOOL
            return INT32  # fallback
        # For 64-bit family:
        if tset & NUM64:
            if tset & INT64: return INT64
            if tset & FLOAT64: return FLOAT64
            return INT64  # fallback

        # Fallback to INT32 if truly TOP or unknown. Assumption keeps this safe.
        return INT32

    def _required_types_for_use(self, inst, use_index):
        opcode = inst.opcodes[0]
        ttab = self.instructionTypeTable.get(opcode, None)
        if not ttab:
            return TOP
        uses = ttab[1]
        req = TOP
        if use_index < len(uses):
            req = uses[use_index]
        # Apply modifiers
        mods = self.modifierOverrideTable.get(opcode, None)
        if mods:
            for mod, (def_over, use_over) in mods.items():
                if mod in inst.opcodes[1:]:
                    if use_index < len(use_over):
                        req = use_over[use_index]
        return req

    def _same_width_assumed(self, tsetA, tsetB):
        # Under assumption: conflicts only among same width.
        # We treat NUM32-family vs NUM32-family or NUM64 vs NUM64 as same width.
        has32A = bool(tsetA & NUM32) or bool(tsetA & NUM1)
        has64A = bool(tsetA & NUM64)
        has32B = bool(tsetB & NUM32) or bool(tsetB & NUM1)
        has64B = bool(tsetB & NUM64)

        # If either is TOP, assume same width OK (we won't widen/narrow).
        if tsetA == TOP or tsetB == TOP:
            return True

        if has32A and has64A:
            # Ambiguous; assume OK per problem statement (we won't see such conflicts)
            return True
        if has32B and has64B:
            return True

        # Normal check
        if has32A and has32B:
            return True
        if has64A and has64B:
            return True
        # Else, this would be a width conflict (not expected as per assumption)
        return False

    def reconcile_use_site_casts(self, function, OpTypes):
        """
        For each use, if producer's canonical type doesn't satisfy required type set,
        insert a BITCAST at the use (or on the PHI edge).
        Assumption: conflicts are same width (so BITCAST is sufficient).
        """
        changed = False

        def name_new(reg, tag):
            return f"{reg}_{tag}"

        def insert_before(bb, inst, newinst):
            idx = bb.instructions.index(inst)
            bb.instructions.insert(idx, newinst)

        def ensure_kind_at_use(bb, inst, use_operand, need_set):
            # Producer canonical type
            prod_reg = use_operand.Reg
            have_set = self.CanonicalType.get(prod_reg, self.GetOptype(OpTypes, use_operand))
            # If compatible, nothing to do
            if have_set & need_set:
                return False

            # Sanity under assumption: same width only
            if not self._same_width_assumed(have_set, need_set):
                print(f"Warning: unexpected width mismatch between {have_set} and {need_set} at {inst}. Skipping cast.")
                return False

            # Insert BITCAST before 'inst'
            new_reg = name_new(prod_reg, f"bitcast_{inst.id}")
            dst_op = Operand.fromReg(new_reg, new_reg)
            src_op = Operand.fromReg(prod_reg, prod_reg)
            bc = Instruction(
                id=f"{inst.id}_kindcast",
                opcodes=["BITCAST"],
                operands=[dst_op, src_op],
                parentBB=bb
            )
            insert_before(bb, inst, bc)
            # Patch the use to new reg
            use_operand.SetReg(new_reg)
            # Seed new type
            OpTypes[new_reg] = need_set
            # Canonicalize new reg to exactly the required type
            self.CanonicalType[new_reg] = self.choose_canonical_type(need_set)
            return True

        def ensure_kind_on_phi_edge(phi_inst, use_index, use_operand, need_set):
            prod_reg = use_operand.Reg
            have_set = self.CanonicalType.get(prod_reg, self.GetOptype(OpTypes, use_operand))
            if have_set & need_set:
                return False
            if not self._same_width_assumed(have_set, need_set):
                print(f"Warning: unexpected width mismatch between {have_set} and {need_set} on PHI {phi_inst}. Skipping cast.")
                return False

            # Insert in predecessor (before terminator)
            phi_bb = phi_inst.parentBB
            pred_bb = phi_bb._preds[use_index]
            term_idx = len(pred_bb.instructions) - 1
            if term_idx < 0:
                term_idx = 0
            new_reg = name_new(prod_reg, f"edgebc_{phi_bb.id}_{phi_inst.id}")
            dst_op = Operand.fromReg(new_reg, new_reg)
            src_op = Operand.fromReg(prod_reg, prod_reg)
            bc = Instruction(
                id=f"{phi_inst.id}_edge_kindcast",
                opcodes=["BITCAST"],
                operands=[dst_op, src_op],
                parentBB=pred_bb
            )
            pred_bb.instructions.insert(term_idx, bc)
            # Patch PHI incoming
            phi_inst.GetUses()[use_index].SetReg(new_reg)
            # Seed new types
            OpTypes[new_reg] = need_set
            self.CanonicalType[new_reg] = self.choose_canonical_type(need_set)
            return True

        # Apply reconciliation
        for bb in function.blocks:
            # Copy because we may insert before 'inst'
            for inst in list(bb.instructions):
                uses = inst.GetUses()
                for i, u in enumerate(uses):
                    req = self._required_types_for_use(inst, i)
                    # No requirement
                    if req == TOP:
                        continue

                    # For predicates or PT, force BOOL
                    if u.IsPredicateReg or u.IsPT:
                        req = BOOL

                    # If this is a PHI, handle on the edge
                    if inst.opcodes[0] == "PHI":
                        changed |= ensure_kind_on_phi_edge(inst, i, u, req)
                    else:
                        changed |= ensure_kind_at_use(bb, inst, u, req)

        if changed:
            # IR changed; rebuild def-use elsewhere
            return True
        return False

    def GetTypeDesc(self, Operand, OpTypes):
        # Prefer canonical type for registers
        if Operand.Reg in self.CanonicalType:
            # return a single type string
            cset = self.CanonicalType[Operand.Reg]
            if cset & INT32: return list(INT32)[0]
            if cset & FLOAT32: return list(FLOAT32)[0]
            if cset & INT64: return list(INT64)[0]
            if cset & FLOAT64: return list(FLOAT64)[0]
            if cset & BOOL: return list(BOOL)[0]
            if cset & HALF2: return list(HALF2)[0]
            # fallback
            return list(cset)[0]

        # Otherwise, derive from OpTypes
        if Operand in OpTypes:
            types = OpTypes[Operand]
        elif Operand.Reg in OpTypes:
            types = OpTypes[Operand.Reg]
        else:
            # Fallback: assume 32-bit int
            return list(INT32)[0]

        # Choose a stable representative
        if types & INT32: return list(INT32)[0]
        if types & FLOAT32: return list(FLOAT32)[0]
        if types & INT64: return list(INT64)[0]
        if types & FLOAT64: return list(FLOAT64)[0]
        if types & BOOL: return list(BOOL)[0]
        if types & HALF2: return list(HALF2)[0]
        # fallback
        return list(types)[0]
