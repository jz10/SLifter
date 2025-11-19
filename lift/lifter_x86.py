from llvmlite import ir, binding
import re


from transform.cg_patterns import CGPatterns
from transform.pack64 import Pack64
from transform.sr_substitute import SR_TO_OFFSET, SRSubstitute
from transform.opmod_transform import OpModTransform
from transform.defuse_analysis import DefUseAnalysis
from transform.ssa import SSA
from transform.reg_remap import RegRemap
from transform.xmad_to_imad import XmadToImad
from transform.fp_hack import FPHack
from transform.set_zero import SetZero
from transform.dce import DCE
from transform.mov_eliminate import MovEliminate
from transform.opaggregate import OperAggregate
from transform.typeanalysis import TypeAnalysis
from lift.lifter import Lifter


class UnsupportedOperatorException(Exception):
    pass


class UnsupportedInstructionException(Exception):
    pass


class InvalidTypeException(Exception):
    pass

class X86Lifter(Lifter):

    def get_transform_passes(self):
        return [
            CGPatterns(),
            Pack64(),
            SRSubstitute(),
            OpModTransform(),
            DefUseAnalysis(),
            SSA(),
            RegRemap(),
            DefUseAnalysis(),
            XmadToImad(),
            FPHack(),
            RegRemap(),
            SetZero(),
            DefUseAnalysis(),
            MovEliminate(),
            DefUseAnalysis(),
            DCE(),
            DefUseAnalysis(),
            OperAggregate(),
            RegRemap(),
            DefUseAnalysis(),
            DCE(),
            DefUseAnalysis(),
            TypeAnalysis(),
            DefUseAnalysis(),
            RegRemap(),
        ]

    def GetCmpOp(self, Opcode):
        if Opcode == "GE" or Opcode == "GEU":
            return ">="
        elif Opcode == "EQ":
            return "=="
        elif Opcode == "NE" or Opcode == "NEU":
            return "!="
        elif Opcode == "LE" or Opcode == "LEU":
            return "<="
        elif Opcode == "GT" or Opcode == "GTU":
            return ">"
        elif Opcode == "LT" or Opcode == "LTU":
            return "<"

        return ""
        
            
    def AddIntrinsics(self, llvm_module):
        # # Create thread idx function
        # FuncTy = self.ir.FunctionType(self.ir.IntType(32), [])
        
        # # Thread index function
        # FuncName = "thread_idx"
        # self.GetThreadIdx = self.ir.Function(llvm_module, FuncTy, FuncName)
        
        # # Block dimension function  
        # FuncName = "block_dim"
        # self.GetBlockDim = self.ir.Function(llvm_module, FuncTy, FuncName)
        
        # # Block index function
        # FuncName = "block_idx" 
        # self.GetBlockIdx = self.ir.Function(llvm_module, FuncTy, FuncName)
        
        # # Lane ID function
        # FuncName = "lane_id"
        # self.GetLaneId = self.ir.Function(llvm_module, FuncTy, FuncName)
        
        # # Warp ID function
        # FuncName = "warp_id"
        # self.GetWarpId = self.ir.Function(llvm_module, FuncTy, FuncName)

        # Constant memory
        BankTy =  self.ir.ArrayType(self.ir.IntType(8), 4096)
        ConstArrayTy =  self.ir.ArrayType(BankTy, 5)
        self.ConstMem = self.ir.GlobalVariable(llvm_module, ConstArrayTy, "const_mem")
        SharedArrayTy =  self.ir.ArrayType(self.ir.IntType(8), 32768)
        self.SharedMem = self.ir.GlobalVariable(llvm_module, SharedArrayTy, "shared_mem")
        LocalArrayTy =  self.ir.ArrayType(self.ir.IntType(8), 32768)
        self.LocalMem = self.ir.GlobalVariable(llvm_module, LocalArrayTy, "local_mem")


        # Runtime functions
        self.DeviceFuncs = {}

        # sync threads function
        FuncTy = self.ir.FunctionType(self.ir.VoidType(), [])
        SyncThreads = self.ir.Function(llvm_module, FuncTy, "syncthreads")
        self.DeviceFuncs["syncthreads"] = SyncThreads

        # leader thread function(if (threadIdx.x == 0) *ptr = val)
        FuncTy = self.ir.FunctionType(self.ir.IntType(1), [self.ir.PointerType(self.ir.IntType(32)), self.ir.IntType(32)], False)
        LeaderStore = self.ir.Function(llvm_module, FuncTy, "LeaderStore")
        self.DeviceFuncs["LeaderStore"] = LeaderStore

        # absolute function
        FuncTy = self.ir.FunctionType(self.ir.IntType(32), [self.ir.IntType(32)], False)
        AbsFunc = self.ir.Function(llvm_module, FuncTy, "abs")
        self.DeviceFuncs["abs"] = AbsFunc

    def _load_const_mem_u32(self, builder, offset, tag):
        ptr = builder.gep(
            self.ConstMem,
            [
                self.ir.Constant(self.ir.IntType(64), 0),
                self.ir.Constant(self.ir.IntType(64), 0),
                self.ir.Constant(self.ir.IntType(64), offset),
            ],
            f"{tag}_ptr",
        )
        ptr = builder.bitcast(ptr, self.ir.PointerType(self.ir.IntType(32)))
        return builder.load(ptr, tag)

    def _load_lane_id_value(self, builder):
        offset = SR_TO_OFFSET.get("SR_LANEID")
        if offset is None:
            return self.ir.Constant(self.ir.IntType(32), 0)
        return self._load_const_mem_u32(builder, offset, "laneid")

    def _emit_lane_mask(self, builder, kind, tag):
        lane_i32 = self._load_lane_id_value(builder)
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        lane_i64 = builder.zext(lane_i32, i64, f"{tag}_lane")
        one64 = self.ir.Constant(i64, 1)
        all_ones64 = self.ir.Constant(i64, 0xFFFFFFFFFFFFFFFF)

        if kind == "eq":
            mask64 = builder.shl(one64, lane_i64, f"{tag}_shift")
        elif kind == "le":
            lane_inc = builder.add(lane_i64, one64, f"{tag}_inc")
            mask64 = builder.sub(builder.shl(one64, lane_inc, f"{tag}_shift"), one64, f"{tag}_sub")
        elif kind == "lt":
            mask64 = builder.sub(builder.shl(one64, lane_i64, f"{tag}_shift"), one64, f"{tag}_sub")
        elif kind == "ge":
            mask64 = builder.shl(all_ones64, lane_i64, f"{tag}_shift")
        elif kind == "gt":
            lane_inc = builder.add(lane_i64, one64, f"{tag}_inc")
            mask64 = builder.shl(all_ones64, lane_inc, f"{tag}_shift")
        else:
            mask64 = self.ir.Constant(i64, 0)

        return builder.trunc(mask64, i32, tag)

    def _load_const_mem_u32(self, builder, offset, tag):
        ptr = builder.gep(
            self.ConstMem,
            [
                self.ir.Constant(self.ir.IntType(64), 0),
                self.ir.Constant(self.ir.IntType(64), 0),
                self.ir.Constant(self.ir.IntType(64), offset),
            ],
            f"{tag}_ptr",
        )
        ptr = builder.bitcast(ptr, self.ir.PointerType(self.ir.IntType(32)))
        return builder.load(ptr, tag)

    def _load_lane_id_value(self, builder):
        offset = SR_TO_OFFSET.get("SR_LANEID")
        if offset is None:
            return self.ir.Constant(self.ir.IntType(32), 0)
        return self._load_const_mem_u32(builder, offset, "laneid")

    def _emit_lane_mask(self, builder, kind, tag):
        lane_i32 = self._load_lane_id_value(builder)
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        lane_i64 = builder.zext(lane_i32, i64, f"{tag}_lane")
        one64 = self.ir.Constant(i64, 1)
        all_ones64 = self.ir.Constant(i64, 0xFFFFFFFFFFFFFFFF)

        if kind == "eq":
            mask64 = builder.shl(one64, lane_i64, f"{tag}_shift")
        elif kind == "le":
            lane_inc = builder.add(lane_i64, one64, f"{tag}_inc")
            mask64 = builder.sub(builder.shl(one64, lane_inc, f"{tag}_shift"), one64, f"{tag}_sub")
        elif kind == "lt":
            mask64 = builder.sub(builder.shl(one64, lane_i64, f"{tag}_shift"), one64, f"{tag}_sub")
        elif kind == "ge":
            mask64 = builder.shl(all_ones64, lane_i64, f"{tag}_shift")
        elif kind == "gt":
            lane_inc = builder.add(lane_i64, one64, f"{tag}_inc")
            mask64 = builder.shl(all_ones64, lane_inc, f"{tag}_shift")
        else:
            mask64 = self.ir.Constant(i64, 0)

        return builder.trunc(mask64, i32, tag)

    def _lift_function(self, func, llvm_module):
        args = func.GetArgs(self)

        func_ty = self.ir.FunctionType(self.ir.VoidType(), [])
        ir_function = self.ir.Function(llvm_module, func_ty, func.name)

        func.BlockMap = {}
        func.BuildBBToIRMap(ir_function, func.BlockMap)

        const_mem = {}
        entry_block = func.BlockMap[func.blocks[0]]
        builder = self.ir.IRBuilder(entry_block)
        builder.position_at_start(entry_block)

        offset_to_sr = {v: k for k, v in SR_TO_OFFSET.items()}

        for entry in args:
            addr = builder.gep(
                self.ConstMem,
                [
                    self.ir.Constant(self.ir.IntType(64), 0),
                    self.ir.Constant(self.ir.IntType(64), 0),
                    self.ir.Constant(self.ir.IntType(64), entry.ArgOffset),
                ],
            )
            name = offset_to_sr.get(entry.ArgOffset, entry.GetIRName(self))
            addr = builder.bitcast(
                addr, self.ir.PointerType(self.GetIRType(entry.TypeDesc))
            )
            val = builder.load(addr, name)
            const_mem[entry.GetIRName(self)] = val

        ir_regs = {}

        for bb in func.blocks:
            ir_block = func.BlockMap[bb]
            builder = self.ir.IRBuilder(ir_block)
            self._lift_basic_block(bb, builder, ir_regs, func.BlockMap, const_mem)

        for bb in func.blocks:
            builder = self.ir.IRBuilder(func.BlockMap[bb])
            self._populate_phi_nodes(bb, builder, ir_regs, func.BlockMap)

    def _lift_basic_block(self, bb, builder, ir_regs, block_map, const_mem):
        for inst in bb.instructions:
            self.lift_instruction(inst, builder, ir_regs, const_mem, block_map)

    def _populate_phi_nodes(self, bb, builder, ir_regs, block_map):
        ir_block = block_map[bb]

        def rough_search(op):
            reg = op.Reg
            name = op.GetIRName(self)

            best_key = max(ir_regs.keys(), key=lambda k: (k.startswith(reg), len(k)))

            val = builder.bitcast(
                ir_regs[best_key], op.GetIRType(self), f"{name}_cast"
            )
            return val

        for idx, inst in enumerate(bb.instructions):
            if inst.opcodes[0] not in {"PHI", "PHI64"}:
                continue

            ir_inst = ir_block.instructions[idx]

            for pred_idx, op in enumerate(inst._operands[1:]):
                pred_bb = bb._preds[pred_idx]
                if op.IsRZ:
                    val = self.ir.Constant(op.GetIRType(self), 0)
                elif op.IsPT:
                    val = self.ir.Constant(op.GetIRType(self), 1)
                else:
                    ir_name = op.GetIRName(self)
                    # if ir_name not in ir_regs:
                    #     val = rough_search(op)
                    # else:
                    val = ir_regs[ir_name]
                ir_inst.add_incoming(val, block_map[pred_bb])


    def lift_instruction(self, Inst, IRBuilder: ir.IRBuilder, IRRegs, ConstMem, BlockMap):
        if len(Inst._opcodes) == 0:
            raise UnsupportedInstructionException("Empty opcode list")
        opcode = Inst._opcodes[0]

        def roughSearch(op):
            reg = op.Reg
            name = op.GetIRName(self)
            targetType = name.replace(reg, "")

            bestKey = max(IRRegs.keys(), key=lambda k: (k.startswith(reg), len(k)))

            val = IRBuilder.bitcast(IRRegs[bestKey], op.GetIRType(self), f"{name}_cast")

            return val

        def _get_val(op, name=""):
            if op.IsRZ:
                return self.ir.Constant(op.GetIRType(self), 0)
            if op.IsPT:
                return self.ir.Constant(op.GetIRType(self), not op.IsNotReg)
            if op.IsReg:
                irName = op.GetIRName(self)
                val = IRRegs[irName]
                    
                if op.IsNegativeReg:
                    if op.GetTypeDesc().startswith('F'):
                        val = IRBuilder.fneg(val, f"{name}_fneg")
                    else:
                        val = IRBuilder.neg(val, f"{name}_neg")
                if op.IsNotReg:
                    val = IRBuilder.not_(val, f"{name}_not")
                if op.IsAbsReg:
                    val = IRBuilder.call(self.DeviceFuncs["abs"], [val], f"{name}_abs")
                return val
            if op.IsArg:
                    return ConstMem[op.GetIRName(self)]
            if op.IsImmediate:
                return self.ir.Constant(op.GetIRType(self), op.ImmediateValue)
            raise UnsupportedInstructionException(f"Unsupported operand type: {op}")

        def _as_pointer(addr_val, pointee_ty, name):
            if getattr(addr_val.type, "is_pointer", False):
                return addr_val

            if not hasattr(addr_val.type, "width"):
                raise InvalidTypeException(f"Expected integer address for {name}, got {addr_val.type}")

            if addr_val.type.width != 64:
                raise InvalidTypeException(f"Expected 64-bit address for {name}, got {addr_val.type.width}-bit value")

            ptr_ty = self.ir.PointerType(pointee_ty)
            return IRBuilder.inttoptr(addr_val, ptr_ty, name)

        if opcode == "MOV" or opcode == "MOV64" or opcode == "UMOV":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            val = _get_val(src, "mov")
            IRRegs[dest.GetIRName(self)] = val
            
        elif opcode == "S2UR":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            val = _get_val(src, "s2ur")
            IRRegs[dest.GetIRName(self)] = val

        elif opcode == "MOV32I":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            if not src.IsImmediate:
                raise UnsupportedInstructionException(f"MOV32I expects immediate, got: {src}")
            val = self.ir.Constant(src.GetIRType(self), src.ImmediateValue)
            IRRegs[dest.GetIRName(self)] = val

        elif opcode == "SETZERO":
            dest = Inst.GetDefs()[0]
            zero_val = self.ir.Constant(dest.GetIRType(self), 0)
            IRRegs[dest.GetIRName(self)] = zero_val
            
        elif opcode == "IMAD" or opcode == "UIMAD":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "imad_lhs")
            v2 = _get_val(uses[1], "imad_rhs")
            v3 = _get_val(uses[2], "imad_addend")


            tmp = IRBuilder.mul(v1, v2, "imad_tmp")
            tmp = IRBuilder.add(tmp, v3, "imad")
            IRRegs[dest.GetIRName(self)] = tmp

        elif opcode == "IMAD64":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "imad_lhs")
            v1_64 = IRBuilder.sext(v1, self.ir.IntType(64), "imad_lhs_64")
            v2 = _get_val(uses[1], "imad_rhs")
            v2_64 = IRBuilder.sext(v2, self.ir.IntType(64), "imad_rhs_64")
            v3 = _get_val(uses[2], "imad_addend")


            tmp = IRBuilder.mul(v1_64, v2_64, "imad_tmp")
            tmp = IRBuilder.add(tmp, v3, "imad")
            IRRegs[dest.GetIRName(self)] = tmp
            
        elif opcode == "EXIT":
            IRBuilder.ret_void()

        elif opcode == "FADD":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "fadd_lhs")
            v2 = _get_val(uses[1], "fadd_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.fadd(v1, v2, "fadd")

        elif opcode == "FFMA":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "ffma_lhs")
            v2 = _get_val(uses[1], "ffma_rhs")
            v3 = _get_val(uses[2], "ffma_addend")
            tmp = IRBuilder.fmul(v1, v2, "ffma_tmp")
            tmp = IRBuilder.fadd(tmp, v3, "ffma")
            IRRegs[dest.GetIRName(self)] = tmp
            
        elif opcode == "FMUL":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "fmul_lhs")
            v2 = _get_val(uses[1], "fmul_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.fmul(v1, v2, "fmul")
            
        elif opcode == "FMNMX":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "fmnmx_lhs")
            v2 = _get_val(uses[1], "fmnmx_rhs")
            pred = Inst.GetUses()[2]
            if pred.IsPT:
                r = IRBuilder.fcmp_ordered(">", v1, v2, "fmnmx")
                IRRegs[dest.GetIRName(self)] = IRBuilder.select(r, v1, v2, "fmnmx_select")
            else:
                r = IRBuilder.fcmp_ordered("<", v1, v2, "fmnmx")
                IRRegs[dest.GetIRName(self)] = IRBuilder.select(r, v1, v2, "fmnmx_select")
            
        elif opcode == "FCHK":
            # Checks for various special values, nan, inf, etc
            # Create a device function for it
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "fchk_val")
            chk_type = uses[1]
            
            func_name = "fchk"
            if func_name not in self.DeviceFuncs:
                func_ty = ir.FunctionType(ir.IntType(1), [ir.FloatType(), ir.IntType(32)], False)
                self.DeviceFuncs[func_name] = ir.Function(IRBuilder.module, func_ty, func_name)
                
            IRRegs[dest.GetIRName(self)] = IRBuilder.call(self.DeviceFuncs[func_name], [v1, _get_val(chk_type, "fchk_type")], "fchk_call")
            
        elif opcode == "DADD":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "dadd_lhs")
            v2 = _get_val(uses[1], "dadd_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.fadd(v1, v2, "dadd")
            
        elif opcode == "DFMA":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "dfma_lhs")
            v2 = _get_val(uses[1], "dfma_rhs")
            v3 = _get_val(uses[2], "dfma_addend")
            tmp = IRBuilder.fmul(v1, v2, "dfma_tmp")
            tmp = IRBuilder.fadd(tmp, v3, "dfma")
            IRRegs[dest.GetIRName(self)] = tmp
            
        elif opcode == "DMUL":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "dmul_lhs")
            v2 = _get_val(uses[1], "dmul_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.fmul(v1, v2, "dmul")

        elif opcode == "ISCADD":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "iscadd_lhs")
            v2 = _get_val(uses[1], "iscadd_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.add(v1, v2, "iscadd")

        elif opcode == "IADD3" or opcode == "UIADD3" or opcode == "IADD364":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "iadd3_o1")
            v2 = _get_val(uses[1], "iadd3_o2")
            v3 = _get_val(uses[2], "iadd3_o3")
            tmp = IRBuilder.add(v1, v2, "iadd3_tmp")
            IRRegs[dest.GetIRName(self)] = IRBuilder.add(tmp, v3, "iadd3")

        elif opcode == "ISUB":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "isub_lhs")
            v2 = _get_val(uses[1], "isub_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.add(v1, v2, "sub")
                    
        elif opcode == "SHL" or opcode == "USHL":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "shl_lhs")
            v2 = _get_val(uses[1], "shl_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.shl(v1, v2, "shl")

        elif opcode == "SHL64":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "shl_lhs_64")
            v2 = _get_val(uses[1], "shl_rhs_64")
            IRRegs[dest.GetIRName(self)] = IRBuilder.shl(v1, v2, "shl")

        elif opcode == "SHR" or opcode == "SHR64" or opcode == "USHR":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "shr_lhs")
            v2 = _get_val(uses[1], "shr_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.lshr(v1, v2, "shr")

        elif opcode == "SHF" or opcode == "USHF":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "shf_lo")
            v2 = _get_val(uses[1], "shf_shift")
            v3 = _get_val(uses[2], "shf_hi")
            
            # concatentate lo and hi
            lo64 = IRBuilder.zext(v1, self.ir.IntType(64))
            hi64 = IRBuilder.zext(v3, self.ir.IntType(64))
            v = IRBuilder.or_(
                lo64,
                IRBuilder.shl(hi64, self.ir.Constant(self.ir.IntType(64), 32)),
                "shf_concat"
            )
            
            v2 = IRBuilder.zext(v2, self.ir.IntType(64), "shf_shift_64")

            
            right = Inst.opcodes[1] == "R"
            signed = "S" in Inst.opcodes[2]
            high = "HI" in Inst.opcodes
            if right:
                if signed:
                    r = IRBuilder.ashr(v, v2)
                else:
                    r = IRBuilder.lshr(v, v2)
            else:
                r = IRBuilder.shl(v, v2)
            
            if high:
                r = IRBuilder.lshr(r, self.ir.Constant(self.ir.IntType(64), 32), "shf_hi")
            IRRegs[dest.GetIRName(self)] = IRBuilder.trunc(r, dest.GetIRType(self), "shf_result")
                
        elif opcode == "IADD":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "iadd_lhs")
            v2 = _get_val(uses[1], "iadd_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.add(v1, v2, "iadd")
            
        elif opcode == "SEL" or opcode == "FSEL" or opcode == "USEL":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "sel_true")
            v2 = _get_val(uses[1], "sel_false")
            pred = _get_val(uses[2], "sel_pred")
            IRRegs[dest.GetIRName(self)] = IRBuilder.select(pred, v1, v2, "sel")
        
        elif opcode == "IADD64":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "iadd_lhs")
            v2 = _get_val(uses[1], "iadd_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.add(v1, v2, "iadd")

        elif opcode == "IADD32I" or opcode == "IADD32I64":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            op1, op2 = uses[0], uses[1]
            v1 = _get_val(op1, "iadd32i_lhs")
            
            # TODO: temporary fix
            def sx(v, n):
                v &= (1 << n) - 1
                return (v ^ (1 << (n-1))) - (1 << (n-1))
            op2.ImmediateValue = sx(int(op2.Name, 16), 24)
            v2 = _get_val(op2, "iadd32i_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.add(v1, v2, "iadd32i")

        elif opcode == "PHI" or opcode == "PHI64":
            dest = Inst.GetDefs()[0]
            phi_val = IRBuilder.phi(dest.GetIRType(self), "phi")

            # Some values may be unknown at this point
            # Don't add incoming values yet

            IRRegs[dest.GetIRName(self)] = phi_val

        elif opcode == "S2R":
            dest = Inst.GetDefs()[0]
            valop = Inst.GetUses()[0]
            offset = SR_TO_OFFSET.get(valop.Name)
            if offset is not None:
                val = self._load_const_mem_u32(IRBuilder, offset, f"sr_{offset:x}")
            elif valop.IsWarpSize:
                val = self.ir.Constant(self.ir.IntType(32), 32)
            elif valop.IsActiveMask:
                val = self.ir.Constant(self.ir.IntType(32), 0xFFFFFFFF)
            elif valop.IsLaneMaskEQ:
                val = self._emit_lane_mask(IRBuilder, "eq", "lanemask_eq")
            elif valop.IsLaneMaskLE:
                val = self._emit_lane_mask(IRBuilder, "le", "lanemask_le")
            elif valop.IsLaneMaskLT:
                val = self._emit_lane_mask(IRBuilder, "lt", "lanemask_lt")
            elif valop.IsLaneMaskGE:
                val = self._emit_lane_mask(IRBuilder, "ge", "lanemask_ge")
            elif valop.IsLaneMaskGT:
                val = self._emit_lane_mask(IRBuilder, "gt", "lanemask_gt")
            else:
                print(f"S2R: Unknown special register {valop.Name}")
                val = self.ir.Constant(self.ir.IntType(32), 0)
            IRRegs[dest.GetIRName(self)] = val
                
        elif opcode == "LDG" or opcode == "LDG64":
            dest = Inst.GetDefs()[0]
            ptr = Inst.GetUses()[0]
            addr = _get_val(ptr, "ldg_addr")
            pointee_ty = dest.GetIRType(self)
            addr_ptr = _as_pointer(addr, pointee_ty, f"{ptr.GetIRName(self)}_addr_ptr" if ptr.IsReg else "ldg_addr_ptr")
            val = IRBuilder.load(addr_ptr, "ldg", typ=pointee_ty)
            IRRegs[dest.GetIRName(self)] = val
                
        elif opcode == "STG":
            uses = Inst.GetUses()
            ptr, val = uses[0], uses[1]
            addr = _get_val(ptr, "stg_addr")
            v = _get_val(val, "stg_val")
            addr_ptr = _as_pointer(addr, v.type, f"{ptr.GetIRName(self)}_addr_ptr" if ptr.IsReg else "stg_addr_ptr")
            IRBuilder.store(v, addr_ptr)

        elif opcode == "LDS":
            dest = Inst.GetDefs()[0]
            ptr = Inst.GetUses()[0]
            addr = _get_val(ptr, "lds_addr")
            addr = IRBuilder.gep(self.SharedMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "lds_shared_addr")
            if addr.type.pointee != dest.GetIRType(self):
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(dest.GetIRType(self)),
                    "lds_shared_addr_cast",
                )
            val = IRBuilder.load(addr, "lds", typ=dest.GetIRType(self))
            IRRegs[dest.GetIRName(self)] = val

        elif opcode == "STS":
            uses = Inst.GetUses()
            ptr, val = uses[0], uses[1]
            addr = _get_val(ptr, "sts_addr")
            addr = IRBuilder.gep(self.SharedMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "sts_shared_addr")
            if addr.type.pointee != val.GetIRType(self):
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(val.GetIRType(self)),
                    "sts_shared_addr_cast",
                )
            v = _get_val(val, "sts_val")
            IRBuilder.store(v, addr)
            
        elif opcode == "LDL":
            dest = Inst.GetDefs()[0]
            ptr = Inst.GetUses()[0]
            addr = _get_val(ptr, "ldl_addr")
            addr = IRBuilder.gep(self.LocalMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "ldl_local_addr")
            val = IRBuilder.load(addr, "ldl", typ=dest.GetIRType(self))
            IRRegs[dest.GetIRName(self)] = val
            
        elif opcode == "STL":
            uses = Inst.GetUses()
            ptr, val = uses[0], uses[1]
            addr = _get_val(ptr, "stl_addr")
            addr = IRBuilder.gep(self.LocalMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "stl_local_addr")
            if addr.type.pointee != val.GetIRType(self):
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(val.GetIRType(self)),
                    "stl_local_addr_cast",
                )
            v = _get_val(val, "stl_val")
            IRBuilder.store(v, addr)

        elif opcode == "FMUL":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "fmul_lhs")
            v2 = _get_val(uses[1], "fmul_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.fmul(v1, v2, "fmul")

        elif opcode == "FFMA":
            raise UnsupportedInstructionException

        elif opcode == "LD":
            dest = Inst.GetDefs()[0]
            ptr = Inst.GetUses()[0]
            addr = _get_val(ptr, "ld_addr")
            pointee_ty = dest.GetIRType(self)
            addr_ptr = _as_pointer(addr, pointee_ty, f"{ptr.GetIRName(self)}_addr_ptr" if ptr.IsReg else "ld_addr_ptr")
            val = IRBuilder.load(addr_ptr, "ld", typ=pointee_ty)
            IRRegs[dest.GetIRName(self)] = val
                
        elif opcode == "ST":
            uses = Inst.GetUses()
            ptr, val = uses[0], uses[1]
            v = _get_val(val, "st_val")
            addr = _get_val(ptr, "st_addr")
            addr_ptr = _as_pointer(addr, v.type, f"{ptr.GetIRName(self)}_addr_ptr" if ptr.IsReg else "st_addr_ptr")
            IRBuilder.store(v, addr_ptr)
        
        elif opcode == "LOP" or opcode == "ULOP":
            dest = Inst.GetDefs()[0]
            a, b = Inst.GetUses()[0], Inst.GetUses()[1]
            subop = Inst._opcodes[1] if len(Inst._opcodes) > 1 else None
            vb = _get_val(b, "lop_b")

            if subop == "PASS_B":
                IRRegs[dest.GetIRName(self)] = vb
            else:
                raise UnsupportedInstructionException
        
        elif opcode == "LOP32I" or opcode == "ULOP32I":
            dest = Inst.GetDefs()[0]
            a, b = Inst.GetUses()[0], Inst.GetUses()[1]
            v1 = _get_val(a, "lop32i_a")
            v2 = _get_val(b, "lop32i_b")
            func = Inst._opcodes[1] if len(Inst._opcodes) > 1 else None

            if func == "AND":
                IRRegs[dest.GetIRName(self)] = IRBuilder.and_(v1, v2, "lop32i_and")
            else:
                raise UnsupportedInstructionException
        
        elif opcode == "BFE":
            raise UnsupportedInstructionException
        
        elif opcode == "BFI":
            raise UnsupportedInstructionException
        
        elif opcode == "SSY":
            pass

        elif opcode == "SYNC":
            pass
        
        
        elif opcode == "WARPSYNC":
            pass
        
        elif opcode == "BRA":
            targetId = Inst.GetUses()[-1].Name.zfill(4)
            for BB, IRBB in BlockMap.items():
                if int(BB.addr_content ,16) != int(targetId, 16):
                    continue
                targetBB = IRBB
            
            IRBuilder.branch(targetBB)
        
        elif opcode == "BRK":
            raise UnsupportedInstructionException
        
        elif opcode == "IMNMX":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "imnmx_lhs")
            v2 = _get_val(uses[1], "imnmx_rhs")

            isUnsigned = "U32" in Inst._opcodes
            isMax = "MXA" in Inst._opcodes

            if isUnsigned:
                if isMax:
                    cond = IRBuilder.icmp_unsigned('>', v1, v2, "imnmx_cmp")
                else:
                    cond = IRBuilder.icmp_unsigned('<', v1, v2, "imnmx_cmp")
            else:
                if isMax:
                    cond = IRBuilder.icmp_signed('>', v1, v2, "imnmx_cmp")
                else:
                    cond = IRBuilder.icmp_signed('<', v1, v2, "imnmx_cmp")

            IRRegs[dest.GetIRName(self)] = IRBuilder.select(cond, v1, v2, "imnmx_max")
                
        elif opcode == "PSETP" or opcode == "UPSETP64":
            raise UnsupportedInstructionException
        
        elif opcode == "PBK":
            raise UnsupportedInstructionException
             
        elif opcode == "LEA" or opcode == "LEA64" or opcode == "ULEA":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()

            v1 = _get_val(uses[0], "lea_a")
            v2 = _get_val(uses[1], "lea_b")
            v3 = _get_val(uses[2], "lea_scale")

            tmp = IRBuilder.shl(v1, v3, "lea_tmp")
            IRRegs[dest.GetIRName(self)] = IRBuilder.add(tmp, v2, "lea")
                    
        elif opcode == "F2I":
            dest = Inst.GetDefs()[0]
            op1 = Inst.GetUses()[0]

            isUnsigned = "U32" in Inst._opcodes

            v1 = _get_val(op1, "f2i_src")

            if isUnsigned:
                val = IRBuilder.fptoui(v1, dest.GetIRType(self), "f2i")
            else:
                val = IRBuilder.fptosi(v1, dest.GetIRType(self), "f2i")
            IRRegs[dest.GetIRName(self)] = val
                    
        elif opcode == "I2F":
            dest = Inst.GetDefs()[0]
            op1 = Inst.GetUses()[0]

            isUnsigned = "U32" in Inst._opcodes

            v1 = _get_val(op1, "i2f_src")

            if isUnsigned:
                val = IRBuilder.uitofp(v1, dest.GetIRType(self), "i2f")
            else:
                val = IRBuilder.sitofp(v1, dest.GetIRType(self), "i2f")
            IRRegs[dest.GetIRName(self)] = val
                    
        elif opcode == "MUFU":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            func = Inst._opcodes[1] if len(Inst._opcodes) > 1 else None
            v = _get_val(src, "mufu_src")

            if func == "RCP": # 1/v
                one = self.ir.Constant(dest.GetIRType(self), 1.0)
                res = IRBuilder.fdiv(one, v, "mufu_rcp")
                IRRegs[dest.GetIRName(self)] = res
            else:
                raise UnsupportedInstructionException
                    
        elif opcode == "IABS":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            v = _get_val(src, "iabs_src")
            res = IRBuilder.call(self.DeviceFuncs["abs"], [v], "iabs")
            IRRegs[dest.GetIRName(self)] = res

                    
        elif opcode == "LOP3" or opcode == "ULOP3" or opcode == "PLOP3" or opcode == "UPLOP3":
            
            if Inst.opcodes[1] != "LUT":
                raise UnsupportedInstructionException
            
            
            destPred = Inst.GetDefs()[0] if len(Inst.GetDefs()) > 1 else None
            destReg = Inst.GetDefs()[-1]
            src1, src2, src3 = Inst.GetUses()[0], Inst.GetUses()[1], Inst.GetUses()[2]
            
            lut = Inst.GetUses()[3]
            src4 = Inst.GetUses()[4]

            a = _get_val(src1, "lop3_a")
            b = _get_val(src2, "lop3_b")
            c = _get_val(src3, "lop3_c")
            q = _get_val(src4, "lop3_q")

            lut_val = lut.ImmediateValue
            
            na = IRBuilder.xor(a, self.ir.Constant(a.type, -1))
            nb = IRBuilder.xor(b, self.ir.Constant(b.type, -1))
            nc = IRBuilder.xor(c, self.ir.Constant(c.type, -1))
            
            zero = self.ir.Constant(destReg.GetIRType(self), 0)
            r = zero
            
            for idx in range(8):
                if (lut_val >> idx) & 1 == 0:
                    continue 

                # idx = (a_bit << 2) | (b_bit << 1) | c_bit
                a_bit = (idx >> 2) & 1
                b_bit = (idx >> 1) & 1
                c_bit = idx        & 1

                va = a  if a_bit else na
                vb = b  if b_bit else nb
                vc = c  if c_bit else nc

                t_ab   = IRBuilder.and_(va, vb)
                t_term = IRBuilder.and_(t_ab, vc)

                if r is zero:
                    r = t_term
                else:
                    r = IRBuilder.or_(r, t_term)

            if destReg.IsWritableReg:
                IRRegs[destReg.GetIRName(self)] = r
                
            if destPred and destPred.IsWritableReg:
                tmp = IRBuilder.icmp_signed('!=', r, zero, "lop3_pred_cmp")
                IRRegs[destPred.GetIRName(self)] = IRBuilder.or_(tmp, q, "lop3_p")

        elif opcode == "MOVM":
            # TODO: dummy implementation
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            val = _get_val(src, "movm")
            IRRegs[dest.GetIRName(self)] = val
                    
        elif opcode == "HMMA":
            
            size = Inst.opcodes[1]
            type = Inst.opcodes[2]

            if type != "F32":
                raise UnsupportedInstructionException
            
            if size != "1688":
                raise UnsupportedInstructionException

            if "hmma1688f32" not in self.DeviceFuncs:
                # hmma1688f32(float*8) -> float*4
                outputType = ir.LiteralStructType([ir.FloatType()] * 4)
                inputType = [ir.FloatType()] * 8
                func_ty = ir.FunctionType(outputType, inputType)
                self.DeviceFuncs["hmma1688f32"] = ir.Function(IRBuilder.module, func_ty, name="hmma1688f32")

            func = self.DeviceFuncs["hmma1688f32"]
            uses = Inst.GetUses()
            args = [_get_val(uses[i], f"hmma_arg_{i}") for i in range(8)]
            val = IRBuilder.call(func, args, "hmma_call")
            
            # # unpack into 4 dest registers
            # dests = Inst.GetDefs()
            # for i in range(4):
            #     IRRegs[dests[i].GetIRName(self)] = IRBuilder.extract_value(val, i, f"hmma_res_{i}")
        
        elif opcode == "DEPBAR":
            pass

        elif opcode == "ULDC" or opcode == "ULDC64" or opcode == "LDC":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            IRRegs[dest.GetIRName(self)] = _get_val(src, "ldc_const")
        
        elif opcode == "CS2R":
            # CS2R (Convert Special Register to Register)
            ResOp = Inst.GetDefs()[0]
            ValOp = Inst.GetUses()[0]
            if ResOp.IsReg:
                dest_name = ResOp.GetIRName(self)
                if ValOp.IsRZ:
                    IRVal = self.ir.Constant(self.ir.IntType(32), 0)
                else:
                    offset = SR_TO_OFFSET.get(ValOp.Name)
                    if offset is not None:
                        IRVal = self._load_const_mem_u32(IRBuilder, offset, f"cs2r_{offset:x}")
                    elif ValOp.IsWarpSize:
                        IRVal = self.ir.Constant(self.ir.IntType(32), 32)
                    elif ValOp.IsActiveMask:
                        IRVal = self.ir.Constant(self.ir.IntType(32), 0xFFFFFFFF)
                    elif ValOp.IsLaneMaskEQ:
                        IRVal = self._emit_lane_mask(IRBuilder, "eq", "cs2r_lanemask_eq")
                    elif ValOp.IsLaneMaskLE:
                        IRVal = self._emit_lane_mask(IRBuilder, "le", "cs2r_lanemask_le")
                    elif ValOp.IsLaneMaskLT:
                        IRVal = self._emit_lane_mask(IRBuilder, "lt", "cs2r_lanemask_lt")
                    elif ValOp.IsLaneMaskGE:
                        IRVal = self._emit_lane_mask(IRBuilder, "ge", "cs2r_lanemask_ge")
                    elif ValOp.IsLaneMaskGT:
                        IRVal = self._emit_lane_mask(IRBuilder, "gt", "cs2r_lanemask_gt")
                    else:
                        print(f"CS2R: Unknown special register {ValOp}")
                        IRVal = self.ir.Constant(self.ir.IntType(32), 0)

                if dest_name in IRRegs and getattr(IRRegs[dest_name].type, "is_pointer", False):
                    IRBuilder.store(IRVal, IRRegs[dest_name])
                else:
                    IRRegs[dest_name] = IRVal
                    
        elif opcode == "FSETP" or opcode == "DSETP":
            dest1 = Inst.GetDefs()[0]
            dest2 = Inst.GetDefs()[1]
            a, b = Inst.GetUses()[0], Inst.GetUses()[1]
            src = Inst.GetUses()[2]
            
            val1 = _get_val(a, "fsetrp_lhs")
            val2 = _get_val(b, "fsetrp_rhs")
            val3 = _get_val(src, "fsetrp_src")
            
            cmp1 = Inst.opcodes[1]
            cmp2 = Inst.opcodes[2]
            
            isUnordered = "U" in cmp1
            
            funcs_map = {
                "EQ": "==",
                "NE": "!=",
                "LT": "<",
                "LE": "<=",
                "GT": ">",
                "GE": ">=",
                "NEU": "!=",
                "LTU": "<",
                "LEU": "<=",
                "GTU": ">",
                "GEU": ">=",
            }
            
            cmp2_map = {
                "AND": IRBuilder.and_,
                "OR": IRBuilder.or_,
                "XOR": IRBuilder.xor
            }
            
            if isUnordered:
                r = IRBuilder.fcmp_unordered(funcs_map[cmp1], val1, val2, "fsetrp_cmp1")
            else:
                r = IRBuilder.fcmp_ordered(funcs_map[cmp1], val1, val2, "fsetrp_cmp1")
                
            cmp2_func = cmp2_map.get(cmp2)
            if not cmp2_func:
                raise UnsupportedInstructionException
            
            r1 = cmp2_func(r, val3, "fsetrp_cmp2")
            
            if not dest1.IsPT:
                IRRegs[dest1.GetIRName(self)] = r1
                
            if not dest2.IsPT:
                r2 = IRBuilder.not_(r1, "fsetrp_not")
                IRRegs[dest2.GetIRName(self)] = r2

        elif opcode == "ISETP" or opcode == "ISETP64" or opcode == "UISETP":
            dest1 = Inst.GetDefs()[0]
            dest2 = Inst.GetDefs()[1]
            a, b = Inst.GetUses()[0], Inst.GetUses()[1]
            src = Inst.GetUses()[2]
            
            val1 = _get_val(a, "isetrp_lhs")
            val2 = _get_val(b, "isetrp_rhs")
            val3 = _get_val(src, "isetrp_src")
            
            cmp1 = Inst.opcodes[1]
            if "U" in Inst.opcodes[2]:
                isUnsigned = True
                cmp2 = Inst.opcodes[3]
            else:
                isUnsigned = False
                cmp2 = Inst.opcodes[2]
            
            funcs_map = {
                "EQ": "==",
                "NE": "!=",
                "LT": "<",
                "LE": "<=",
                "GT": ">",
                "GE": ">=",
            }
            
            cmp2_map = {
                "AND": IRBuilder.and_,
                "OR": IRBuilder.or_,
                "XOR": IRBuilder.xor
            }
            
            if isUnsigned:
                r = IRBuilder.icmp_unsigned(funcs_map[cmp1], val1, val2, "isetrp_cmp1")
            else:
                r = IRBuilder.icmp_signed(funcs_map[cmp1], val1, val2, "isetrp_cmp1")
                
            cmp2_func = cmp2_map.get(cmp2)
            if not cmp2_func:
                raise UnsupportedInstructionException
            
            r1 = cmp2_func(r, val3, "isetrp_cmp2")
            
            if not dest1.IsPT:
                IRRegs[dest1.GetIRName(self)] = r1
                
            if not dest2.IsPT:
                r2 = IRBuilder.not_(r1, "isetrp_not")
                IRRegs[dest2.GetIRName(self)] = r2

        elif opcode == "PBRA":
            pred = Inst.GetUses()[0]

            cond = _get_val(pred, "cond")

            TrueBr, FalseBr = Inst.parent.GetBranchPair(Inst, BlockMap.keys())
            IRBuilder.cbranch(cond, BlockMap[TrueBr], BlockMap[FalseBr])

        elif opcode == "BAR":
            if len(Inst._opcodes) > 1 and Inst._opcodes[1] != "SYNC":
                raise UnsupportedInstructionException

            IRBuilder.call(self.DeviceFuncs["syncthreads"], [], "barrier")

        elif opcode == "PACK64":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0])
            v2 = _get_val(uses[1])
            lo64 = IRBuilder.zext(v1, ir.IntType(64), "pack64_lo")
            hi64 = IRBuilder.zext(v2, ir.IntType(64), "pack64_hi")
            hiShift = IRBuilder.shl(hi64, ir.Constant(ir.IntType(64), 32), "pack64_hi_shift")
            packed = IRBuilder.or_(lo64, hiShift, "pack64_result")
            IRRegs[dest.GetIRName(self)] = packed
            
        elif opcode == "UNPACK64":
            dests = Inst.GetDefs()
            src = Inst.GetUses()[0]
            val = _get_val(src)
            lo32 = IRBuilder.trunc(val, ir.IntType(32), "unpack64_lo")
            hi32_shift = IRBuilder.lshr(val, ir.Constant(ir.IntType(64), 32), "unpack64_hi_shift")
            hi32 = IRBuilder.trunc(hi32_shift, ir.IntType(32), "unpack64_hi")
            IRRegs[dests[0].GetIRName(self)] = lo32
            IRRegs[dests[1].GetIRName(self)] = hi32

        elif opcode == "CAST64":
            dest = Inst.GetDefs()[0]
            op1 = Inst.GetUses()[0]
            if not dest.IsReg:
                raise UnsupportedInstructionException(f"CAST64 expects a register operand, got: {dest}")

            val = _get_val(op1)

            val64 = IRBuilder.sext(val, ir.IntType(64), "cast64")
            IRRegs[dest.GetIRName(self)] = val64

        elif opcode == "BITCAST":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            dest_type = dest.GetIRType(self)

            val = _get_val(src)

            cast_val = IRBuilder.bitcast(val, dest_type, "cast")
            IRRegs[dest.GetIRName(self)] = cast_val
            
        elif opcode == "VOTE" or opcode == "VOTEU":
            dest = Inst.GetDefs()[0]
            pred_val = _get_val(Inst.GetUses()[0], "vote_pred")
            mask_val = _get_val(Inst.GetUses()[1], "vote_mask")

            if pred_val.type != ir.IntType(1):
                zero = ir.Constant(pred_val.type, 0)
                pred_val = IRBuilder.icmp_unsigned("!=", pred_val, zero, "vote_pred_i1")

            if mask_val.type != ir.IntType(32):
                if hasattr(mask_val.type, "width") and mask_val.type.width > 32:
                    mask_val = IRBuilder.trunc(mask_val, ir.IntType(32), "vote_mask_i32")
                else:
                    mask_val = IRBuilder.zext(mask_val, ir.IntType(32), "vote_mask_i32")

            mode = Inst.opcodes[1].upper()
            vote_ret_map = {
                "ANY": ir.IntType(1),
                "ALL": ir.IntType(1),
                "UNI": ir.IntType(1),
                "BALLOT": ir.IntType(32),
            }
            if mode not in vote_ret_map:
                raise UnsupportedInstructionException

            func_name = f"vote_{mode.lower()}"
            if func_name not in self.DeviceFuncs:
                func_ty = ir.FunctionType(vote_ret_map[mode], [ir.IntType(32), ir.IntType(1)], False)
                self.DeviceFuncs[func_name] = ir.Function(IRBuilder.module, func_ty, func_name)

            voteVal = IRBuilder.call(self.DeviceFuncs[func_name], [mask_val, pred_val], func_name)

            dest_ty = dest.GetIRType(self)
            vote_ty = voteVal.type
            if dest_ty != vote_ty:
                if hasattr(dest_ty, "width") and hasattr(vote_ty, "width"):
                    if dest_ty.width > vote_ty.width:
                        voteVal = IRBuilder.zext(voteVal, dest_ty, f"vote_{mode.lower()}_zext")
                    else:
                        voteVal = IRBuilder.trunc(voteVal, dest_ty, f"vote_{mode.lower()}_trunc")
                else:
                    voteVal = IRBuilder.bitcast(voteVal, dest_ty, f"vote_{mode.lower()}_cast")

            IRRegs[dest.GetIRName(self)] = voteVal
            
        elif opcode == "SHFL":
            destPred = Inst.GetDefs()[0]
            destReg = Inst.GetDefs()[1]
            srcReg = Inst.GetUses()[0]
            offset = Inst.GetUses()[1]
            width = Inst.GetUses()[2]
            
            val = _get_val(srcReg, "shfl_val")
            off = _get_val(offset, "shfl_offset")
            wid = _get_val(width, "shfl_width")
            
            mode = Inst.opcodes[1].upper()
            
            dtype = destReg.GetIRType(self)
            i32 = self.ir.IntType(32)
            mask_const = self.ir.Constant(i32, 0xFFFFFFFF)
            
            func_name = "warp_shfl_" + mode.lower() + "_" + dtype.intrinsic_name
            func_ty = ir.FunctionType(dtype, [i32, dtype, i32, i32], False)

            if func_name not in self.DeviceFuncs:
                self.DeviceFuncs[func_name] = ir.Function(IRBuilder.module, func_ty, func_name)

            shflVal = IRBuilder.call(
                self.DeviceFuncs[func_name],
                [mask_const, val, off, wid],
                func_name,
            )

            dest_type = destReg.GetIRType(self)

            IRRegs[destReg.GetIRName(self)] = shflVal
            if not destPred.IsPT:
                IRRegs[destPred.GetIRName(self)] = self.ir.Constant(self.ir.IntType(1), 1)

        elif opcode == "MATCH":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]

            val = _get_val(src)
            mode = Inst.opcodes[1]            
            type = Inst.opcodes[2]

            if mode != 'ANY':
                raise UnsupportedInstructionException

            if type == 'U64':
                dtype = ir.IntType(64)
            elif type == 'U32':
                dtype = ir.IntType(32)
            else:
                raise UnsupportedInstructionException

            funcName = f"match_{mode}_{type}"
            if funcName not in self.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [dtype], False)
                self.DeviceFuncs[funcName] = ir.Function(IRBuilder.module, FuncTy, funcName)

            matchVal = IRBuilder.call(self.DeviceFuncs[funcName], [val], "match")
            IRRegs[dest.GetIRName(self)] = matchVal

        elif opcode == "BREV" or opcode == "UBREV":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            val = _get_val(src)

            if "brev" not in self.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [ir.IntType(32)], False)
                self.DeviceFuncs["brev"] = ir.Function(IRBuilder.module, FuncTy, "brev")

            revVal = IRBuilder.call(self.DeviceFuncs["brev"], [val], "brev")
            IRRegs[dest.GetIRName(self)] = revVal

        elif opcode == "FLO" or opcode == "UFLO":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            val = _get_val(src)

            type = Inst.opcodes[1]
            mode = Inst.opcodes[2] if len(Inst.opcodes) > 2 else None

            if type == 'U64':
                dtype = ir.IntType(64)
                typeName = 'i64'
            elif type == 'U32':
                dtype = ir.IntType(32)
                typeName = 'i32'
            else:
                raise UnsupportedInstructionException

            if mode == 'SH' or mode == None:
                funcName = f"llvm.ctlz.{typeName}"     
            else:
                raise UnsupportedInstructionException

            if funcName not in self.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [dtype], False)
                self.DeviceFuncs[funcName] = ir.Function(IRBuilder.module, FuncTy, funcName)

            floVal = IRBuilder.call(self.DeviceFuncs[funcName], [val], "flo")
            IRRegs[dest.GetIRName(self)] = floVal

        elif opcode == "POPC" or opcode == "UPOPC":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            val = _get_val(src)

            funcName = f"llvm.ctpop.i32"     

            if funcName not in self.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [ir.IntType(32)], False)
                self.DeviceFuncs[funcName] = ir.Function(IRBuilder.module, FuncTy, funcName)

            popcVal = IRBuilder.call(self.DeviceFuncs[funcName], [val], "popc")
            IRRegs[dest.GetIRName(self)] = popcVal

        elif opcode == "RED":
            # TODO: incorrect impl
            uses = Inst.GetUses()
            src1, src2 = uses[0], uses[1]
            val1 = _get_val(src1)
            val2 = _get_val(src2)
            mode = Inst.opcodes[2]
            order = 'seq_cst'
            IRBuilder.atomic_rmw(mode, val1, val2, order)

        elif opcode == "ATOM":
            modifiers = [tok.upper() for tok in Inst.opcodes[1:]]
            if len(Inst.opcodes) < 4 or Inst.opcodes[2].upper() != "ADD":
                raise UnsupportedInstructionException("ATOM currently only supports ADD")
            type_token = Inst.opcodes[3].upper()
            if not type_token.startswith("F"):
                raise UnsupportedInstructionException(
                    "ATOM expects the fourth modifier to describe an F-type"
                )
            if "STRONG" not in modifiers:
                raise UnsupportedInstructionException("ATOM requires STRONG memory scope")

            uses = Inst.GetUses()
            if not uses:
                raise UnsupportedInstructionException("ATOM requires operands")

            dest_reg = None
            if uses[0].IsReg and not uses[0].IsMemAddr:
                dest_reg = uses[0]

            mem_ops = [op for op in uses if op.IsMemAddr]
            if len(mem_ops) != 1:
                raise UnsupportedInstructionException("ATOM expects exactly one memory operand")
            ptr_op = mem_ops[0]
            addr_val = _get_val(ptr_op, "atom_addr")
            float_ty = self.ir.FloatType()
            ptr = _as_pointer(
                addr_val,
                float_ty,
                f"{ptr_op.GetIRName(self)}_atom_addr" if ptr_op.IsReg else "atom_addr_ptr",
            )

            add_src = None
            for op in reversed(uses):
                if op is dest_reg or op.IsMemAddr:
                    continue
                add_src = op
                break
            if add_src is None:
                raise UnsupportedInstructionException("ATOM ADD is missing its addend")

            add_val = _get_val(add_src, "atom_add_val")
            if add_val.type != float_ty:
                if isinstance(add_val.type, ir.IntType) and add_val.type.width == 32:
                    add_val = IRBuilder.bitcast(add_val, float_ty, "atom_add_bitcast")
                elif isinstance(add_val.type, ir.FloatType) and add_val.type.width < 32:
                    add_val = IRBuilder.fpext(add_val, float_ty, "atom_add_fpext")
                else:
                    raise UnsupportedInstructionException("ATOM ADD expects 32-bit payload")

            result = IRBuilder.atomic_rmw("fadd", ptr, add_val, "seq_cst", "atom_fadd")

            if dest_reg and dest_reg.IsWritableReg and not dest_reg.IsRZ:
                dest_ty = dest_reg.GetIRType(self)
                final_val = result
                if final_val.type != dest_ty:
                    if (
                        isinstance(dest_ty, ir.IntType)
                        and isinstance(final_val.type, ir.FloatType)
                        and dest_ty.width == 32
                    ):
                        final_val = IRBuilder.bitcast(result, dest_ty, "atom_result_bitcast")
                    else:
                        raise UnsupportedInstructionException("ATOM destination type mismatch")
                IRRegs[dest_reg.GetIRName(self)] = final_val

            defs = Inst.GetDefs()
            if defs:
                pred = defs[0]
                if pred.IsPredicateReg:
                    IRRegs[pred.GetIRName(self)] = self.ir.Constant(self.ir.IntType(1), 1)

        elif opcode == "ATOMS" or opcode == "ATOMG":
            tokens = [tok.upper() for tok in Inst.opcodes[1:]]
            atomic_kind = None
            for tok in tokens:
                if tok.startswith("CAS"):
                    atomic_kind = "cas"
                    break
                if tok in {"ADD", "AND", "OR", "XOR", "MAX", "MIN", "EXCH", "INC", "DEC"}:
                    atomic_kind = tok.lower()
                    break
            if atomic_kind is None:
                raise UnsupportedInstructionException(f"Unsupported atomic modifiers {Inst.opcodes}")

            def cast_value(value, target_ty, name):
                if target_ty is None or value.type == target_ty:
                    return value
                if isinstance(value.type, ir.IntType) and isinstance(target_ty, ir.IntType):
                    if target_ty.width > value.type.width:
                        return IRBuilder.zext(value, target_ty, f"{name}_zext")
                    if target_ty.width < value.type.width:
                        return IRBuilder.trunc(value, target_ty, f"{name}_trunc")
                return IRBuilder.bitcast(value, target_ty, f"{name}_cast")

            def float_as_int_ty(fty):
                if isinstance(fty, ir.HalfType):
                    return self.ir.IntType(16)
                if isinstance(fty, ir.FloatType):
                    return self.ir.IntType(32)
                if isinstance(fty, ir.DoubleType):
                    return self.ir.IntType(64)
                raise UnsupportedInstructionException("Unsupported floating-point width for atomic operation")

            def cmpxchg_loop(ptr, compute_new_value, base_name):
                func = IRBuilder.function
                loop_bb = func.append_basic_block(f"{base_name}_loop")
                exit_bb = func.append_basic_block(f"{base_name}_exit")
                IRBuilder.branch(loop_bb)
                IRBuilder.position_at_end(loop_bb)
                current = IRBuilder.load(ptr, f"{base_name}_old")
                new_val = compute_new_value(current)
                cmpxchg = IRBuilder.cmpxchg(ptr, current, new_val, "seq_cst", "seq_cst")
                old_val = IRBuilder.extract_value(cmpxchg, 0, f"{base_name}_oldval")
                success = IRBuilder.extract_value(cmpxchg, 1, f"{base_name}_success")
                IRBuilder.cbranch(success, exit_bb, loop_bb)
                IRBuilder.position_at_end(exit_bb)
                phi = IRBuilder.phi(old_val.type, f"{base_name}_result")
                phi.add_incoming(old_val, loop_bb)
                return phi

            def cas_spin_loop(ptr, cmp_val, new_val, base_name):
                func = IRBuilder.function
                loop_bb = func.append_basic_block(f"{base_name}_loop")
                exit_bb = func.append_basic_block(f"{base_name}_exit")
                IRBuilder.branch(loop_bb)
                IRBuilder.position_at_end(loop_bb)
                cmpxchg = IRBuilder.cmpxchg(ptr, cmp_val, new_val, "seq_cst", "seq_cst")
                old_val = IRBuilder.extract_value(cmpxchg, 0, f"{base_name}_oldval")
                success = IRBuilder.extract_value(cmpxchg, 1, f"{base_name}_success")
                IRBuilder.cbranch(success, exit_bb, loop_bb)
                IRBuilder.position_at_end(exit_bb)
                phi = IRBuilder.phi(old_val.type, f"{base_name}_result")
                phi.add_incoming(old_val, loop_bb)
                return phi

            uses = Inst.GetUses()
            if atomic_kind == "cas":
                if len(uses) < 3:
                    raise UnsupportedInstructionException("CAS requires pointer, compare, and value")
                value_operands = [uses[1], uses[2]]
            else:
                if len(uses) < 2:
                    raise UnsupportedInstructionException(f"{opcode} requires at least two operands")
                value_operands = [uses[1]]

            value_values = [_get_val(op, f"atomic_val_{idx}") for idx, op in enumerate(value_operands)]
            elem_ty = value_values[0].type
            value_values = [cast_value(val, elem_ty, f"atomic_{atomic_kind}_arg{idx}") for idx, val in enumerate(value_values)]

            ptr_operand = uses[0]
            if opcode == "ATOMS":
                addr_index = _get_val(ptr_operand, "atoms_addr")
                ptr = IRBuilder.gep(
                    self.SharedMem,
                    [ir.Constant(ir.IntType(32), 0), addr_index],
                    "atoms_shared_ptr",
                )
            else:
                addr_val = _get_val(ptr_operand, "atomg_addr")
                ptr_name = ptr_operand.GetIRName(self) if hasattr(ptr_operand, "GetIRName") else "atom_addr_ptr"
                ptr = _as_pointer(
                    addr_val,
                    elem_ty,
                    ptr_name if ptr_operand.IsReg else "atom_addr_ptr",
                )

            original_elem_ty = elem_ty
            is_float = isinstance(elem_ty, ir.HalfType) or isinstance(elem_ty, ir.FloatType) or isinstance(elem_ty, ir.DoubleType)
            result = None
            base_name = f"atomic_{atomic_kind}_{Inst.id}"

            if atomic_kind == "cas" and is_float:
                elem_ty = float_as_int_ty(elem_ty)
                value_values = [cast_value(value_values[0], elem_ty, f"{base_name}_cmp_int"), cast_value(value_values[1], elem_ty, f"{base_name}_val_int")]
                is_float = False

            ptr_ty = self.ir.PointerType(elem_ty)
            if ptr.type != ptr_ty:
                ptr = IRBuilder.bitcast(ptr, ptr_ty, "atomic_ptr_cast")

            if atomic_kind == "cas":
                cmp_val = value_values[0]
                new_val = value_values[1]
                spin = "SPIN" in tokens
                if spin:
                    result = cas_spin_loop(ptr, cmp_val, new_val, base_name)
                else:
                    cmpxchg = IRBuilder.cmpxchg(ptr, cmp_val, new_val, "seq_cst", "seq_cst")
                    result = IRBuilder.extract_value(cmpxchg, 0, f"{base_name}_old")
            elif atomic_kind == "inc":
                if not isinstance(elem_ty, ir.IntType):
                    raise UnsupportedInstructionException("Atomic INC only supports integer types")
                limit_val = cast_value(value_values[0], elem_ty, f"{base_name}_limit")
                zero_const = ir.Constant(elem_ty, 0)
                one_const = ir.Constant(elem_ty, 1)

                def compute_inc(old_val):
                    limit_hit = IRBuilder.icmp_unsigned(">=", old_val, limit_val, f"{base_name}_limit_cmp")
                    inc_val = IRBuilder.add(old_val, one_const, f"{base_name}_inc")
                    return IRBuilder.select(limit_hit, zero_const, inc_val, f"{base_name}_select")

                result = cmpxchg_loop(ptr, compute_inc, base_name)
            elif atomic_kind == "dec":
                if not isinstance(elem_ty, ir.IntType):
                    raise UnsupportedInstructionException("Atomic DEC only supports integer types")
                limit_val = cast_value(value_values[0], elem_ty, f"{base_name}_limit")
                zero_const = ir.Constant(elem_ty, 0)
                one_const = ir.Constant(elem_ty, 1)

                def compute_dec(old_val):
                    is_zero = IRBuilder.icmp_unsigned("==", old_val, zero_const, f"{base_name}_is_zero")
                    above_limit = IRBuilder.icmp_unsigned(">", old_val, limit_val, f"{base_name}_gt_limit")
                    wrap = IRBuilder.or_(is_zero, above_limit, f"{base_name}_wrap")
                    dec_val = IRBuilder.sub(old_val, one_const, f"{base_name}_dec")
                    return IRBuilder.select(wrap, limit_val, dec_val, f"{base_name}_select")

                result = cmpxchg_loop(ptr, compute_dec, base_name)
            elif atomic_kind in {"add", "and", "or", "xor", "exch"}:
                val = value_values[0]
                if is_float:
                    if atomic_kind != "add":
                        raise UnsupportedInstructionException(f"Atomic {atomic_kind} not supported for floating-point types")
                    rmw_op = "fadd"
                else:
                    rmw_map = {
                        "add": "add",
                        "and": "and",
                        "or": "or",
                        "xor": "xor",
                        "exch": "xchg",
                    }
                    rmw_op = rmw_map[atomic_kind]
                result = IRBuilder.atomic_rmw(rmw_op, ptr, val, "seq_cst")
            elif atomic_kind in {"max", "min"}:
                if is_float:
                    raise UnsupportedInstructionException(f"Atomic {atomic_kind} not supported for floating-point types")
                rmw_op = "max" if atomic_kind == "max" else "min"
                result = IRBuilder.atomic_rmw(rmw_op, ptr, value_values[0], "seq_cst")
            else:
                raise UnsupportedInstructionException(f"Unhandled atomic operation {atomic_kind}")

            if Inst.GetDefs():
                dest = Inst.GetDefs()[0]
                if not dest.IsRZ:
                    final_value = result
                    if atomic_kind == "cas" and isinstance(original_elem_ty, (ir.HalfType, ir.FloatType, ir.DoubleType)):
                        final_value = IRBuilder.bitcast(result, original_elem_ty, "atomic_cas_result_cast")
                    IRRegs[dest.GetIRName(self)] = final_value

        elif opcode == "PRMT" or opcode == "UPRMT":
            dest = Inst.GetDefs()[0]
            a, sel, b = Inst.GetUses()[0], Inst.GetUses()[1], Inst.GetUses()[2]
            v1 = _get_val(a, "prmt_a")
            v2 = _get_val(b, "prmt_b")
            imm8 = _get_val(sel, "prmt_sel")
            
            # Reinterpret any non-i32 types to i32
            if v1.type != ir.IntType(32):
                v1 = IRBuilder.bitcast(v1, ir.IntType(32), "prmt_a_i32")
            if v2.type != ir.IntType(32):
                v2 = IRBuilder.bitcast(v2, ir.IntType(32), "prmt_b_i32")
            if imm8.type != ir.IntType(32):
                imm8 = IRBuilder.bitcast(imm8, ir.IntType(32), "prmt_sel_i32")

            if "prmt" not in self.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(32), ir.IntType(32)], False)
                self.DeviceFuncs["prmt"] = ir.Function(IRBuilder.module, FuncTy, "prmt")

            prmtVal = IRBuilder.call(self.DeviceFuncs["prmt"], [v1, v2, imm8], "prmt")
            IRRegs[dest.GetIRName(self)] = prmtVal
        
        elif opcode == "HADD2":
            dest = Inst.GetDefs()[0]
            src1, src2 = Inst.GetUses()[0], Inst.GetUses()[1]
            val1 = _get_val(src1)
            val2 = _get_val(src2)

            if "hadd2" not in self.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(32)], False)
                self.DeviceFuncs["hadd2"] = ir.Function(IRBuilder.module, FuncTy, "hadd2")

            haddVal = IRBuilder.call(self.DeviceFuncs["hadd2"], [val1, val2], "hadd2")
            IRRegs[dest.GetIRName(self)] = haddVal

        else:
            print("Unhandled instruction: ", opcode)
            raise UnsupportedInstructionException 


    def GetIRType(self, TypeDesc):
        if TypeDesc == "Int32":
            return self.ir.IntType(32)
        elif TypeDesc == "Float32":
            return self.ir.FloatType()
        elif TypeDesc == "Int32_PTR":
            return self.ir.PointerType(self.ir.IntType(32), 1)
        elif TypeDesc == "Float32_PTR":
            return self.ir.PointerType(self.ir.FloatType(), 1)
        elif TypeDesc == "Int64":
            return self.ir.IntType(64)
        elif TypeDesc == "Int64_PTR":
            return self.ir.PointerType(self.ir.IntType(64), 1)
        elif TypeDesc == "Float64":
            return self.ir.DoubleType()
        elif TypeDesc == "Float64_PTR":
            return self.ir.PointerType(self.ir.DoubleType(), 1)
        elif TypeDesc == "Bool":
            return self.ir.IntType(1)
        elif TypeDesc == "PTR":
            return self.ir.PointerType(self.ir.IntType(8), 1)
        elif TypeDesc == "NOTYPE":
            return self.ir.IntType(32) # Fallback to Int32

        raise ValueError(f"Unknown type: {TypeDesc}")

    def Shutdown(self):
        # Cleanup LLVM environment
        binding.shutdown()
        
    def postprocess_ir(self, ir_code):
        return re.sub(
            r'(@"(?:const_mem|local_mem)"\s*=\s*external)\s+global\b',
            r'\1 thread_local global',
            ir_code
        )

        


Lifter = X86Lifter
