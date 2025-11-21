import re

from llvmlite import binding, ir

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
from lift.lifter import Lifter, lift_for


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

    def setup_module(self, llvm_module):
        bank_ty = self.ir.ArrayType(self.ir.IntType(8), 4096)
        const_array_ty = self.ir.ArrayType(bank_ty, 5)
        self.ConstMem = self.ir.GlobalVariable(llvm_module, const_array_ty, "const_mem")
        shared_array_ty = self.ir.ArrayType(self.ir.IntType(8), 32768)
        self.SharedMem = self.ir.GlobalVariable(
            llvm_module, shared_array_ty, "shared_mem"
        )
        local_array_ty = self.ir.ArrayType(self.ir.IntType(8), 32768)
        self.LocalMem = self.ir.GlobalVariable(llvm_module, local_array_ty, "local_mem")

    def build_intrinsic(self, name, ret_ty=None, arg_tys=None, elem_ty=None, ptr_addrspace=None):
        i1 = self.ir.IntType(1)
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)

        if name == "abs":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            return self.ir.Function(self.llvm_module, func_ty, "abs")
        if name == "syncthreads":
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [], False)
            return self.ir.Function(self.llvm_module, func_ty, "syncthreads")
        if name == "LeaderStore":
            func_ty = self.ir.FunctionType(
                i1, [self.ir.PointerType(i32), i32], False
            )
            return self.ir.Function(self.llvm_module, func_ty, "LeaderStore")
        if name == "fchk":
            func_ty = self.ir.FunctionType(i1, [self.ir.FloatType(), i32], False)
            return self.ir.Function(self.llvm_module, func_ty, "fchk")
        if name == "hmma1688f32":
            output_ty = ir.LiteralStructType([ir.FloatType()] * 4)
            input_tys = [ir.FloatType()] * 8
            func_ty = ir.FunctionType(output_ty, input_tys, False)
            return ir.Function(self.llvm_module, func_ty, name="hmma1688f32")

        if name.startswith("vote_"):
            mode = name.split("_", 1)[1]
            ret_map = {
                "any": i1,
                "all": i1,
                "uni": i1,
                "ballot": i32,
            }
            ret = ret_map.get(mode)
            if ret is None:
                raise UnsupportedInstructionException(f"Unknown vote intrinsic {name}")
            func_ty = self.ir.FunctionType(ret, [i32, i1], False)
            return self.ir.Function(self.llvm_module, func_ty, name)

        if name.startswith("warp_shfl_"):
            parts = name.split("_")
            if len(parts) < 3:
                raise UnsupportedInstructionException(f"Malformed shuffle intrinsic {name}")
            dtype_name = parts[-1]
            dtype_map = {
                "i32": i32,
                "i64": i64,
                "f32": self.ir.FloatType(),
                "f64": self.ir.DoubleType(),
            }
            dtype = dtype_map.get(dtype_name)
            if dtype is None:
                raise UnsupportedInstructionException(f"Unsupported shuffle type {dtype_name}")
            func_ty = self.ir.FunctionType(dtype, [i32, dtype, i32, i32], False)
            return self.ir.Function(self.llvm_module, func_ty, name)

        if name.startswith("match_"):
            _, mode, type_tok = name.split("_")
            if mode != "ANY":
                raise UnsupportedInstructionException(f"Unsupported match mode {mode}")
            if type_tok == "U64":
                dtype = i64
            elif type_tok == "U32":
                dtype = i32
            else:
                raise UnsupportedInstructionException(f"Unsupported match type {type_tok}")
            func_ty = self.ir.FunctionType(i32, [dtype], False)
            return self.ir.Function(self.llvm_module, func_ty, name)

        if name == "brev":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            return self.ir.Function(self.llvm_module, func_ty, "brev")

        if name.startswith("llvm.ctlz."):
            if ret_ty is None or arg_tys is None:
                dtype_name = name.split(".")[-1]
                dtype = {"i32": i32, "i64": i64}.get(dtype_name)
                if dtype is None:
                    raise UnsupportedInstructionException(f"Unsupported ctlz width {name}")
                ret_ty = i32
                arg_tys = [dtype]
            func_ty = self.ir.FunctionType(ret_ty, arg_tys, False)
            return self.ir.Function(self.llvm_module, func_ty, name)

        if name.startswith("llvm.ctpop."):
            if ret_ty is None or arg_tys is None:
                ret_ty = i32
                arg_tys = [i32]
            func_ty = self.ir.FunctionType(ret_ty, arg_tys, False)
            return self.ir.Function(self.llvm_module, func_ty, name)

        if name == "prmt":
            func_ty = self.ir.FunctionType(i32, [i32, i32, i32], False)
            return self.ir.Function(self.llvm_module, func_ty, "prmt")
        if name == "hadd2":
            func_ty = self.ir.FunctionType(i32, [i32, i32], False)
            return self.ir.Function(self.llvm_module, func_ty, "hadd2")

        raise UnsupportedInstructionException(f"Unknown intrinsic {name}")

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
            mask64 = builder.sub(
                builder.shl(one64, lane_inc, f"{tag}_shift"), one64, f"{tag}_sub"
            )
        elif kind == "lt":
            mask64 = builder.sub(
                builder.shl(one64, lane_i64, f"{tag}_shift"), one64, f"{tag}_sub"
            )
        elif kind == "ge":
            mask64 = builder.shl(all_ones64, lane_i64, f"{tag}_shift")
        elif kind == "gt":
            lane_inc = builder.add(lane_i64, one64, f"{tag}_inc")
            mask64 = builder.shl(all_ones64, lane_inc, f"{tag}_shift")
        else:
            mask64 = self.ir.Constant(i64, 0)

        return builder.trunc(mask64, i32, tag)

    def _lift_function(self, func, llvm_module):
        args = func.get_args(self)

        func_ty = self.ir.FunctionType(self.ir.VoidType(), [])
        ir_function = self.ir.Function(llvm_module, func_ty, func.name)

        func.block_map = {}
        func.build_bb_to_ir_map(ir_function, func.block_map)

        const_mem = {}
        entry_block = func.block_map[func.blocks[0]]
        builder = self.ir.IRBuilder(entry_block)
        builder.position_at_start(entry_block)

        offset_to_sr = {v: k for k, v in SR_TO_OFFSET.items()}

        for entry in args:
            addr = builder.gep(
                self.ConstMem,
                [
                    self.ir.Constant(self.ir.IntType(64), 0),
                    self.ir.Constant(self.ir.IntType(64), 0),
                    self.ir.Constant(self.ir.IntType(64), entry.arg_offset),
                ],
            )
            name = offset_to_sr.get(entry.arg_offset, entry.get_ir_name(self))
            addr = builder.bitcast(
                addr, self.ir.PointerType(self.get_ir_type(entry.type_desc))
            )
            val = builder.load(addr, name)
            const_mem[entry.get_ir_name(self)] = val

        ir_regs = {}

        for bb in func.blocks:
            ir_block = func.block_map[bb]
            builder = self.ir.IRBuilder(ir_block)
            self._lift_basic_block(bb, builder, ir_regs, func.block_map, const_mem)

        for bb in func.blocks:
            builder = self.ir.IRBuilder(func.block_map[bb])
            self._populate_phi_nodes(bb, builder, ir_regs, func.block_map)

    def lift_instruction(self, Inst, IRBuilder: ir.IRBuilder, IRRegs, ConstMem, block_map):
        if len(Inst.opcodes) == 0:
            raise UnsupportedInstructionException("Empty opcode list")
        opcode = Inst.opcodes[0]

        def roughSearch(op):
            reg = op.reg
            name = op.get_ir_name(self)
            targetType = name.replace(reg, "")

            bestKey = max(IRRegs.keys(), key=lambda k: (k.startswith(reg), len(k)))

            val = IRBuilder.bitcast(IRRegs[bestKey], op.get_ir_type(self), f"{name}_cast")

            return val

        def _get_val(op, name=""):
            if op.is_rz:
                return self.ir.Constant(op.get_ir_type(self), 0)
            if op.is_pt:
                return self.ir.Constant(op.get_ir_type(self), not op.is_not_reg)
            if op.is_reg:
                irName = op.get_ir_name(self)
                val = IRRegs[irName]
                    
                if op.is_negative_reg:
                    if op.get_type_desc().startswith('F'):
                        val = IRBuilder.fneg(val, f"{name}_fneg")
                    else:
                        val = IRBuilder.neg(val, f"{name}_neg")
                if op.is_not_reg:
                    val = IRBuilder.not_(val, f"{name}_not")
                if op.is_abs_reg:
                    if op.get_type_desc().startswith('F'):
                        fabs_fn = self.get_intrinsic("fabs")
                        val = IRBuilder.call(fabs_fn, [val], f"{name}_fabs")
                    else:
                        abs_fn = self.get_intrinsic("abs")
                        val = IRBuilder.call(abs_fn, [val], f"{name}_abs")
                return val
            if op.is_arg:
                return ConstMem[op.get_ir_name(self)]
            if op.is_immediate:
                return self.ir.Constant(op.get_ir_type(self), op.immediate_value)
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

        handler = self.liftMap.get(opcode)
        if handler is None:
            raise UnsupportedInstructionException(f"Unhandled instruction: {opcode}")
        return handler(Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=_get_val, _as_pointer=_as_pointer)
    
    @lift_for("S2R")
    def lift_s2r(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        valop = Inst.get_uses()[0]
        offset = SR_TO_OFFSET.get(valop.name)
        if offset is not None:
            val = self._load_const_mem_u32(IRBuilder, offset, f"sr_{offset:x}")
        elif valop.is_warp_size:
            val = self.ir.Constant(self.ir.IntType(32), 32)
        elif valop.is_active_mask:
            val = self.ir.Constant(self.ir.IntType(32), 0xFFFFFFFF)
        elif valop.is_lane_mask_eq:
            val = self._emit_lane_mask(IRBuilder, "eq", "lanemask_eq")
        elif valop.is_lane_mask_le:
            val = self._emit_lane_mask(IRBuilder, "le", "lanemask_le")
        elif valop.is_lane_mask_lt:
            val = self._emit_lane_mask(IRBuilder, "lt", "lanemask_lt")
        elif valop.is_lane_mask_ge:
            val = self._emit_lane_mask(IRBuilder, "ge", "lanemask_ge")
        elif valop.is_lane_mask_gt:
            val = self._emit_lane_mask(IRBuilder, "gt", "lanemask_gt")
        else:
            print(f"S2R: Unknown special register {valop.name}")
            val = self.ir.Constant(self.ir.IntType(32), 0)
        IRRegs[dest.get_ir_name(self)] = val


    @lift_for("LDG", "LDG64")
    def lift_ldg(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        ptr = Inst.get_uses()[0]
        addr = _get_val(ptr, "ldg_addr")
        pointee_ty = dest.get_ir_type(self)
        addr_ptr = _as_pointer(
            addr,
            pointee_ty,
            f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "ldg_addr_ptr"
            )
        val = IRBuilder.load(addr_ptr, "ldg", typ=pointee_ty)
        IRRegs[dest.get_ir_name(self)] = val


    @lift_for("STG")
    def lift_stg(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        uses = Inst.get_uses()
        ptr, val = uses[0], uses[1]
        addr = _get_val(ptr, "stg_addr")
        v = _get_val(val, "stg_val")
        addr_ptr = _as_pointer(
            addr,
            v.type,
            f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "stg_addr_ptr"
            )
        IRBuilder.store(v, addr_ptr)


    @lift_for("LDS")
    def lift_lds(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        ptr = Inst.get_uses()[0]
        addr = _get_val(ptr, "lds_addr")
        addr = IRBuilder.gep(self.SharedMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "lds_shared_addr")
        if addr.type.pointee != dest.get_ir_type(self):
            addr = IRBuilder.bitcast(
                addr,
                self.ir.PointerType(dest.get_ir_type(self)),
                "lds_shared_addr_cast",
            )
        val = IRBuilder.load(addr, "lds", typ=dest.get_ir_type(self))
        IRRegs[dest.get_ir_name(self)] = val


    @lift_for("STS")
    def lift_sts(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        uses = Inst.get_uses()
        ptr, val = uses[0], uses[1]
        addr = _get_val(ptr, "sts_addr")
        addr = IRBuilder.gep(self.SharedMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "sts_shared_addr")
        if addr.type.pointee != val.get_ir_type(self):
            addr = IRBuilder.bitcast(
                addr,
                self.ir.PointerType(val.get_ir_type(self)),
                "sts_shared_addr_cast",
            )
        v = _get_val(val, "sts_val")
        IRBuilder.store(v, addr)


    @lift_for("LDL")
    def lift_ldl(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        ptr = Inst.get_uses()[0]
        addr = _get_val(ptr, "ldl_addr")
        addr = IRBuilder.gep(self.LocalMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "ldl_local_addr")
        val = IRBuilder.load(addr, "ldl", typ=dest.get_ir_type(self))
        IRRegs[dest.get_ir_name(self)] = val


    @lift_for("STL")
    def lift_stl(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        uses = Inst.get_uses()
        ptr, val = uses[0], uses[1]
        addr = _get_val(ptr, "stl_addr")
        addr = IRBuilder.gep(self.LocalMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "stl_local_addr")
        if addr.type.pointee != val.get_ir_type(self):
            addr = IRBuilder.bitcast(
                addr,
                self.ir.PointerType(val.get_ir_type(self)),
                "stl_local_addr_cast",
            )
        v = _get_val(val, "stl_val")
        IRBuilder.store(v, addr)


    @lift_for("MUFU")
    def lift_mufu(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        func = Inst.opcodes[1] if len(Inst.opcodes) > 1 else None
        v = _get_val(src, "mufu_src")

        if func == "RCP": # 1/v
            one = self.ir.Constant(dest.get_ir_type(self), 1.0)
            res = IRBuilder.fdiv(one, v, "mufu_rcp")
            IRRegs[dest.get_ir_name(self)] = res
        else:
            raise UnsupportedInstructionException

    @lift_for("CS2R")
    def lift_cs2r(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        ResOp = Inst.get_defs()[0]
        ValOp = Inst.get_uses()[0]
        if ResOp.is_reg:
            dest_name = ResOp.get_ir_name(self)
            if ValOp.is_rz:
                IRVal = self.ir.Constant(self.ir.IntType(32), 0)
            else:
                offset = SR_TO_OFFSET.get(ValOp.name)
                if offset is not None:
                    IRVal = self._load_const_mem_u32(IRBuilder, offset, f"cs2r_{offset:x}")
                elif ValOp.is_warp_size:
                    IRVal = self.ir.Constant(self.ir.IntType(32), 32)
                elif ValOp.is_active_mask:
                    IRVal = self.ir.Constant(self.ir.IntType(32), 0xFFFFFFFF)
                elif ValOp.is_lane_mask_eq:
                    IRVal = self._emit_lane_mask(IRBuilder, "eq", "cs2r_lanemask_eq")
                elif ValOp.is_lane_mask_le:
                    IRVal = self._emit_lane_mask(IRBuilder, "le", "cs2r_lanemask_le")
                elif ValOp.is_lane_mask_lt:
                    IRVal = self._emit_lane_mask(IRBuilder, "lt", "cs2r_lanemask_lt")
                elif ValOp.is_lane_mask_ge:
                    IRVal = self._emit_lane_mask(IRBuilder, "ge", "cs2r_lanemask_ge")
                elif ValOp.is_lane_mask_gt:
                    IRVal = self._emit_lane_mask(IRBuilder, "gt", "cs2r_lanemask_gt")
                else:
                    print(f"CS2R: Unknown special register {ValOp}")
                    IRVal = self.ir.Constant(self.ir.IntType(32), 0)

            if dest_name in IRRegs and getattr(IRRegs[dest_name].type, "is_pointer", False):
                IRBuilder.store(IRVal, IRRegs[dest_name])
            else:
                IRRegs[dest_name] = IRVal


    @lift_for("BAR")
    def lift_bar(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        if len(Inst.opcodes) > 1 and Inst.opcodes[1] != "SYNC":
            raise UnsupportedInstructionException

        sync_fn = self.get_intrinsic("syncthreads")
        IRBuilder.call(sync_fn, [], "barrier")


    @lift_for("VOTE", "VOTEU")
    def lift_vote(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        pred_val = _get_val(Inst.get_uses()[0], "vote_pred")
        mask_val = _get_val(Inst.get_uses()[1], "vote_mask")

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
        vote_fn = self.get_intrinsic(func_name)

        voteVal = IRBuilder.call(vote_fn, [mask_val, pred_val], func_name)

        dest_ty = dest.get_ir_type(self)
        vote_ty = voteVal.type
        if dest_ty != vote_ty:
            if hasattr(dest_ty, "width") and hasattr(vote_ty, "width"):
                if dest_ty.width > vote_ty.width:
                    voteVal = IRBuilder.zext(voteVal, dest_ty, f"vote_{mode.lower()}_zext")
                else:
                    voteVal = IRBuilder.trunc(voteVal, dest_ty, f"vote_{mode.lower()}_trunc")
            else:
                voteVal = IRBuilder.bitcast(voteVal, dest_ty, f"vote_{mode.lower()}_cast")

        IRRegs[dest.get_ir_name(self)] = voteVal


    @lift_for("SHFL")
    def lift_shfl(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        destPred = Inst.get_defs()[0]
        destReg = Inst.get_defs()[1]
        srcReg = Inst.get_uses()[0]
        offset = Inst.get_uses()[1]
        width = Inst.get_uses()[2]

        val = _get_val(srcReg, "shfl_val")
        off = _get_val(offset, "shfl_offset")
        wid = _get_val(width, "shfl_width")

        mode = Inst.opcodes[1].upper()

        dtype = destReg.get_ir_type(self)
        i32 = self.ir.IntType(32)
        mask_const = self.ir.Constant(i32, 0xFFFFFFFF)

        func_name = "warp_shfl_" + mode.lower() + "_" + dtype.intrinsic_name
        shfl_fn = self.get_intrinsic(func_name)

        shflVal = IRBuilder.call(
            shfl_fn,
            [mask_const, val, off, wid],
            func_name,
        )

        dest_type = destReg.get_ir_type(self)

        IRRegs[destReg.get_ir_name(self)] = shflVal
        if not destPred.is_pt:
            IRRegs[destPred.get_ir_name(self)] = self.ir.Constant(self.ir.IntType(1), 1)


    @lift_for("MATCH")
    def lift_match(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]

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
        match_fn = self.get_intrinsic(funcName)

        matchVal = IRBuilder.call(match_fn, [val], "match")
        IRRegs[dest.get_ir_name(self)] = matchVal


    @lift_for("BREV", "UBREV")
    def lift_brev(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src)

        rev_fn = self.get_intrinsic("brev")

        revVal = IRBuilder.call(rev_fn, [val], "brev")
        IRRegs[dest.get_ir_name(self)] = revVal


    @lift_for("FLO", "UFLO")
    def lift_flo(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
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

        ctlz_fn = self.get_intrinsic(
            funcName, ret_ty=ir.IntType(32), arg_tys=[dtype]
        )

        floVal = IRBuilder.call(ctlz_fn, [val], "flo")
        IRRegs[dest.get_ir_name(self)] = floVal


    @lift_for("ATOM")
    def lift_atom(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        raise UnsupportedInstructionException()


    @lift_for("ATOMS", "ATOMG")
    def lift_atoms(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        raise UnsupportedInstructionException()

    @lift_for("PRMT", "UPRMT")
    def lift_prmt(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        a, sel, b = Inst.get_uses()[0], Inst.get_uses()[1], Inst.get_uses()[2]
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

        prmt_fn = self.get_intrinsic("prmt")

        prmtVal = IRBuilder.call(prmt_fn, [v1, v2, imm8], "prmt")
        IRRegs[dest.get_ir_name(self)] = prmtVal


    def postprocess_ir(self, ir_code):
        return re.sub(
            r'(@"(?:const_mem|local_mem)"\s*=\s*external)\s+global\b',
            r'\1 thread_local global',
            ir_code
        )

        


Lifter = X86Lifter
