import re

from llvmlite import binding, ir

from transform.cg_patterns import CGPatterns
from transform.pack64 import Pack64
from transform.sr_substitute_reverse import SRSubstituteReverse
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


class NVVMLifter(Lifter):
    def get_transform_passes(self):
        return [
            CGPatterns(),
            Pack64(),
            SRSubstituteReverse(),
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
        llvm_module.triple = "nvptx64-nvidia-cuda"
        llvm_module.data_layout = "e-i64:64-v16:16-v32:32-n16:32:64"
        nvvm_version_node = llvm_module.add_metadata(
            [
                self.ir.Constant(self.ir.IntType(32), 2),
                self.ir.Constant(self.ir.IntType(32), 0),
            ]
        )
        llvm_module.add_named_metadata("nvvmir.version").add(nvvm_version_node)

        bank_ty = self.ir.ArrayType(self.ir.IntType(8), 4096)
        const_mem_ty = self.ir.ArrayType(bank_ty, 5)
        self.ConstMem = self.ir.GlobalVariable(llvm_module, const_mem_ty, "const_mem", 4)
        self.ConstMem.global_constant = True

        shared_array_ty = self.ir.ArrayType(self.ir.IntType(32), 4096)
        self.SharedMem = self.ir.GlobalVariable(
            llvm_module, shared_array_ty, "shared_mem", 3
        )

        local_array_ty = self.ir.ArrayType(self.ir.IntType(8), 4096)
        self.LocalMem = self.ir.GlobalVariable(llvm_module, local_array_ty, "local_mem", 5)

    def build_intrinsic(self, name, ret_ty=None, arg_tys=None, elem_ty=None, ptr_addrspace=None):
        i1 = self.ir.IntType(1)
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)

        if name == "abs":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            return self.ir.Function(self.llvm_module, func_ty, "abs")
        if name == "fabs":
            func_ty = self.ir.FunctionType(self.ir.FloatType(), [self.ir.FloatType()], False)
            return self.ir.Function(self.llvm_module, func_ty, "fabs")
        if name == "fchk":
            func_ty = self.ir.FunctionType(i1, [self.ir.FloatType(), i32], False)
            return self.ir.Function(self.llvm_module, func_ty, "fchk")
        if name == "LeaderStore":
            func_ty = self.ir.FunctionType(
                i1, [self.ir.PointerType(i32, 1), i32], False
            )
            return self.ir.Function(self.llvm_module, func_ty, "LeaderStore")
        if name == "hmma1688f32":
            output_ty = ir.LiteralStructType([ir.FloatType()] * 4)
            input_tys = [ir.FloatType()] * 8
            func_ty = ir.FunctionType(output_ty, input_tys, False)
            return ir.Function(self.llvm_module, func_ty, name="hmma1688f32")
        if name == "hadd2":
            func_ty = ir.FunctionType(i32, [i32, i32], False)
            return ir.Function(self.llvm_module, func_ty, "hadd2")

        if name.startswith("llvm.nvvm.read.ptx.sreg."):
            func_ty = self.ir.FunctionType(i32, [])
            return self.ir.Function(self.llvm_module, func_ty, name)
        if name == "llvm.nvvm.barrier0":
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [])
            return self.ir.Function(self.llvm_module, func_ty, name)

        vote_map = {
            "llvm.nvvm.vote.any.sync": (i1, [i32, i1]),
            "llvm.nvvm.vote.all.sync": (i1, [i32, i1]),
            "llvm.nvvm.vote.uni.sync": (i1, [i32, i1]),
            "llvm.nvvm.vote.ballot.sync": (i32, [i32, i1]),
            "llvm.nvvm.match.any.sync.i32": (i32, [i32, i32]),
            "llvm.nvvm.match.any.sync.i64": (i32, [i32, i64]),
            "llvm.nvvm.brev32": (i32, [i32]),
            "llvm.nvvm.brev64": (i64, [i64]),
            "llvm.nvvm.prmt": (i32, [i32, i32, i32]),
        }
        if name in vote_map:
            ret, args = vote_map[name]
            func_ty = self.ir.FunctionType(ret, args, False)
            return self.ir.Function(self.llvm_module, func_ty, name)

        if name.startswith("llvm.nvvm.shfl.sync."):
            target_ret = ret_ty
            target_args = arg_tys
            if target_ret is None or target_args is None:
                dtype_name = name.rsplit(".", 1)[-1]
                dtype_map = {
                    "i32": i32,
                    "i64": i64,
                    "f32": self.ir.FloatType(),
                    "f64": self.ir.DoubleType(),
                }
                target_ret = dtype_map.get(dtype_name)
                if target_ret is None:
                    raise UnsupportedInstructionException(f"Unsupported shuffle type {dtype_name}")
                target_args = [i32, target_ret, i32, i32]
            func_ty = self.ir.FunctionType(target_ret, target_args, False)
            return self.ir.Function(self.llvm_module, func_ty, name)

        if name in {
            "llvm.nvvm.rcp.approx.ftz.f",
            "llvm.nvvm.rsqrt.approx.f",
            "llvm.sqrt.f32",
            "llvm.nvvm.sin.approx.f",
            "llvm.nvvm.cos.approx.f",
            "llvm.nvvm.ex2.approx.f",
            "llvm.nvvm.lg2.approx.f",
        }:
            func_ty = self.ir.FunctionType(self.ir.FloatType(), [self.ir.FloatType()], False)
            return self.ir.Function(self.llvm_module, func_ty, name)

        if name.startswith("llvm.ctlz."):
            if ret_ty is None or arg_tys is None:
                dtype_name = name.split(".")[-1]
                dtype = {"i32": i32, "i64": i64}.get(dtype_name)
                if dtype is None:
                    raise UnsupportedInstructionException(f"Unsupported ctlz width {name}")
                ret_ty = i32
                arg_tys = [dtype, i1]
            func_ty = self.ir.FunctionType(ret_ty, arg_tys, False)
            return self.ir.Function(self.llvm_module, func_ty, name)

        if name.startswith("llvm.ctpop."):
            if ret_ty is None or arg_tys is None:
                ret_ty = i32
                arg_tys = [i32]
            func_ty = self.ir.FunctionType(ret_ty, arg_tys, False)
            return self.ir.Function(self.llvm_module, func_ty, name)

        raise UnsupportedInstructionException(f"Unknown intrinsic {name}")

    def _lift_function(self, func, llvm_module):
        args = [arg for arg in func.get_args(self) if arg.is_arg]

        param_types = [self.get_ir_type(arg.type_desc) for arg in args]
        func_ty = self.ir.FunctionType(self.ir.VoidType(), param_types, False)
        ir_function = self.ir.Function(llvm_module, func_ty, func.name)
        self._mark_kernel(llvm_module, ir_function)

        func.block_map = {}
        func.build_bb_to_ir_map(ir_function, func.block_map)

        args_map = {}
        for idx, entry in enumerate(args):
            arg_name = entry.get_ir_name(self)
            ir_arg = ir_function.args[idx]
            ir_arg.name = arg_name
            args_map[arg_name] = ir_arg

        entry_block = func.block_map[func.blocks[0]]
        builder = self.ir.IRBuilder(entry_block)
        builder.position_at_start(entry_block)

        ir_regs = {}

        for bb in func.blocks:
            ir_block = func.block_map[bb]
            builder = self.ir.IRBuilder(ir_block)
            self._lift_basic_block(bb, builder, ir_regs, func.block_map, args_map)

        for bb in func.blocks:
            builder = self.ir.IRBuilder(func.block_map[bb])
            self._populate_phi_nodes(bb, builder, ir_regs, func.block_map)

    def _addrspace_pointer(self, builder, value, pointee_ty, addrspace, name):
        tgt_ptr_ty = self.ir.PointerType(pointee_ty, addrspace)
        ty = value.type
        if getattr(ty, "is_pointer", False):
            src_as = getattr(ty, "address_space", 0)
            if src_as == addrspace:
                if ty.pointee == pointee_ty:
                    return value
                return builder.bitcast(value, tgt_ptr_ty, name)
            if ty.pointee == pointee_ty:
                return builder.addrspacecast(value, tgt_ptr_ty, name)
            same_as_ptr = builder.bitcast(
                value,
                self.ir.PointerType(pointee_ty, src_as),
                f"{name}_bitcast_srcas",
            )
            return builder.addrspacecast(same_as_ptr, tgt_ptr_ty, name)

        if not hasattr(ty, "width"):
            raise InvalidTypeException(
                f"Expected integer address for {name}, got {ty}"
            )

        if ty.width < 64:
            value = builder.zext(
                value, self.ir.IntType(64), f"{name}_zext64"
            )
        elif ty.width > 64:
            value = builder.trunc(
                value, self.ir.IntType(64), f"{name}_trunc64"
            )

        return builder.inttoptr(value, tgt_ptr_ty, name)

    def _mark_kernel(self, module, fn):
        try:
            md_node = module.add_metadata(
                [
                    fn,
                    ir.MetaDataString(module, "kernel"),
                    ir.Constant(ir.IntType(32), 1),
                ]
            )
            module.add_named_metadata("nvvm.annotations").add(md_node)
        except Exception as e:
            print("Warning: failed to add nvvm.annotations metadata:", e)


    def lift_instruction(self, Inst, IRBuilder: ir.IRBuilder, IRRegs, ConstMem, block_map):
        if len(Inst.opcodes) == 0:
            raise UnsupportedInstructionException("Empty opcode list")
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
            if op.is_const_mem and not op.is_arg:
                bank = op.const_mem_bank
                offset = op.offsetOrImm
                zero = self.ir.Constant(self.ir.IntType(32), 0)
                bank_idx = self.ir.Constant(self.ir.IntType(32), bank)
                offset_idx = self.ir.Constant(self.ir.IntType(32), int(offset))
                ptr = IRBuilder.gep(
                    self.ConstMem,
                    [zero, bank_idx, offset_idx],
                    f"{name}_cmem_gep" if name else "const_mem_gep",
                )
                typed_ptr = IRBuilder.bitcast(
                    ptr,
                    self.ir.PointerType(op.get_ir_type(self), 4),
                    f"{name}_cmem_ptr" if name else "const_mem_ptr",
                )
                load_name = f"{name}_cmem_load" if name else "const_mem_load"
                return IRBuilder.load(typed_ptr, load_name)
            if op.is_mem_addr and not op.is_reg: # E.g. LDS.U R2 = [0xc]
                return self.ir.Constant(op.get_ir_type(self), op.immediate_value)
            raise UnsupportedInstructionException(f"Unsupported operand type: {op}")

        def _as_pointer(addr_val, pointee_ty, name, addrspace=0):
            return self._addrspace_pointer(IRBuilder, addr_val, pointee_ty, addrspace, name)

        opcode = Inst.opcodes[0]
        handler = self.liftMap.get(opcode)
        if handler is None:
            raise UnsupportedInstructionException(f"Unhandled instruction: {opcode}")
        return handler(Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=_get_val, _as_pointer=_as_pointer)
    
    @lift_for("S2R")
    def lift_s2r(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        valop = Inst.get_uses()[0]

        def _call_nvvm(name, tag):
            return IRBuilder.call(self.get_intrinsic(name), [], tag)

        if valop.is_thread_idx_x:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.tid.x", "tid_x")
        elif valop.is_thread_idx_y:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.tid.y", "tid_y")
        elif valop.is_thread_idx_z:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.tid.z", "tid_z")
        elif valop.is_block_dim_x:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.ntid.x", "ntid_x")
        elif valop.is_block_dim_y:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.ntid.y", "ntid_y")
        elif valop.is_block_dim_z:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.ntid.z", "ntid_z")
        elif valop.is_block_idx_x:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.ctaid.x", "ctaid_x")
        elif valop.is_block_idx_y:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.ctaid.y", "ctaid_y")
        elif valop.is_block_idx_z:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.ctaid.z", "ctaid_z")
        elif valop.is_grid_dim_x:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.nctaid.x", "nctaid_x")
        elif valop.is_grid_dim_y:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.nctaid.y", "nctaid_y")
        elif valop.is_grid_dim_z:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.nctaid.z", "nctaid_z")
        elif valop.is_lane_id:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.laneid", "laneid")
        elif valop.is_warp_id:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.warpid", "warpid")
        elif valop.is_warp_size:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.warpsize", "warpsize")
        elif valop.is_active_mask:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.activemask", "activemask")
        elif valop.is_lane_mask_eq:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.lanemask.eq", "lanemask_eq")
        elif valop.is_lane_mask_le:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.lanemask.le", "lanemask_le")
        elif valop.is_lane_mask_lt:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.lanemask.lt", "lanemask_lt")
        elif valop.is_lane_mask_ge:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.lanemask.ge", "lanemask_ge")
        elif valop.is_lane_mask_gt:
            val = _call_nvvm("llvm.nvvm.read.ptx.sreg.lanemask.gt", "lanemask_gt")
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
            f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "ldg_addr_ptr",
            addrspace=1,
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
            f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "stg_addr_ptr",
            addrspace=1,
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
                self.ir.PointerType(dest.get_ir_type(self), 3),
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
                self.ir.PointerType(val.get_ir_type(self), 3),
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
                self.ir.PointerType(val.get_ir_type(self), 5),
                "stl_local_addr_cast",
            )
        v = _get_val(val, "stl_val")
        IRBuilder.store(v, addr)


    @lift_for("MUFU")
    def lift_mufu(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        func = Inst.opcodes[1]
        v = _get_val(src, "mufu_src")

        intrinsic_name = None
        if func == "RCP":
            intrinsic_name = "llvm.nvvm.rcp.approx.ftz.f"
        elif func == "RSQ":
            intrinsic_name = "llvm.nvvm.rsqrt.approx.f"
        elif func == "SQRT":
            intrinsic_name = "llvm.sqrt.f32"
        elif func == "SIN":
            intrinsic_name = "llvm.nvvm.sin.approx.f"
        elif func == "COS":
            intrinsic_name = "llvm.nvvm.cos.approx.f"
        elif func == "EX2":
            intrinsic_name = "llvm.nvvm.ex2.approx.f"
        elif func == "LG2":
            intrinsic_name = "llvm.nvvm.lg2.approx.f"
        else:
            raise UnsupportedInstructionException

        intrinsic = self.get_intrinsic(
            intrinsic_name, ret_ty=self.ir.FloatType(), arg_tys=[self.ir.FloatType()]
        )
        res = IRBuilder.call(intrinsic, [v], f"mufu_{func.lower()}")
        IRRegs[dest.get_ir_name(self)] = res
    @lift_for("CS2R")
    def lift_cs2r(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        # CS2R (Convert Special Register to Register)
        ResOp = Inst.get_defs()[0]
        ValOp = Inst.get_uses()[0]
        if ResOp.is_reg:
            dest_name = ResOp.get_ir_name(self)

            # Determine which special register this is using the new approach
            if ValOp.is_rz:
                IRVal = self.ir.Constant(self.ir.IntType(32), 0)
            else:
                if ValOp.is_thread_idx_x or (ValOp.is_thread_idx and not ValOp.special_register_axis):
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.tid.x"), [], "cs2r_tid_x")
                elif ValOp.is_thread_idx_y:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.tid.y"), [], "cs2r_tid_y")
                elif ValOp.is_thread_idx_z:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.tid.z"), [], "cs2r_tid_z")
                elif ValOp.is_block_dim_x or (ValOp.is_block_dim and not ValOp.special_register_axis):
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.ntid.x"), [], "cs2r_ntid_x")
                elif ValOp.is_block_dim_y:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.ntid.y"), [], "cs2r_ntid_y")
                elif ValOp.is_block_dim_z:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.ntid.z"), [], "cs2r_ntid_z")
                elif ValOp.is_block_idx_x or (ValOp.is_block_idx and not ValOp.special_register_axis):
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.ctaid.x"), [], "cs2r_ctaid_x")
                elif ValOp.is_block_idx_y:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.ctaid.y"), [], "cs2r_ctaid_y")
                elif ValOp.is_block_idx_z:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.ctaid.z"), [], "cs2r_ctaid_z")
                elif ValOp.is_grid_dim_x or (ValOp.is_grid_dim and not ValOp.special_register_axis):
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.nctaid.x"), [], "cs2r_nctaid_x")
                elif ValOp.is_grid_dim_y:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.nctaid.y"), [], "cs2r_nctaid_y")
                elif ValOp.is_grid_dim_z:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.nctaid.z"), [], "cs2r_nctaid_z")
                elif ValOp.is_lane_id:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.laneid"), [], "cs2r_lane")
                elif ValOp.is_warp_id:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.warpid"), [], "cs2r_warp")
                elif ValOp.is_warp_size:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.warpsize"), [], "cs2r_warpsize")
                elif ValOp.is_active_mask:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.activemask"), [], "cs2r_activemask")
                elif ValOp.is_lane_mask_eq:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.lanemask.eq"), [], "cs2r_lanemask_eq")
                elif ValOp.is_lane_mask_le:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.lanemask.le"), [], "cs2r_lanemask_le")
                elif ValOp.is_lane_mask_lt:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.lanemask.lt"), [], "cs2r_lanemask_lt")
                elif ValOp.is_lane_mask_ge:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.lanemask.ge"), [], "cs2r_lanemask_ge")
                elif ValOp.is_lane_mask_gt:
                    IRVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.read.ptx.sreg.lanemask.gt"), [], "cs2r_lanemask_gt")
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

        barrier = self.get_intrinsic("llvm.nvvm.barrier0")
        IRBuilder.call(barrier, [], "barrier0")


    @lift_for("VOTE", "VOTEU")
    def lift_vote(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        mask_op, pred_op = Inst.get_uses()[0], Inst.get_uses()[1]

        pred_val = _get_val(pred_op, "vote_pred")
        mask_val = _get_val(mask_op, "vote_mask")

        if mask_op.is_pt and not mask_op.is_not_reg:
            mask_val = self.ir.Constant(self.ir.IntType(32), -1)
        else:
            mask_val = _get_val(mask_op, "vote_mask")

        mode = Inst.opcodes[1].upper()

        if mode == "ANY":
            vote_val = IRBuilder.call(
                self.get_intrinsic("llvm.nvvm.vote.any.sync"),
                [mask_val, pred_val],
                "vote_any",
            )
        elif mode == "ALL":
            vote_val = IRBuilder.call(
                self.get_intrinsic("llvm.nvvm.vote.all.sync"),
                [mask_val, pred_val],
                "vote_all",
            )
        elif mode == "UNI":
            vote_val = IRBuilder.call(
                self.get_intrinsic("llvm.nvvm.vote.uni.sync"),
                [mask_val, pred_val],
                "vote_uni",
            )
        elif mode == "BALLOT":
            vote_val = IRBuilder.call(
                self.get_intrinsic("llvm.nvvm.vote.ballot.sync"),
                [mask_val, pred_val],
                "vote_ballot",
            )
        else:
            raise UnsupportedInstructionException

        dest_ty = dest.get_ir_type(self)
        vote_ty = vote_val.type
        if dest_ty != vote_ty:
            if hasattr(dest_ty, "width") and hasattr(vote_ty, "width"):
                if dest_ty.width > vote_ty.width:
                    vote_val = IRBuilder.zext(vote_val, dest_ty, f"vote_{mode.lower()}_zext")
                else:
                    vote_val = IRBuilder.trunc(vote_val, dest_ty, f"vote_{mode.lower()}_trunc")
            else:
                vote_val = IRBuilder.bitcast(vote_val, dest_ty, f"vote_{mode.lower()}_cast")

        IRRegs[dest.get_ir_name(self)] = vote_val


    @lift_for("SHFL")
    def lift_shfl(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        # SHFL.<MODE> PT, R3 = R0, 0x8, 0x1f
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

        func_name = "llvm.nvvm.shfl.sync." + mode.lower() + "." + dtype.intrinsic_name

        if mode == "DOWN":
            shfl_name = "shfl_down"
            shfl_intr = self.get_intrinsic(func_name, ret_ty=dtype, arg_tys=[i32, dtype, i32, i32])
        elif mode == "UP":
            shfl_name = "shfl_up"
            shfl_intr = self.get_intrinsic(func_name, ret_ty=dtype, arg_tys=[i32, dtype, i32, i32])
        elif mode == "BFLY":
            shfl_name = "shfl_bfly"
            shfl_intr = self.get_intrinsic(func_name, ret_ty=dtype, arg_tys=[i32, dtype, i32, i32])
        elif mode == "IDX":
            shfl_name = "shfl_idx"
            shfl_intr = self.get_intrinsic(func_name, ret_ty=dtype, arg_tys=[i32, dtype, i32, i32])
        else:
            raise UnsupportedInstructionException

        shflVal = IRBuilder.call(
            shfl_intr,
            [mask_const, val, off, wid],
            shfl_name,
        )

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

        if mode != "ANY":
            raise UnsupportedInstructionException

        if type == "U64":
            dtype = self.ir.IntType(64)
        elif type == "U32":
            dtype = self.ir.IntType(32)
        else:
            raise UnsupportedInstructionException

        if val.type != dtype:
            if hasattr(val.type, "width") and val.type.width > dtype.width:
                val = IRBuilder.trunc(val, dtype, "match_val_cast")
            else:
                val = IRBuilder.zext(val, dtype, "match_val_cast")

        mask_const = self.ir.Constant(self.ir.IntType(32), 0xFFFFFFFF)

        if dtype.width == 64:
            matchVal = IRBuilder.call(
                self.get_intrinsic("llvm.nvvm.match.any.sync.i64"),
                [mask_const, val],
                "match_any_i64",
            )
        else:
            matchVal = IRBuilder.call(
                self.get_intrinsic("llvm.nvvm.match.any.sync.i32"),
                [mask_const, val],
                "match_any_i32",
            )

        if dest.get_ir_type(self) != matchVal.type:
            matchVal = IRBuilder.zext(matchVal, dest.get_ir_type(self), "match_any_ext")

        IRRegs[dest.get_ir_name(self)] = matchVal


    @lift_for("BREV", "UBREV")
    def lift_brev(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src, "brev_val")

        dest_ty = dest.get_ir_type(self)
        if dest_ty.width == 64:
            if val.type != self.ir.IntType(64):
                if hasattr(val.type, "width") and val.type.width > 64:
                    val = IRBuilder.trunc(val, self.ir.IntType(64), "brev64_val")
                else:
                    val = IRBuilder.zext(val, self.ir.IntType(64), "brev64_val")
            revVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.brev64"), [val], "brev64")
        else:
            if val.type != self.ir.IntType(32):
                if hasattr(val.type, "width") and val.type.width > 32:
                    val = IRBuilder.trunc(val, self.ir.IntType(32), "brev32_val")
                else:
                    val = IRBuilder.zext(val, self.ir.IntType(32), "brev32_val")
            revVal = IRBuilder.call(self.get_intrinsic("llvm.nvvm.brev32"), [val], "brev32")

        if revVal.type != dest_ty:
            if dest_ty.width < revVal.type.width:
                revVal = IRBuilder.trunc(revVal, dest_ty, "brev_trunc")
            elif dest_ty.width > revVal.type.width:
                revVal = IRBuilder.zext(revVal, dest_ty, "brev_zext")

        IRRegs[dest.get_ir_name(self)] = revVal


    @lift_for("FLO", "UFLO")
    def lift_flo(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src)

        type = Inst.opcodes[1]
        mode = Inst.opcodes[2] if len(Inst.opcodes) > 2 else None

        if type == "U64":
            dtype = ir.IntType(64)
            typeName = "i64"
        elif type == "U32":
            dtype = ir.IntType(32)
            typeName = "i32"
        else:
            raise UnsupportedInstructionException

        if mode == "SH" or mode is None:
            funcName = f"llvm.ctlz.{typeName}"
        else:
            raise UnsupportedInstructionException

        flo_fn = self.get_intrinsic(
            funcName,
            ret_ty=ir.IntType(32),
            arg_tys=[dtype, ir.IntType(1)],
        )
        floVal = IRBuilder.call(
            flo_fn,
            [val, ir.Constant(ir.IntType(1), 0)],
            "flo",
        )
        IRRegs[dest.get_ir_name(self)] = floVal


    @lift_for("ATOM")
    def lift_atom(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]

        if len(Inst.opcodes) < 4 or Inst.opcodes[2].upper() != "ADD":
            raise UnsupportedInstructionException()

        if not Inst.opcodes[3].startswith("F"):
            raise UnsupportedInstructionException()
        if "STRONG" not in modifiers:
            raise UnsupportedInstructionException()

        uses = Inst.get_uses()

        dest_reg = None
        if uses[0].is_reg and not uses[0].is_mem_addr:
            dest_reg = uses[0]

        mem_ops = [op for op in uses if op.is_mem_addr]
        ptr_op = mem_ops[0]
        addr_val = _get_val(ptr_op, "atom_addr")
        float_ty = self.ir.FloatType()
        ptr = _as_pointer(
            addr_val,
            float_ty,
            f"{ptr_op.get_ir_name(self)}_atom_addr" if ptr_op.is_reg else "atom_addr_ptr",
            addrspace=1,
        )

        add_src = uses[-1]

        add_val = _get_val(add_src, "atom_add_val")

        result = IRBuilder.atomic_rmw("fadd", ptr, add_val, "seq_cst", "atom_fadd")

        if dest_reg and dest_reg.is_writable_reg and not dest_reg.is_rz:
            dest_ty = dest_reg.get_ir_type(self)
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
            IRRegs[dest_reg.get_ir_name(self)] = final_val

        defs = Inst.get_defs()
        if defs:
            pred = defs[0]
            if pred.is_predicate_reg:
                IRRegs[pred.get_ir_name(self)] = self.ir.Constant(self.ir.IntType(1), 1)


    @lift_for("ATOMS", "ATOMG")
    def lift_atoms(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        raise UnsupportedInstructionException("ATOMS/ATOMG not implemented")


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

        prmt_intr = self.get_intrinsic("llvm.nvvm.prmt")
        prmtVal = IRBuilder.call(prmt_intr, [v1, v2, imm8], "prmt")
        IRRegs[dest.get_ir_name(self)] = prmtVal


    @lift_for("QSPC")
    def lift_qspc(self, Inst, IRBuilder, IRRegs, ConstMem, block_map, _get_val=None, _as_pointer=None):
        print("QSPC met; don't know what to do ")
        pass

Lifter = NVVMLifter
