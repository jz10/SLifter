from llvmlite import ir, binding

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
from lift.lifter import Lifter


class UnsupportedOperatorException(Exception):
    pass


class UnsupportedInstructionException(Exception):
    pass


class InvalidTypeException(Exception):
    pass


class NVVMLifter(Lifter):
    def get_transform_passes(self):
        return [
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
        
    def Declare_nvvm(self, name, ret_ty, arg_tys):
        if name not in self.DeviceFuncs:
            func_ty = self.ir.FunctionType(ret_ty, arg_tys)
            func = self.ir.Function(self.llvm_module, func_ty, name)
            self.DeviceFuncs[name] = func

        return self.DeviceFuncs[name]
    
    def AddIntrinsics(self, llvm_module):
        # NVPTX target setup (required for correct addrspace lowering)
        llvm_module.triple = "nvptx64-nvidia-cuda"
        llvm_module.data_layout = "e-i64:64-v16:16-v32:32-n16:32:64"
        nvvm_version_node = llvm_module.add_metadata([self.ir.Constant(self.ir.IntType(32), 2), self.ir.Constant(self.ir.IntType(32), 0)])
        llvm_module.add_named_metadata("nvvmir.version").add(nvvm_version_node)

        self.DeviceFuncs = {}

        # NVVM special register readers
        i32 = self.ir.IntType(32)
        self._nvvm_tid_x = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.tid.x", i32, [])
        self._nvvm_tid_y = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.tid.y", i32, [])
        self._nvvm_tid_z = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.tid.z", i32, [])
        self._nvvm_ntid_x = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.ntid.x", i32, [])
        self._nvvm_ntid_y = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.ntid.y", i32, [])
        self._nvvm_ntid_z = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.ntid.z", i32, [])
        self._nvvm_ctaid_x = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.ctaid.x", i32, [])
        self._nvvm_ctaid_y = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.ctaid.y", i32, [])
        self._nvvm_ctaid_z = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.ctaid.z", i32, [])
        self._nvvm_nctaid_x = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.nctaid.x", i32, [])
        self._nvvm_nctaid_y = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.nctaid.y", i32, [])
        self._nvvm_nctaid_z = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.nctaid.z", i32, [])
        self._nvvm_laneid = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.laneid", i32, [])
        self._nvvm_warpid = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.warpid", i32, [])
        self._nvvm_warpsize = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.warpsize", i32, [])
        self._nvvm_activemask = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.activemask", i32, [])
        self._nvvm_lanemask_eq = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.lanemask.eq", i32, [])
        self._nvvm_lanemask_le = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.lanemask.le", i32, [])
        self._nvvm_lanemask_lt = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.lanemask.lt", i32, [])
        self._nvvm_lanemask_ge = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.lanemask.ge", i32, [])
        self._nvvm_lanemask_gt = self.Declare_nvvm("llvm.nvvm.read.ptx.sreg.lanemask.gt", i32, [])

        # Synchronization and warp-specialized intrinsics
        self._nvvm_barrier0 = self.Declare_nvvm("llvm.nvvm.barrier0", self.ir.VoidType(), [])
        self._nvvm_vote_any_sync = self.Declare_nvvm("llvm.nvvm.vote.any.sync", self.ir.IntType(1), [i32, self.ir.IntType(1)])
        self._nvvm_vote_all_sync = self.Declare_nvvm("llvm.nvvm.vote.all.sync", self.ir.IntType(1), [i32, self.ir.IntType(1)])
        self._nvvm_vote_uni_sync = self.Declare_nvvm("llvm.nvvm.vote.uni.sync", self.ir.IntType(1), [i32, self.ir.IntType(1)])
        self._nvvm_vote_ballot_sync = self.Declare_nvvm("llvm.nvvm.vote.ballot.sync", i32, [i32, self.ir.IntType(1)])
        self._nvvm_match_any_sync_i32 = self.Declare_nvvm("llvm.nvvm.match.any.sync.i32", i32, [i32, i32])
        self._nvvm_match_any_sync_i64 = self.Declare_nvvm("llvm.nvvm.match.any.sync.i64", i32, [i32, self.ir.IntType(64)])
        self._nvvm_brev32 = self.Declare_nvvm("llvm.nvvm.brev32", i32, [i32])
        self._nvvm_brev64 = self.Declare_nvvm("llvm.nvvm.brev64", self.ir.IntType(64), [self.ir.IntType(64)])
        self._nvvm_prmt = self.Declare_nvvm("llvm.nvvm.prmt", i32, [i32, i32, i32])

        # Constant, shared, and local memories in proper NVVM address spaces
        BankTy = self.ir.ArrayType(self.ir.IntType(8), 4096)
        ConstMemTy  = self.ir.ArrayType(BankTy, 5)
        self.ConstMem = self.ir.GlobalVariable(llvm_module, ConstMemTy, "const_mem", 4)
        self.ConstMem.global_constant = True

        SharedArrayTy = self.ir.ArrayType(self.ir.IntType(32), 4096)
        self.SharedMem = self.ir.GlobalVariable(llvm_module, SharedArrayTy, "shared_mem", 3)

        LocalArrayTy = self.ir.ArrayType(self.ir.IntType(8), 4096)
        self.LocalMem = self.ir.GlobalVariable(llvm_module, LocalArrayTy, "local_mem", 5)


        FuncTy = self.ir.FunctionType(
            self.ir.IntType(1),
            [self.ir.PointerType(self.ir.IntType(32), 1), self.ir.IntType(32)],
            False,
        )
        LeaderStore = self.ir.Function(llvm_module, FuncTy, "LeaderStore")
        self.DeviceFuncs["LeaderStore"] = LeaderStore

        FuncTy = self.ir.FunctionType(self.ir.IntType(32), [self.ir.IntType(32)], False)
        AbsFunc = self.ir.Function(llvm_module, FuncTy, "abs")
        self.DeviceFuncs["abs"] = AbsFunc

        FuncTy = self.ir.FunctionType(self.ir.FloatType(), [self.ir.FloatType()], False)
        FabsFunc = self.ir.Function(llvm_module, FuncTy, "fabs")
        self.DeviceFuncs["fabs"] = FabsFunc


    def _lift_function(self, func, llvm_module):
        args = [arg for arg in func.GetArgs(self) if arg.IsArg]

        param_types = [self.GetIRType(arg.TypeDesc) for arg in args]
        func_ty = self.ir.FunctionType(self.ir.VoidType(), param_types, False)
        ir_function = self.ir.Function(llvm_module, func_ty, func.name)
        self._mark_kernel(llvm_module, ir_function)

        func.BlockMap = {}
        func.BuildBBToIRMap(ir_function, func.BlockMap)

        ArgsMap = {}
        for idx, entry in enumerate(args):
            arg_name = entry.GetIRName(self)
            ir_arg = ir_function.args[idx]
            ir_arg.name = arg_name
            ArgsMap[arg_name] = ir_arg

        entry_block = func.BlockMap[func.blocks[0]]
        builder = self.ir.IRBuilder(entry_block)
        builder.position_at_start(entry_block)

        ir_regs = {}

        for bb in func.blocks:
            ir_block = func.BlockMap[bb]
            builder = self.ir.IRBuilder(ir_block)
            self._lift_basic_block(bb, builder, ir_regs, func.BlockMap, ArgsMap)

        for bb in func.blocks:
            builder = self.ir.IRBuilder(func.BlockMap[bb])
            self._populate_phi_nodes(bb, builder, ir_regs, func.BlockMap)

    def _lift_basic_block(self, bb, builder, ir_regs, block_map, ArgsMap):
        for inst in bb.instructions:
            self.lift_instruction(inst, builder, ir_regs, ArgsMap, block_map)

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


    def lift_instruction(self, Inst, IRBuilder: ir.IRBuilder, IRRegs, ArgsMap, BlockMap):
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
                    if op.GetTypeDesc().startswith('F'):
                        val = IRBuilder.call(self.DeviceFuncs["fabs"], [val], f"{name}_fabs")
                    else:
                        val = IRBuilder.call(self.DeviceFuncs["abs"], [val], f"{name}_abs")
                return val
            if op.IsArg:
                return ArgsMap[op.GetIRName(self)]
            if op.IsImmediate:
                return self.ir.Constant(op.GetIRType(self), op.ImmediateValue)
            if op.IsConstMem and not op.IsArg:
                bank = op.ConstMemBank
                offset = op.OffsetOrImm
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
                    self.ir.PointerType(op.GetIRType(self), 4),
                    f"{name}_cmem_ptr" if name else "const_mem_ptr",
                )
                load_name = f"{name}_cmem_load" if name else "const_mem_load"
                return IRBuilder.load(typed_ptr, load_name)
            if op.IsMemAddr and not op.IsReg: # E.g. LDS.U R2 = [0xc]
                return self.ir.Constant(op.GetIRType(self), op.ImmediateValue)
            raise UnsupportedInstructionException(f"Unsupported operand type: {op}")

        def _as_pointer(addr_val, pointee_ty, name, addrspace=0):
            return self._addrspace_pointer(IRBuilder, addr_val, pointee_ty, addrspace, name)

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
            if valop.IsThreadIdxX or (valop.IsThreadIdx and not valop.SpecialRegisterAxis):
                val = IRBuilder.call(self._nvvm_tid_x, [], "tid_x")
            elif valop.IsThreadIdxY:
                val = IRBuilder.call(self._nvvm_tid_y, [], "tid_y")
            elif valop.IsThreadIdxZ:
                val = IRBuilder.call(self._nvvm_tid_z, [], "tid_z")
            elif valop.IsBlockDimX or (valop.IsBlockDim and not valop.SpecialRegisterAxis):
                val = IRBuilder.call(self._nvvm_ntid_x, [], "ntid_x")
            elif valop.IsBlockDimY:
                val = IRBuilder.call(self._nvvm_ntid_y, [], "ntid_y")
            elif valop.IsBlockDimZ:
                val = IRBuilder.call(self._nvvm_ntid_z, [], "ntid_z")
            elif valop.IsBlockIdxX or (valop.IsBlockIdx and not valop.SpecialRegisterAxis):
                val = IRBuilder.call(self._nvvm_ctaid_x, [], "ctaid_x")
            elif valop.IsBlockIdxY:
                val = IRBuilder.call(self._nvvm_ctaid_y, [], "ctaid_y")
            elif valop.IsBlockIdxZ:
                val = IRBuilder.call(self._nvvm_ctaid_z, [], "ctaid_z")
            elif valop.IsGridDimX or (valop.IsGridDim and not valop.SpecialRegisterAxis):
                val = IRBuilder.call(self._nvvm_nctaid_x, [], "nctaid_x")
            elif valop.IsGridDimY:
                val = IRBuilder.call(self._nvvm_nctaid_y, [], "nctaid_y")
            elif valop.IsGridDimZ:
                val = IRBuilder.call(self._nvvm_nctaid_z, [], "nctaid_z")
            elif valop.IsLaneId:
                val = IRBuilder.call(self._nvvm_laneid, [], "laneid")
            elif valop.IsWarpId:
                val = IRBuilder.call(self._nvvm_warpid, [], "warpid")
            elif valop.IsWarpSize:
                val = IRBuilder.call(self._nvvm_warpsize, [], "warpsize")
            elif valop.IsActiveMask:
                val = IRBuilder.call(self._nvvm_activemask, [], "activemask")
            elif valop.IsLaneMaskEQ:
                val = IRBuilder.call(self._nvvm_lanemask_eq, [], "lanemask_eq")
            elif valop.IsLaneMaskLE:
                val = IRBuilder.call(self._nvvm_lanemask_le, [], "lanemask_le")
            elif valop.IsLaneMaskLT:
                val = IRBuilder.call(self._nvvm_lanemask_lt, [], "lanemask_lt")
            elif valop.IsLaneMaskGE:
                val = IRBuilder.call(self._nvvm_lanemask_ge, [], "lanemask_ge")
            elif valop.IsLaneMaskGT:
                val = IRBuilder.call(self._nvvm_lanemask_gt, [], "lanemask_gt")
            else:
                print(f"S2R: Unknown special register {valop.Name}")
                val = self.ir.Constant(self.ir.IntType(32), 0)
            IRRegs[dest.GetIRName(self)] = val
                
        elif opcode == "LDG" or opcode == "LDG64":
            dest = Inst.GetDefs()[0]
            ptr = Inst.GetUses()[0]
            addr = _get_val(ptr, "ldg_addr")
            pointee_ty = dest.GetIRType(self)
            addr_ptr = _as_pointer(
                addr,
                pointee_ty,
                f"{ptr.GetIRName(self)}_addr_ptr" if ptr.IsReg else "ldg_addr_ptr",
                addrspace=1,
            )
            val = IRBuilder.load(addr_ptr, "ldg", typ=pointee_ty)
            IRRegs[dest.GetIRName(self)] = val

        elif opcode == "STG":
            uses = Inst.GetUses()
            ptr, val = uses[0], uses[1]
            addr = _get_val(ptr, "stg_addr")
            v = _get_val(val, "stg_val")
            addr_ptr = _as_pointer(
                addr,
                v.type,
                f"{ptr.GetIRName(self)}_addr_ptr" if ptr.IsReg else "stg_addr_ptr",
                addrspace=1,
            )
            IRBuilder.store(v, addr_ptr)

        elif opcode == "LDS":
            dest = Inst.GetDefs()[0]
            ptr = Inst.GetUses()[0]
            addr = _get_val(ptr, "lds_addr")
            addr = IRBuilder.gep(self.SharedMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "lds_shared_addr")
            if addr.type.pointee != dest.GetIRType(self):
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(dest.GetIRType(self), 3),
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
                    self.ir.PointerType(val.GetIRType(self), 3),
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
                    self.ir.PointerType(val.GetIRType(self), 5),
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
            addr_ptr = _as_pointer(
                addr,
                pointee_ty,
                f"{ptr.GetIRName(self)}_addr_ptr" if ptr.IsReg else "ld_addr_ptr",
                addrspace=1,
            )
            val = IRBuilder.load(addr_ptr, "ld", typ=pointee_ty)
            IRRegs[dest.GetIRName(self)] = val

        elif opcode == "ST":
            uses = Inst.GetUses()
            ptr, val = uses[0], uses[1]
            v = _get_val(val, "st_val")
            addr = _get_val(ptr, "st_addr")
            addr_ptr = _as_pointer(
                addr,
                v.type,
                f"{ptr.GetIRName(self)}_addr_ptr" if ptr.IsReg else "st_addr_ptr",
                addrspace=1,
            )
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
            
        elif opcode == "F2F":
            dest = Inst.GetDefs()[0]
            op1 = Inst.GetUses()[0]
            
            src_type = Inst.opcodes[1]
            dest_type = Inst.opcodes[2]
            
            src_width = int(src_type[1:])
            dest_width = int(dest_type[1:])

            v1 = _get_val(op1, "f2f_src")
            if src_width < dest_width:            
                IRRegs[dest.GetIRName(self)] = IRBuilder.fpext(
                    v1, dest.GetIRType(self), "f2f_ext"
                )
            elif src_width > dest_width:
                IRRegs[dest.GetIRName(self)] = IRBuilder.fptrunc(
                    v1, dest.GetIRType(self), "f2f_trunc"
                )
            else:
                # impossible
                raise UnsupportedInstructionException()
                    
        elif opcode == "MUFU":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            func = Inst._opcodes[1]
            v = _get_val(src, "mufu_src")

            intrinsic_name = None
            if func == "RCP":
                intrinsic_name = "llvm.nvvm.rcp.approx.f"
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

            intrinsic = self.DeviceFuncs.get(intrinsic_name)
            if intrinsic is None:
                func_ty = self.ir.FunctionType(self.ir.FloatType(), [self.ir.FloatType()])
                intrinsic = self.ir.Function(IRBuilder.module, func_ty, intrinsic_name)
                self.DeviceFuncs[intrinsic_name] = intrinsic

            res = IRBuilder.call(intrinsic, [v], f"mufu_{func.lower()}")
            IRRegs[dest.GetIRName(self)] = res
                    
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

                # Determine which special register this is using the new approach
                if ValOp.IsRZ:
                    IRVal = self.ir.Constant(self.ir.IntType(32), 0)
                else:
                    if ValOp.IsThreadIdxX or (ValOp.IsThreadIdx and not ValOp.SpecialRegisterAxis):
                        IRVal = IRBuilder.call(self._nvvm_tid_x, [], "cs2r_tid_x")
                    elif ValOp.IsThreadIdxY:
                        IRVal = IRBuilder.call(self._nvvm_tid_y, [], "cs2r_tid_y")
                    elif ValOp.IsThreadIdxZ:
                        IRVal = IRBuilder.call(self._nvvm_tid_z, [], "cs2r_tid_z")
                    elif ValOp.IsBlockDimX or (ValOp.IsBlockDim and not ValOp.SpecialRegisterAxis):
                        IRVal = IRBuilder.call(self._nvvm_ntid_x, [], "cs2r_ntid_x")
                    elif ValOp.IsBlockDimY:
                        IRVal = IRBuilder.call(self._nvvm_ntid_y, [], "cs2r_ntid_y")
                    elif ValOp.IsBlockDimZ:
                        IRVal = IRBuilder.call(self._nvvm_ntid_z, [], "cs2r_ntid_z")
                    elif ValOp.IsBlockIdxX or (ValOp.IsBlockIdx and not ValOp.SpecialRegisterAxis):
                        IRVal = IRBuilder.call(self._nvvm_ctaid_x, [], "cs2r_ctaid_x")
                    elif ValOp.IsBlockIdxY:
                        IRVal = IRBuilder.call(self._nvvm_ctaid_y, [], "cs2r_ctaid_y")
                    elif ValOp.IsBlockIdxZ:
                        IRVal = IRBuilder.call(self._nvvm_ctaid_z, [], "cs2r_ctaid_z")
                    elif ValOp.IsGridDimX or (ValOp.IsGridDim and not ValOp.SpecialRegisterAxis):
                        IRVal = IRBuilder.call(self._nvvm_nctaid_x, [], "cs2r_nctaid_x")
                    elif ValOp.IsGridDimY:
                        IRVal = IRBuilder.call(self._nvvm_nctaid_y, [], "cs2r_nctaid_y")
                    elif ValOp.IsGridDimZ:
                        IRVal = IRBuilder.call(self._nvvm_nctaid_z, [], "cs2r_nctaid_z")
                    elif ValOp.IsLaneId:
                        IRVal = IRBuilder.call(self._nvvm_laneid, [], "cs2r_lane")
                    elif ValOp.IsWarpId:
                        IRVal = IRBuilder.call(self._nvvm_warpid, [], "cs2r_warp")
                    elif ValOp.IsWarpSize:
                        IRVal = IRBuilder.call(self._nvvm_warpsize, [], "cs2r_warpsize")
                    elif ValOp.IsActiveMask:
                        IRVal = IRBuilder.call(self._nvvm_activemask, [], "cs2r_activemask")
                    elif ValOp.IsLaneMaskEQ:
                        IRVal = IRBuilder.call(self._nvvm_lanemask_eq, [], "cs2r_lanemask_eq")
                    elif ValOp.IsLaneMaskLE:
                        IRVal = IRBuilder.call(self._nvvm_lanemask_le, [], "cs2r_lanemask_le")
                    elif ValOp.IsLaneMaskLT:
                        IRVal = IRBuilder.call(self._nvvm_lanemask_lt, [], "cs2r_lanemask_lt")
                    elif ValOp.IsLaneMaskGE:
                        IRVal = IRBuilder.call(self._nvvm_lanemask_ge, [], "cs2r_lanemask_ge")
                    elif ValOp.IsLaneMaskGT:
                        IRVal = IRBuilder.call(self._nvvm_lanemask_gt, [], "cs2r_lanemask_gt")
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

            IRBuilder.call(self._nvvm_barrier0, [], "barrier0")

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
            mask_op, pred_op = Inst.GetUses()[0], Inst.GetUses()[1]

            pred_val = _get_val(pred_op, "vote_pred")
            mask_val = _get_val(mask_op, "vote_mask")
            
            if mask_op.IsPT and not mask_op.IsNotReg:
                mask_val = self.ir.Constant(self.ir.IntType(32), -1)
            else:
                mask_val = _get_val(mask_op, "vote_mask")

            mode = Inst.opcodes[1].upper()

            if mode == "ANY":
                vote_val = IRBuilder.call(
                    self._nvvm_vote_any_sync,
                    [mask_val, pred_val],
                    "vote_any",
                )
            elif mode == "ALL":
                vote_val = IRBuilder.call(
                    self._nvvm_vote_all_sync,
                    [mask_val, pred_val],
                    "vote_all",
                )
            elif mode == "UNI":
                vote_val = IRBuilder.call(
                    self._nvvm_vote_uni_sync,
                    [mask_val, pred_val],
                    "vote_uni",
                )
            elif mode == "BALLOT":
                vote_val = IRBuilder.call(
                    self._nvvm_vote_ballot_sync,
                    [mask_val, pred_val],
                    "vote_ballot",
                )
            else:
                raise UnsupportedInstructionException

            dest_ty = dest.GetIRType(self)
            vote_ty = vote_val.type
            if dest_ty != vote_ty:
                if hasattr(dest_ty, "width") and hasattr(vote_ty, "width"):
                    if dest_ty.width > vote_ty.width:
                        vote_val = IRBuilder.zext(vote_val, dest_ty, f"vote_{mode.lower()}_zext")
                    else:
                        vote_val = IRBuilder.trunc(vote_val, dest_ty, f"vote_{mode.lower()}_trunc")
                else:
                    vote_val = IRBuilder.bitcast(vote_val, dest_ty, f"vote_{mode.lower()}_cast")

            IRRegs[dest.GetIRName(self)] = vote_val
            
        elif opcode == "SHFL":
            # SHFL.<MODE> PT, R3 = R0, 0x8, 0x1f
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
            
            func_name = "llvm.nvvm.shfl.sync." + mode.lower() + "." + dtype.intrinsic_name

            if mode == "DOWN":
                shfl_intr = self.Declare_nvvm(func_name, dtype, [i32, dtype, i32, i32])
                shfl_name = "shfl_down"
            elif mode == "UP":
                shfl_intr = self.Declare_nvvm(func_name, dtype, [i32, dtype, i32, i32])
                shfl_name = "shfl_up"
            elif mode == "BFLY":
                shfl_intr = self.Declare_nvvm(func_name, dtype, [i32, dtype, i32, i32])
                shfl_name = "shfl_bfly"
            elif mode == "IDX":
                shfl_intr = self.Declare_nvvm(func_name, dtype, [i32, dtype, i32, i32])
                shfl_name = "shfl_idx"
            else:
                raise UnsupportedInstructionException

            shflVal = IRBuilder.call(
                shfl_intr,
                [mask_const, val, off, wid],
                shfl_name,
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
                dtype = self.ir.IntType(64)
            elif type == 'U32':
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
                    self._nvvm_match_any_sync_i64,
                    [mask_const, val],
                    "match_any_i64",
                )
            else:
                matchVal = IRBuilder.call(
                    self._nvvm_match_any_sync_i32,
                    [mask_const, val],
                    "match_any_i32",
                )

            if dest.GetIRType(self) != matchVal.type:
                matchVal = IRBuilder.zext(matchVal, dest.GetIRType(self), "match_any_ext")

            IRRegs[dest.GetIRName(self)] = matchVal

        elif opcode == "BREV" or opcode == "UBREV":
            dest = Inst.GetDefs()[0]
            src = Inst.GetUses()[0]
            val = _get_val(src, "brev_val")

            dest_ty = dest.GetIRType(self)
            if dest_ty.width == 64:
                if val.type != self.ir.IntType(64):
                    if hasattr(val.type, "width") and val.type.width > 64:
                        val = IRBuilder.trunc(val, self.ir.IntType(64), "brev64_val")
                    else:
                        val = IRBuilder.zext(val, self.ir.IntType(64), "brev64_val")
                revVal = IRBuilder.call(self._nvvm_brev64, [val], "brev64")
            else:
                if val.type != self.ir.IntType(32):
                    if hasattr(val.type, "width") and val.type.width > 32:
                        val = IRBuilder.trunc(val, self.ir.IntType(32), "brev32_val")
                    else:
                        val = IRBuilder.zext(val, self.ir.IntType(32), "brev32_val")
                revVal = IRBuilder.call(self._nvvm_brev32, [val], "brev32")

            if revVal.type != dest_ty:
                if dest_ty.width < revVal.type.width:
                    revVal = IRBuilder.trunc(revVal, dest_ty, "brev_trunc")
                elif dest_ty.width > revVal.type.width:
                    revVal = IRBuilder.zext(revVal, dest_ty, "brev_zext")

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
                FuncTy = ir.FunctionType(ir.IntType(32), [dtype, ir.IntType(1)], False)
                self.DeviceFuncs[funcName] = ir.Function(IRBuilder.module, FuncTy, funcName)

            floVal = IRBuilder.call(
                self.DeviceFuncs[funcName],
                [val, ir.Constant(ir.IntType(1), 0)],
                "flo",
            )
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
                raise UnsupportedInstructionException()
            
            if not Inst.opcodes[3].startswith("F"):
                raise UnsupportedInstructionException()
            if "STRONG" not in modifiers:
                raise UnsupportedInstructionException()

            uses = Inst.GetUses()

            dest_reg = None
            if uses[0].IsReg and not uses[0].IsMemAddr:
                dest_reg = uses[0]

            mem_ops = [op for op in uses if op.IsMemAddr]
            ptr_op = mem_ops[0]
            addr_val = _get_val(ptr_op, "atom_addr")
            float_ty = self.ir.FloatType()
            ptr = _as_pointer(
                addr_val,
                float_ty,
                f"{ptr_op.GetIRName(self)}_atom_addr" if ptr_op.IsReg else "atom_addr_ptr",
                addrspace=1,
            )

            add_src = uses[-1]

            add_val = _get_val(add_src, "atom_add_val")

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

            uses = Inst.GetUses()
            if atomic_kind == "cas":
                if len(uses) < 3:
                    raise UnsupportedInstructionException("CAS requires pointer, compare, and value operands")
                value_operands = [uses[1], uses[2]]
            else:
                if len(uses) < 2:
                    raise UnsupportedInstructionException(f"{opcode} requires at least two operands")
                value_operands = [uses[1]]

            value_values = []
            for idx, op in enumerate(value_operands):
                value = _get_val(op, f"atomic_val_{idx}")
                if idx == 0:
                    elem_ty = value.type
                else:
                    value = cast_value(value, elem_ty, f"atomic_{atomic_kind}_arg{idx}")
                value_values.append(value)

            is_shared = opcode == "ATOMS"
            scope = "cta" if is_shared else "sys"
            if "SYS" in tokens:
                scope = "sys"
            elif "CTA" in tokens:
                scope = "cta"

            ptr_operand = uses[0]
            if is_shared:
                addr_index = _get_val(ptr_operand, "atoms_addr")
                ptr = IRBuilder.gep(
                    self.SharedMem,
                    [self.ir.Constant(self.ir.IntType(32), 0), addr_index],
                    "atoms_shared_ptr",
                )
            else:
                addr_val = _get_val(ptr_operand, "atomg_addr")
                ptr_name = ptr_operand.GetIRName(self) if hasattr(ptr_operand, "GetIRName") else "atomg_addr_ptr"
                ptr = _as_pointer(
                    addr_val,
                    elem_ty,
                    ptr_name if ptr_operand.IsReg else "atomg_addr_ptr",
                    addrspace=1,
                )

            original_elem_ty = elem_ty
            is_float = isinstance(elem_ty, ir.HalfType) or isinstance(elem_ty, ir.FloatType) or isinstance(elem_ty, ir.DoubleType)
            intrinsic_name = None
            if atomic_kind == "cas" and is_float:
                int_ty = float_as_int_ty(elem_ty)
                elem_ty = int_ty
                value_values = [IRBuilder.bitcast(value_values[0], int_ty, "atomic_cas_cmp_int"), IRBuilder.bitcast(value_values[1], int_ty, "atomic_cas_val_int")]
                is_float = False

            ptr = self._addrspace_pointer(IRBuilder, ptr, elem_ty, getattr(ptr.type, "address_space", 0), "atomic_ptr_cast")
            ptr_addrspace = getattr(ptr.type, "address_space", 0)

            if atomic_kind == "add":
                suffix = "f" if is_float else "i"
                intrinsic_name = f"llvm.nvvm.atomic.add.gen.{suffix}.{scope}"
            elif atomic_kind == "and":
                intrinsic_name = f"llvm.nvvm.atomic.and.gen.i.{scope}"
            elif atomic_kind == "or":
                intrinsic_name = f"llvm.nvvm.atomic.or.gen.i.{scope}"
            elif atomic_kind == "xor":
                intrinsic_name = f"llvm.nvvm.atomic.xor.gen.i.{scope}"
            elif atomic_kind == "max":
                intrinsic_name = f"llvm.nvvm.atomic.max.gen.i.{scope}"
            elif atomic_kind == "min":
                intrinsic_name = f"llvm.nvvm.atomic.min.gen.i.{scope}"
            elif atomic_kind == "exch":
                intrinsic_name = f"llvm.nvvm.atomic.exch.gen.i.{scope}"
            elif atomic_kind == "inc":
                intrinsic_name = f"llvm.nvvm.atomic.inc.gen.i.{scope}"
            elif atomic_kind == "dec":
                intrinsic_name = f"llvm.nvvm.atomic.dec.gen.i.{scope}"
            elif atomic_kind == "cas":
                intrinsic_name = f"llvm.nvvm.atomic.cas.gen.i.{scope}"
            else:
                raise UnsupportedInstructionException(f"Unhandled atomic operation {atomic_kind}")

            cache = getattr(self, "_atomic_intrinsic_cache", None)
            if cache is None:
                cache = {}
                self._atomic_intrinsic_cache = cache
            key = (intrinsic_name, str(elem_ty), ptr_addrspace, len(value_values))
            func = cache.get(key)
            if func is None:
                ptr_ty = self.ir.PointerType(elem_ty, ptr_addrspace)
                arg_tys = [ptr_ty] + [elem_ty] * len(value_values)
                func_ty = self.ir.FunctionType(elem_ty, arg_tys, False)
                func = ir.Function(IRBuilder.module, func_ty, intrinsic_name)
                cache[key] = func
            call_args = [ptr] + value_values
            result = IRBuilder.call(func, call_args, f"atomic_{atomic_kind}")

            dest = Inst.GetDefs()[0]
            final_value = result
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

            prmtVal = IRBuilder.call(self._nvvm_prmt, [v1, v2, imm8], "prmt")
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
            
        elif opcode == "QSPC":
            print("QSPC met; don't know what to do ")
            pass

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


Lifter = NVVMLifter
