from llvmlite import ir, binding
import llvmlite.binding as llvm
import llvmlite

from transform.sr_substitute import SR_TO_OFFSET

class UnsupportedOperatorException(Exception):
    pass

class UnsupportedInstructionException(Exception):
    pass

class InvalidTypeException(Exception):
    pass

class Lifter :
    def __init__(self):
        # Initialize LLVM environment
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        pkg_version = getattr(llvmlite, "__version__", None)
        llvm_ver = getattr(llvm, "llvm_version_info",
                        getattr(llvm, "llvm_version", None))

        print("llvmlite package version:", pkg_version)
        if llvm_ver is not None:
            print("LLVM version string:", ".".join(map(str, llvm_ver)))
        print("")

        self.ir = ir
        self.lift_errors = []
        
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
        ConstArrayTy =  self.ir.ArrayType(self.ir.IntType(8), 4096)
        self.ConstMem = self.ir.GlobalVariable(llvm_module, ConstArrayTy, "const_mem")
        SharedArrayTy =  self.ir.ArrayType(self.ir.IntType(32), 49152)
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

    def LiftModule(self, module, outfile):
        llvm_module = self.ir.Module(module.name)

        self.AddIntrinsics(llvm_module)

        for func in module.functions:
            self._lift_function(func, llvm_module)

        print(llvm_module)
        print(llvm_module, file=outfile)

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
                    if ir_name not in ir_regs:
                        val = rough_search(op)
                    else:
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
                return self.ir.Constant(op.GetIRType(self), 1)
            if op.IsReg:
                irName = op.GetIRName(self)
                if irName not in IRRegs:
                    val = roughSearch(op)
                else:
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
            v1 = _get_val(uses[0], "shf_lhs")
            v2 = _get_val(uses[1], "shf_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.shl(v1, v2, "shf")
                
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
            if valop.IsThreadIdx:
                val = IRBuilder.call(self.GetThreadIdx, [], "ThreadIdx")
            elif valop.IsBlockDim:
                val = IRBuilder.call(self.GetBlockDim, [], "BlockDim")
            elif valop.IsBlockIdx:
                val = IRBuilder.call(self.GetBlockIdx, [], "BlockIdx")
            elif valop.IsLaneId:
                val = IRBuilder.call(self.GetLaneId, [], "LaneId")
            elif valop.IsWarpId:
                val = IRBuilder.call(self.GetWarpId, [], "WarpId")
            else:
                print(f"S2R: Unknown special register {valop.Name}")
                val = self.ir.Constant(self.ir.IntType(32), 0)
            IRRegs[dest.GetIRName(self)] = val
                
        elif opcode == "LDG" or opcode == "LDG64":
            dest = Inst.GetDefs()[0]
            ptr = Inst.GetUses()[0]
            addr = _get_val(ptr, "ldg_addr")
            val = IRBuilder.load(addr, "ldg", typ=dest.GetIRType(self))
            IRRegs[dest.GetIRName(self)] = val
                
        elif opcode == "STG":
            uses = Inst.GetUses()
            ptr, val = uses[0], uses[1]
            addr = _get_val(ptr, "stg_addr")
            v = _get_val(val, "stg_val")
            IRBuilder.store(v, addr)

        elif opcode == "LDS":
            dest = Inst.GetDefs()[0]
            ptr = Inst.GetUses()[0]
            addr = _get_val(ptr, "lds_addr")
            addr = IRBuilder.gep(self.SharedMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "lds_shared_addr")
            val = IRBuilder.load(addr, "lds", typ=dest.GetIRType(self))
            IRRegs[dest.GetIRName(self)] = val

        elif opcode == "STS":
            uses = Inst.GetUses()
            ptr, val = uses[0], uses[1]
            addr = _get_val(ptr, "sts_addr")
            addr = IRBuilder.gep(self.SharedMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "sts_shared_addr")
            if addr.type.pointee != val.GetIRType(self):
                addr = IRBuilder.bitcast(addr, self.ir.PointerType(val.GetIRType(self)), "sts_shared_addr_cast")
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
                addr = IRBuilder.bitcast(addr, self.ir.PointerType(val.GetIRType(self)), "stl_local_addr_cast")
            v = _get_val(val, "stl_val")
            IRBuilder.store(v, addr)

        elif opcode == "FMUL":
            dest = Inst.GetDefs()[0]
            uses = Inst.GetUses()
            v1 = _get_val(uses[0], "fmul_lhs")
            v2 = _get_val(uses[1], "fmul_rhs")
            IRRegs[dest.GetIRName(self)] = IRBuilder.fmul(v1, v2, "fmul")

        elif opcode == "INTTOPTR": # psudo instruction placed by # transform/inttoptr.py
            ptr = Inst.GetDefs()[0]
            val = Inst.GetUses()[0]
            v1 = _get_val(val, "inttoptr_val")

            IRRegs[ptr.GetIRName(self)] = IRBuilder.inttoptr(v1, ptr.GetIRType(self), "inttoptr")


        elif opcode == "FFMA":
            raise UnsupportedInstructionException

        elif opcode == "LD":
            dest = Inst.GetDefs()[0]
            ptr = Inst.GetUses()[0]
            addr = _get_val(ptr, "ld_addr")
            val = IRBuilder.load(addr, "ld")
            IRRegs[dest.GetIRName(self)] = val
                
        elif opcode == "ST":
            uses = Inst.GetUses()
            ptr, val = uses[0], uses[1]
            addr = _get_val(ptr, "st_addr")
            v = _get_val(val, "st_val")
            IRBuilder.store(v, addr)
        
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
            # Lower LOP3.LUT for both register and predicate destinations.
            dest = Inst.GetDefs()[0]
            src1, src2, src3 = Inst.GetUses()[0], Inst.GetUses()[1], Inst.GetUses()[2]
            func = Inst._opcodes[1] if len(Inst._opcodes) > 1 else None

            if func != "LUT":
                raise UnsupportedInstructionException

            # Find the LUT immediate among uses (last immediate before any PT operand)
            imm8 = None
            for op in reversed(Inst.GetUses()):
                if op.IsImmediate:
                    imm8 = op.ImmediateValue & 0xFF
                    break
            if imm8 is None:
                raise UnsupportedInstructionException

            a = _get_val(src1, "lop3_a")
            b = _get_val(src2, "lop3_b")
            c = _get_val(src3, "lop3_c")

            # Sum-of-products bitwise construction for 32-bit result
            zero = self.ir.Constant(b.type, 0)
            # Fast-path a-agnostic mask used frequently: imm8 == 0xC0 -> b & c
            if imm8 == 0xC0:
                res32 = IRBuilder.and_(b, c, "lop3_bc")
            else:
                nota = IRBuilder.not_(a, "lop3_nota")
                notb = IRBuilder.not_(b, "lop3_notb")
                notc = IRBuilder.not_(c, "lop3_notc")
                res32 = zero
                for idx in range(8):
                    if ((imm8 >> idx) & 1) == 0:
                        continue
                    xa = a if (idx & 1) else nota
                    xb = b if (idx & 2) else notb
                    xc = c if (idx & 4) else notc
                    tmp = IRBuilder.and_(xa, xb, f"lop3_and_ab_{idx}")
                    tmp = IRBuilder.and_(tmp, xc, f"lop3_and_abc_{idx}")
                    res32 = IRBuilder.or_(res32, tmp, f"lop3_or_{idx}")

            if dest.IsPredicateReg:
                # For P-dest, prefer a safe minimal lowering:
                #  - Recognize imm8==0xC0 -> (B & C) != 0 (matches loop3).
                #  - Otherwise, approximate with 1-bit LUT lookup of (LSB(A),LSB(B),LSB(C)).
                uses = Inst.GetUses()
                c_is_imm = len(uses) >= 3 and uses[2].IsImmediate
                if imm8 == 0xC0 and c_is_imm:
                    mask = IRBuilder.and_(b, c, "lop3_bc_mask")
                    pred = IRBuilder.icmp_unsigned("!=", mask, zero, "lop3_pred")
                    IRRegs[dest.GetIRName(self)] = pred
                else:
                    one_i32 = self.ir.Constant(a.type, 1)
                    a_lsb_i32 = IRBuilder.and_(a, one_i32, "lop3_a_lsb")
                    b_lsb_i32 = IRBuilder.and_(b, one_i32, "lop3_b_lsb")
                    c_lsb_i32 = IRBuilder.and_((c if c.type == a.type else IRBuilder.zext(c, a.type, "lop3_c_zext")), one_i32, "lop3_c_lsb")
                    a0 = IRBuilder.icmp_unsigned("!=", a_lsb_i32, self.ir.Constant(a.type, 0), "lop3_a0")
                    b0 = IRBuilder.icmp_unsigned("!=", b_lsb_i32, self.ir.Constant(b.type, 0), "lop3_b0")
                    c0 = IRBuilder.icmp_unsigned("!=", c_lsb_i32, self.ir.Constant(a.type, 0), "lop3_c0")
                    a0_i32 = IRBuilder.sext(a0, a.type)
                    b0_i32 = IRBuilder.sext(b0, b.type)
                    c0_i32 = IRBuilder.sext(c0, a.type)
                    idx = IRBuilder.or_(
                        a0_i32,
                        IRBuilder.or_(
                            IRBuilder.shl(b0_i32, self.ir.Constant(b.type, 1)),
                            IRBuilder.shl(c0_i32, self.ir.Constant(a.type, 2))
                        ),
                        "lop3_idx"
                    )
                    imm_i32 = self.ir.Constant(a.type, imm8 & 0xFF)
                    bit_i32 = IRBuilder.and_(
                        IRBuilder.lshr(imm_i32, idx, "lop3_lut_shift"),
                        self.ir.Constant(a.type, 1),
                        "lop3_lut_bit"
                    )
                    pred = IRBuilder.icmp_unsigned("!=", bit_i32, self.ir.Constant(a.type, 0), "lop3_pred")
                    IRRegs[dest.GetIRName(self)] = pred
            else:
                IRRegs[dest.GetIRName(self)] = res32


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
            val = _get_val(src, "ldc")
            IRRegs[dest.GetIRName(self)] = val
        
        elif opcode == "CS2R":
            # CS2R (Convert Special Register to Register)
            ResOp = Inst.GetDefs()[0]
            ValOp = Inst.GetUses()[0]
            if ResOp.IsReg:
                IRResOp = IRRegs[ResOp.GetIRName(self)]
                
                # Determine which special register this is using the new approach
                if ValOp.IsThreadIdx:
                    IRVal = IRBuilder.call(self.GetThreadIdx, [], "cs2r_tid")
                elif ValOp.IsBlockDim:
                    IRVal = IRBuilder.call(self.GetBlockDim, [], "cs2r_ntid")
                elif ValOp.IsBlockIdx:
                    IRVal = IRBuilder.call(self.GetBlockIdx, [], "cs2r_ctaid")
                elif ValOp.IsLaneId:
                    IRVal = IRBuilder.call(self.GetLaneId, [], "cs2r_lane")
                elif ValOp.IsWarpId:
                    IRVal = IRBuilder.call(self.GetWarpId, [], "cs2r_warp")
                elif ValOp.IsRZ:
                    IRVal = ir.Constant(ir.IntType(32), 0)
                else:
                    print(f"CS2R: Unknown special register {ValOp}")
                    IRVal = ir.Constant(ir.IntType(32), 0)
                
                IRBuilder.store(IRVal, IRResOp)

        elif opcode == "ISETP" or opcode == "ISETP64" or opcode == "FSETP" or opcode == "UISETP":
            uses = Inst.GetUses()
            # Some encodings include a leading PT predicate use; skip it
            start_idx = 1 if len(uses) > 0 and getattr(uses[0], 'IsPT', False) else 0
            r = _get_val(uses[start_idx], "branch_operand_0")

            # Remove U32 from opcodes
            # Currently just assuming every int are signed. May be dangerous?  
            opcodes = [opcode for opcode in Inst._opcodes if opcode != "U32"]
            isUnsigned = "U32" in Inst._opcodes

            for i in range(1, len(opcodes)):
                temp = _get_val(uses[start_idx + i], f"branch_operand_{i}")
                if opcodes[i] == "AND":
                    r = IRBuilder.and_(r, temp, f"branch_and_{i}")
                elif opcodes[i] == "OR":
                    r = IRBuilder.or_(r, temp, f"branch_or_{i}")
                elif opcodes[i] == "XOR":
                    r = IRBuilder.xor(r, temp, f"branch_xor_{i}")
                elif opcodes[i] == "EX":
                    pass #TODO:?
                else:
                    if isUnsigned:
                        r = IRBuilder.icmp_unsigned(self.GetCmpOp(opcodes[i]), r, temp, f"branch_cmp_{i}")
                    else:
                        r = IRBuilder.icmp_signed(self.GetCmpOp(opcodes[i]), r, temp, f"branch_cmp_{i}")

            pred = Inst.GetDefs()[0]
            IRRegs[pred.GetIRName(self)] = r

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
            pred = _get_val(Inst.GetUses()[0])
            mask = _get_val(Inst.GetUses()[1])
            
            mode = Inst.opcodes[1]
            
            if mode == "ANY":
                funcName = "vote_any"
                if funcName not in self.DeviceFuncs:
                    FuncTy = ir.FunctionType(ir.IntType(32), [ir.IntType(1), ir.IntType(1)], False)
                    self.DeviceFuncs[funcName] = ir.Function(IRBuilder.module, FuncTy, funcName)
                voteVal = IRBuilder.call(self.DeviceFuncs[funcName], [pred, mask], "vote_any")
            else:
                raise UnsupportedInstructionException

            IRRegs[dest.GetIRName(self)] = voteVal
            
        elif opcode == "SHFL":
            # SHFL.DOWN PT, R3 = R0, 0x8, 0x1f
            destPred = Inst.GetDefs()[0]
            destReg = Inst.GetDefs()[1]
            srcReg = Inst.GetUses()[0]
            offset = Inst.GetUses()[1]
            width = Inst.GetUses()[2]
            
            val = _get_val(srcReg, "shfl_val")
            off = _get_val(offset, "shfl_offset")
            wid = _get_val(width, "shfl_width")
            
            mode = Inst.opcodes[1]
            
            # TODO: handle different modes
            funcName = "shfl_down"
            if funcName not in self.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(32), ir.IntType(32)], False)
                self.DeviceFuncs[funcName] = ir.Function(IRBuilder.module, FuncTy, funcName)
            shflVal = IRBuilder.call(self.DeviceFuncs[funcName], [val, off, wid], "shfl_down")
            
            IRRegs[destReg.GetIRName(self)] = shflVal
            # TODO: set predicate properly
            IRRegs[destPred.GetIRName(self)] = ir.Constant(ir.IntType(1), 1)  # true predicate

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

        elif opcode == "RED" or opcode == "ATOMG" or opcode == "ATOMS":
            # TODO: incorrect impl
            uses = Inst.GetUses()
            src1, src2 = uses[0], uses[1]
            val1 = _get_val(src1)
            val2 = _get_val(src2)

            mode = Inst.opcodes[2]
            
            order = 'seq_cst'
            res = IRBuilder.atomic_rmw(mode, val1, val2, order)

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
            return self.ir.PointerType(self.ir.IntType(32))
        elif TypeDesc == "Float32_PTR":
            return self.ir.PointerType(self.ir.FloatType())
        elif TypeDesc == "Int64":
            return self.ir.IntType(64)
        elif TypeDesc == "Int64_PTR":
            return self.ir.PointerType(self.ir.IntType(64))
        elif TypeDesc == "Int1":
            return self.ir.IntType(1)
        elif TypeDesc == "PTR":
            return self.ir.PointerType(self.ir.IntType(32))
        elif TypeDesc == "NOTYPE":
            return self.ir.IntType(32) # Fallback to Int32

        raise ValueError(f"Unknown type: {TypeDesc}")

    def Shutdown(self):
        # Cleanup LLVM environment
        binding.shutdown()
