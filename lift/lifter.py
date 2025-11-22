import contextlib
import inspect
import io
import re
from typing import Dict, List

import llvmlite
import llvmlite.binding as llvm
from llvmlite import binding, ir

from transform.transforms import Transforms


def lift_for(*opcodes):
    def decorator(func):
        func._lift_opcodes = tuple(opcodes)
        return func

    return decorator


class UnsupportedInstructionException(Exception):
    pass


class Lifter:
    def __init__(self, elf = None, verbose: bool = False) -> None:
        self._verbose = verbose
        pkg_version = getattr(llvmlite, "__version__", None)
        llvm_ver = getattr(
            llvm,
            "llvm_version_info",
            getattr(llvm, "llvm_version", None),
        )

        if self._verbose:
            print("llvmlite package version:", pkg_version)
            if llvm_ver is not None:
                print("LLVM version string:", ".".join(map(str, llvm_ver)))
            print("")
            
        self.arg_offsets, self.cm_map = self._parse_elf(elf)

        self.ir = ir
        self.lift_errors = []
        self.intrinsics = {}
        self.liftMap = self._build_lift_map()
        
    def _parse_elf(self, text):
        
        if text is None:
            return None, None
        
        arg_offsets = {}
        cm_map = {}

        # parse .nv.info.<kernel> for EIATTR_KPARAM_INFO
        info_pattern = re.compile(
            r'\.nv\.info\.([^\s]+)\s*(.*?)(?=\n\.)',
            re.DOTALL
        )

        for m in info_pattern.finditer(text):
            kernel = m.group(1)
            body = m.group(2)
            
            #   Attribute: EIATTR_PARAM_CBANK
            #   Format:   EIFMT_SVAL
            #   Value:    0x2 0x1c0160
            param_base = 0
            cbank_match = re.search(
                r'Attribute:\s*EIATTR_PARAM_CBANK.*?Value:\s*0x[0-9a-fA-F]+\s+0x([0-9a-fA-F]+)',
                body,
                re.DOTALL
            )
            if cbank_match:
                full_val = int(cbank_match.group(1), 16)
                # Take low 16 bits, e.g. 0x1c0160 -> 0x0160
                param_base = full_val & 0xFFFF

            pairs = []
            for om in re.finditer(
                r'Ordinal\s*:\s*0x([0-9a-fA-F]+).*?Offset\s*:\s*0x([0-9a-fA-F]+)',
                body,
                re.DOTALL
            ):
                ordinal = int(om.group(1), 16)
                offset = int(om.group(2), 16)
                pairs.append((ordinal, offset))

            if pairs:
                pairs.sort()
                max_ord = pairs[-1][0]
                lst = [None] * (max_ord + 1)
                for ord_, off in pairs:
                    lst[ord_] = off + param_base
                arg_offsets[kernel] = lst

        # parse .nv.constant{bank}.<kernel>
        const_pattern = re.compile(
            r'\.nv\.constant([0-9]+)\.([^\s]+)\s*(.*?)(?=\n\.)',
            re.DOTALL
        )

        for m in const_pattern.finditer(text):
            bank = int(m.group(1))
            kernel = m.group(2)
            body = m.group(3)

            # Extract 32-bit words
            words = [int(h, 16) for h in re.findall(r'0x[0-9a-fA-F]+', body)]

            if kernel not in cm_map:
                cm_map[kernel] = {}
            cm_map[kernel][bank] = words

        return arg_offsets, cm_map


    def _build_lift_map(self):
        lift_map = {}
        for cls in reversed(self.__class__.mro()):
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                opcodes = getattr(method, "_lift_opcodes", None)
                if not opcodes:
                    continue
                bound_method = getattr(self, name)
                for opcode in opcodes:
                    lift_map[opcode] = bound_method
        return lift_map

    def lift_module(self, module, outfile):
        self.passes = self.get_transform_passes()
        transforms = Transforms(self.passes, verbose=self._verbose)
        if not self._verbose:
            with contextlib.redirect_stdout(io.StringIO()):
                transforms.apply(module)
        else:
            transforms.apply(module)

        self.llvm_module = self.ir.Module(module.name)
        self.intrinsics = {}

        self.setup_module(self.llvm_module)

        for func in module.functions:
            self.curr_function = func
            self._lift_function(func, self.llvm_module)

        output_ir = str(self.llvm_module)

        output_ir = self.postprocess_ir(output_ir)

        if self._verbose:
            print(output_ir)
        print(output_ir, file=outfile)

    def postprocess_ir(self, ir_code):
        return ir_code

    def setup_module(self, llvm_module):
        raise NotImplementedError

    def get_intrinsic(self, name, ret_ty=None, arg_tys=None, elem_ty=None, ptr_addrspace=None):
        if name in self.intrinsics:
            return self.intrinsics[name]
        intrinsic = self.build_intrinsic(
            name,
            ret_ty=ret_ty,
            arg_tys=arg_tys,
            elem_ty=elem_ty,
            ptr_addrspace=ptr_addrspace,
        )
        self.intrinsics[name] = intrinsic
        return intrinsic

    def build_intrinsic(self, name, ret_ty=None, arg_tys=None, elem_ty=None, ptr_addrspace=None):
        raise NotImplementedError

    def get_transform_passes(self):
        raise NotImplementedError

    def get_cmp_op(self, opcode):
        if opcode in {"GE", "GEU"}:
            return ">="
        if opcode == "EQ":
            return "=="
        if opcode in {"NE", "NEU"}:
            return "!="
        if opcode in {"LE", "LEU"}:
            return "<="
        if opcode in {"GT", "GTU"}:
            return ">"
        if opcode in {"LT", "LTU"}:
            return "<"

        return ""

    def _lift_basic_block(self, bb, builder, ir_regs, block_map, const_mem):
        for inst in bb.instructions:
            self.lift_instruction(inst, builder, ir_regs, const_mem, block_map)

    def _populate_phi_nodes(self, bb, builder, ir_regs, block_map):
        ir_block = block_map[bb]

        for idx, inst in enumerate(bb.instructions):
            if inst.opcodes[0] not in {"PHI", "PHI64"}:
                continue

            ir_inst = ir_block.instructions[idx]

            for pred_idx, op in enumerate(inst.operands[1:]):
                pred_bb = bb.preds[pred_idx]
                if op.is_rz:
                    val = self.ir.Constant(op.get_ir_type(self), 0)
                elif op.is_pt:
                    val = self.ir.Constant(op.get_ir_type(self), 1)
                else:
                    ir_name = op.get_ir_name(self)
                    val = ir_regs[ir_name]
                ir_inst.add_incoming(val, block_map[pred_bb])

    @lift_for("MOV", "MOV64", "UMOV")
    def lift_mov(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src, "mov")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("S2UR")
    def lift_s2ur(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src, "s2ur")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("MOV32I")
    def lift_mov32i(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        if not src.is_immediate:
            raise UnsupportedInstructionException(f"MOV32I expects immediate, got: {src}")
        val = self.ir.Constant(src.get_ir_type(self), src.immediate_value)
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("SETZERO")
    def lift_setzero(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        zero_val = self.ir.Constant(dest.get_ir_type(self), 0)
        self._set_val(IRBuilder, IRRegs, dest, zero_val)

    @lift_for("EXIT")
    def lift_exit(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        IRBuilder.ret_void()

    @lift_for("PHI", "PHI64")
    def lift_phi(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        phi_val = IRBuilder.phi(dest.get_ir_type(self), "phi")
        self._set_val(IRBuilder, IRRegs, dest, phi_val)

    @lift_for("IMAD", "UIMAD")
    def lift_imad(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "imad_lhs")
        v2 = _get_val(uses[1], "imad_rhs")
        v3 = _get_val(uses[2], "imad_addend")

        high = "HI" in Inst.opcodes

        if high:
            v1 = IRBuilder.zext(v1, self.ir.IntType(64), "imad_lhs_64")
            v2 = IRBuilder.zext(v2, self.ir.IntType(64), "imad_rhs_64")
            tmp = IRBuilder.mul(v1, v2, "imad_tmp_64")
            tmp = IRBuilder.lshr(tmp, self.ir.Constant(self.ir.IntType(64), 32), "imad_tmp_hi")
            tmp = IRBuilder.trunc(tmp, dest.get_ir_type(self), "imad_tmp_hi_trunc")
        else:
            tmp = IRBuilder.mul(v1, v2, "imad_tmp")

        tmp = IRBuilder.add(tmp, v3, "imad")
        self._set_val(IRBuilder, IRRegs, dest, tmp)

    @lift_for("IMAD64")
    def lift_imad64(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "imad_lhs")
        v1_64 = IRBuilder.zext(v1, self.ir.IntType(64), "imad_lhs_64")
        v2 = _get_val(uses[1], "imad_rhs")
        v2_64 = IRBuilder.zext(v2, self.ir.IntType(64), "imad_rhs_64")
        v3 = _get_val(uses[2], "imad_addend")

        tmp = IRBuilder.mul(v1_64, v2_64, "imad_tmp")
        tmp = IRBuilder.add(tmp, v3, "imad")
        self._set_val(IRBuilder, IRRegs, dest, tmp)

    @lift_for("FADD")
    def lift_fadd(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "fadd_lhs")
        v2 = _get_val(uses[1], "fadd_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.fadd(v1, v2, "fadd"))

    @lift_for("FFMA")
    def lift_ffma(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "ffma_lhs")
        v2 = _get_val(uses[1], "ffma_rhs")
        v3 = _get_val(uses[2], "ffma_addend")
        tmp = IRBuilder.fmul(v1, v2, "ffma_tmp")
        tmp = IRBuilder.fadd(tmp, v3, "ffma")
        self._set_val(IRBuilder, IRRegs, dest, tmp)

    @lift_for("FMUL")
    def lift_fmul(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "fmul_lhs")
        v2 = _get_val(uses[1], "fmul_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.fmul(v1, v2, "fmul"))

    @lift_for("FMNMX")
    def lift_fmnmx(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "fmnmx_lhs")
        v2 = _get_val(uses[1], "fmnmx_rhs")
        pred = Inst.get_uses()[2]
        if pred.is_pt:
            r = IRBuilder.fcmp_ordered(">", v1, v2, "fmnmx")
        else:
            r = IRBuilder.fcmp_ordered("<", v1, v2, "fmnmx")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.select(r, v1, v2, "fmnmx_select"))

    @lift_for("FCHK")
    def lift_fchk(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "fchk_val")
        chk_type = uses[1]

        func_name = "fchk"
        fchk_fn = self.get_intrinsic(func_name)
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.call(
            fchk_fn, [v1, _get_val(chk_type, "fchk_type")], "fchk_call"
        ))

    @lift_for("DADD")
    def lift_dadd(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "dadd_lhs")
        v2 = _get_val(uses[1], "dadd_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.fadd(v1, v2, "dadd"))

    @lift_for("DFMA")
    def lift_dfma(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "dfma_lhs")
        v2 = _get_val(uses[1], "dfma_rhs")
        v3 = _get_val(uses[2], "dfma_addend")
        tmp = IRBuilder.fmul(v1, v2, "dfma_tmp")
        tmp = IRBuilder.fadd(tmp, v3, "dfma")
        self._set_val(IRBuilder, IRRegs, dest, tmp)

    @lift_for("DMUL")
    def lift_dmul(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "dmul_lhs")
        v2 = _get_val(uses[1], "dmul_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.fmul(v1, v2, "dmul"))

    @lift_for("ISCADD")
    def lift_iscadd(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "iscadd_lhs")
        v2 = _get_val(uses[1], "iscadd_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.add(v1, v2, "iscadd"))

    @lift_for("IADD3", "UIADD3", "IADD364")
    def lift_iadd3(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "iadd3_o1")
        v2 = _get_val(uses[1], "iadd3_o2")
        v3 = _get_val(uses[2], "iadd3_o3")
        tmp = IRBuilder.add(v1, v2, "iadd3_tmp")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.add(tmp, v3, "iadd3"))

    @lift_for("ISUB")
    def lift_isub(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "isub_lhs")
        v2 = _get_val(uses[1], "isub_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.add(v1, v2, "sub"))

    @lift_for("SHL", "USHL")
    def lift_shl(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "shl_lhs")
        v2 = _get_val(uses[1], "shl_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.shl(v1, v2, "shl"))

    @lift_for("SHL64")
    def lift_shl64(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "shl_lhs_64")
        v2 = _get_val(uses[1], "shl_rhs_64")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.shl(v1, v2, "shl"))

    @lift_for("SHR", "SHR64", "USHR")
    def lift_shr(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "shr_lhs")
        v2 = _get_val(uses[1], "shr_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.lshr(v1, v2, "shr"))

    @lift_for("SHF", "USHF")
    def lift_shf(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "shf_lo")
        v2 = _get_val(uses[1], "shf_shift")
        v3 = _get_val(uses[2], "shf_hi")

        lo64 = IRBuilder.zext(v1, self.ir.IntType(64))
        hi64 = IRBuilder.zext(v3, self.ir.IntType(64))
        v = IRBuilder.or_(
            lo64,
            IRBuilder.shl(hi64, self.ir.Constant(self.ir.IntType(64), 32)),
            "shf_concat",
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
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.trunc(
            r, dest.get_ir_type(self), "shf_result"
        ))

    @lift_for("IADD")
    def lift_iadd(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "iadd_lhs")
        v2 = _get_val(uses[1], "iadd_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.add(v1, v2, "iadd"))

    @lift_for("SEL", "FSEL", "USEL")
    def lift_sel(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "sel_true")
        v2 = _get_val(uses[1], "sel_false")
        pred = _get_val(uses[2], "sel_pred")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.select(pred, v1, v2, "sel"))

    @lift_for("IADD64")
    def lift_iadd64(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "iadd_lhs")
        v2 = _get_val(uses[1], "iadd_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.add(v1, v2, "iadd"))

    @lift_for("IADD32I", "IADD32I64")
    def lift_iadd32i(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        op1, op2 = uses[0], uses[1]
        v1 = _get_val(op1, "iadd32i_lhs")

        def sx(v, n):
            v &= (1 << n) - 1
            return (v ^ (1 << (n - 1))) - (1 << (n - 1))

        op2.set_immediate(sx(int(op2.name, 16), 24))
        v2 = _get_val(op2, "iadd32i_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.add(v1, v2, "iadd32i"))

    @lift_for("LOP", "ULOP")
    def lift_lop(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        a, b = Inst.get_uses()[0], Inst.get_uses()[1]
        subop = Inst.opcodes[1] if len(Inst.opcodes) > 1 else None
        vb = _get_val(b, "lop_b")

        if subop == "PASS_B":
            self._set_val(IRBuilder, IRRegs, dest, vb)
        else:
            raise UnsupportedInstructionException

    @lift_for("LOP32I", "ULOP32I")
    def lift_lop32i(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        a, b = Inst.get_uses()[0], Inst.get_uses()[1]
        v1 = _get_val(a, "lop32i_a")
        v2 = _get_val(b, "lop32i_b")
        func = Inst.opcodes[1] if len(Inst.opcodes) > 1 else None

        if func == "AND":
            self._set_val(IRBuilder, IRRegs, dest, IRBuilder.and_(v1, v2, "lop32i_and"))
        else:
            raise UnsupportedInstructionException

    @lift_for("BFE")
    def lift_bfe(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        raise UnsupportedInstructionException

    @lift_for("BFI")
    def lift_bfi(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        raise UnsupportedInstructionException

    @lift_for("SSY")
    def lift_ssy(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        pass

    @lift_for("SYNC")
    def lift_sync(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        pass

    @lift_for("WARPSYNC")
    def lift_warpsync(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        pass

    @lift_for("BRA")
    def lift_bra(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        target_id = Inst.get_uses()[-1].name.zfill(4)
        for bb, ir_bb in block_map.items():
            if int(bb.addr_content, 16) != int(target_id, 16):
                continue
            targetBB = ir_bb

        IRBuilder.branch(targetBB)

    @lift_for("BRK")
    def lift_brk(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        raise UnsupportedInstructionException

    @lift_for("IMNMX")
    def lift_imnmx(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "imnmx_lhs")
        v2 = _get_val(uses[1], "imnmx_rhs")

        isUnsigned = "U32" in Inst.opcodes
        isMax = "MXA" in Inst.opcodes

        if isUnsigned:
            if isMax:
                cond = IRBuilder.icmp_unsigned(">", v1, v2, "imnmx_cmp")
            else:
                cond = IRBuilder.icmp_unsigned("<", v1, v2, "imnmx_cmp")
        else:
            if isMax:
                cond = IRBuilder.icmp_signed(">", v1, v2, "imnmx_cmp")
            else:
                cond = IRBuilder.icmp_signed("<", v1, v2, "imnmx_cmp")

        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.select(cond, v1, v2, "imnmx_max"))

    @lift_for("PSETP", "UPSETP64")
    def lift_psetp(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        raise UnsupportedInstructionException

    @lift_for("PBK")
    def lift_pbk(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        raise UnsupportedInstructionException

    @lift_for("LEA", "LEA64", "ULEA")
    def lift_lea(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        
        high = "HI" in Inst.opcodes 

        v1 = _get_val(uses[0], "lea_low")
        v2 = _get_val(uses[1], "lea_b")
        v3 = _get_val(uses[2], "lea_high")
        v4 = _get_val(uses[3], "lea_shift")
        
        low64 = IRBuilder.zext(v1, self.ir.IntType(64), "lea_low_64")
        high64 = IRBuilder.zext(v3, self.ir.IntType(64), "lea_high_64")
        addr = IRBuilder.or_(
            low64,
            IRBuilder.shl(high64, self.ir.Constant(self.ir.IntType(64), 32)),
            "lea_concat",
        )
        b_64 = IRBuilder.zext(v2, self.ir.IntType(64), "lea_b_64")
        v4_64 = IRBuilder.zext(v4, self.ir.IntType(64), "lea_shift_64")
        tmp = IRBuilder.shl(addr, v4_64, "lea_shifted")
        tmp = IRBuilder.add(tmp, b_64, "lea")
        
        if high:
            tmp = IRBuilder.lshr(tmp, self.ir.Constant(self.ir.IntType(64), 32), "lea_hi")
            
        tmp = IRBuilder.trunc(tmp, dest.get_ir_type(self), "lea_hi_trunc")
            
        
        self._set_val(IRBuilder, IRRegs, dest, tmp)

    @lift_for("F2I")
    def lift_f2i(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        op1 = Inst.get_uses()[0]

        isUnsigned = "U32" in Inst.opcodes

        v1 = _get_val(op1, "f2i_src")

        if isUnsigned:
            val = IRBuilder.fptoui(v1, dest.get_ir_type(self), "f2i")
        else:
            val = IRBuilder.fptosi(v1, dest.get_ir_type(self), "f2i")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("I2F")
    def lift_i2f(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        op1 = Inst.get_uses()[0]

        isUnsigned = "U32" in Inst.opcodes

        v1 = _get_val(op1, "i2f_src")

        if isUnsigned:
            val = IRBuilder.uitofp(v1, dest.get_ir_type(self), "i2f")
        else:
            val = IRBuilder.sitofp(v1, dest.get_ir_type(self), "i2f")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("F2F")
    def lift_f2f(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        op1 = Inst.get_uses()[0]

        src_type = Inst.opcodes[1]
        dest_type = Inst.opcodes[2]

        src_width = int(src_type[1:])
        dest_width = int(dest_type[1:])

        v1 = _get_val(op1, "f2f_src")
        if src_width < dest_width:
            self._set_val(IRBuilder, IRRegs, dest, IRBuilder.fpext(
                v1, dest.get_ir_type(self), "f2f_ext"
            ))
        elif src_width > dest_width:
            self._set_val(IRBuilder, IRRegs, dest, IRBuilder.fptrunc(
                v1, dest.get_ir_type(self), "f2f_trunc"
            ))
        else:
            raise UnsupportedInstructionException()

    @lift_for("IABS")
    def lift_iabs(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        v = _get_val(src, "iabs_src")
        abs_fn = self.get_intrinsic("abs")
        res = IRBuilder.call(abs_fn, [v], "iabs")
        self._set_val(IRBuilder, IRRegs, dest, res)

    @lift_for("LOP3", "ULOP3", "PLOP3", "UPLOP3")
    def lift_lop3(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        if Inst.opcodes[1] != "LUT":
            raise UnsupportedInstructionException

        destPred = Inst.get_defs()[0] if len(Inst.get_defs()) > 1 else None
        destReg = Inst.get_defs()[-1]
        src1, src2, src3 = Inst.get_uses()[0], Inst.get_uses()[1], Inst.get_uses()[2]

        lut = Inst.get_uses()[3]
        src4 = Inst.get_uses()[4]

        a = _get_val(src1, "lop3_a")
        b = _get_val(src2, "lop3_b")
        c = _get_val(src3, "lop3_c")
        q = _get_val(src4, "lop3_q")

        lut_val = lut.immediate_value

        na = IRBuilder.xor(a, self.ir.Constant(a.type, -1))
        nb = IRBuilder.xor(b, self.ir.Constant(b.type, -1))
        nc = IRBuilder.xor(c, self.ir.Constant(c.type, -1))

        zero = self.ir.Constant(destReg.get_ir_type(self), 0)
        r = zero

        for idx in range(8):
            if (lut_val >> idx) & 1 == 0:
                continue

            a_bit = (idx >> 2) & 1
            b_bit = (idx >> 1) & 1
            c_bit = idx & 1

            va = a if a_bit else na
            vb = b if b_bit else nb
            vc = c if c_bit else nc

            t_ab = IRBuilder.and_(va, vb)
            t_term = IRBuilder.and_(t_ab, vc)

            if r is zero:
                r = t_term
            else:
                r = IRBuilder.or_(r, t_term)

        if destReg.is_writable_reg:
            self._set_val(IRBuilder, IRRegs, destReg, r)

        if destPred and destPred.is_writable_reg:
            tmp = IRBuilder.icmp_signed("!=", r, zero, "lop3_pred_cmp")
            self._set_val(IRBuilder, IRRegs, destPred, IRBuilder.or_(tmp, q, "lop3_p"))

    @lift_for("MOVM")
    def lift_movm(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src, "movm")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("HMMA")
    def lift_hmma(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        size = Inst.opcodes[1]
        type = Inst.opcodes[2]

        if type != "F32":
            raise UnsupportedInstructionException

        if size != "1688":
            raise UnsupportedInstructionException

        func = self.get_intrinsic("hmma1688f32")
        uses = Inst.get_uses()
        args = [_get_val(uses[i], f"hmma_arg_{i}") for i in range(8)]
        val = IRBuilder.call(func, args, "hmma_call")

        dests = Inst.get_defs()
        if len(dests) >= 4:
            for i in range(4):
                self._set_val(IRBuilder, IRRegs, dests[i], IRBuilder.extract_value(
                    val, i, f"hmma_res_{i}"
                ))

    @lift_for("DEPBAR")
    def lift_depbar(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        pass

    @lift_for("ULDC", "ULDC64", "LDC")
    def lift_uldc(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        self._set_val(IRBuilder, IRRegs, dest, _get_val(src, "ldc_const"))

    @lift_for("FSETP", "DSETP")
    def lift_fsetp(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest1 = Inst.get_defs()[0]
        dest2 = Inst.get_defs()[1]
        a, b = Inst.get_uses()[0], Inst.get_uses()[1]
        src = Inst.get_uses()[2]

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
            "XOR": IRBuilder.xor,
        }

        if isUnordered:
            r = IRBuilder.fcmp_unordered(funcs_map[cmp1], val1, val2, "fsetrp_cmp1")
        else:
            r = IRBuilder.fcmp_ordered(funcs_map[cmp1], val1, val2, "fsetrp_cmp1")

        cmp2_func = cmp2_map.get(cmp2)
        if not cmp2_func:
            raise UnsupportedInstructionException

        r1 = cmp2_func(r, val3, "fsetrp_cmp2")

        if not dest1.is_pt:
            self._set_val(IRBuilder, IRRegs, dest1, r1)

        if not dest2.is_pt:
            r2 = IRBuilder.not_(r1, "fsetrp_not")
            self._set_val(IRBuilder, IRRegs, dest2, r2)

    @lift_for("ISETP", "ISETP64", "UISETP")
    def lift_isetp(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest1 = Inst.get_defs()[0]
        dest2 = Inst.get_defs()[1]
        a, b = Inst.get_uses()[0], Inst.get_uses()[1]
        src = Inst.get_uses()[2]

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
            "XOR": IRBuilder.xor,
        }

        if isUnsigned:
            r = IRBuilder.icmp_unsigned(funcs_map[cmp1], val1, val2, "isetrp_cmp1")
        else:
            r = IRBuilder.icmp_signed(funcs_map[cmp1], val1, val2, "isetrp_cmp1")

        cmp2_func = cmp2_map.get(cmp2)
        if not cmp2_func:
            raise UnsupportedInstructionException

        r1 = cmp2_func(r, val3, "isetrp_cmp2")

        if not dest1.is_pt:
            self._set_val(IRBuilder, IRRegs, dest1, r1)

        if not dest2.is_pt:
            r2 = IRBuilder.not_(r1, "isetrp_not")
            self._set_val(IRBuilder, IRRegs, dest2, r2)

    @lift_for("PBRA")
    def lift_pbra(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        pred = Inst.get_uses()[0]

        cond = _get_val(pred, "cond")

        true_br, false_br = Inst.parent.get_branch_pair(Inst, block_map.keys())
        IRBuilder.cbranch(cond, block_map[true_br], block_map[false_br])

    @lift_for("PACK64")
    def lift_pack64(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0])
        v2 = _get_val(uses[1])
        lo64 = IRBuilder.zext(v1, ir.IntType(64), "pack64_lo")
        hi64 = IRBuilder.zext(v2, ir.IntType(64), "pack64_hi")
        hiShift = IRBuilder.shl(hi64, ir.Constant(ir.IntType(64), 32), "pack64_hi_shift")
        packed = IRBuilder.or_(lo64, hiShift, "pack64_result")
        self._set_val(IRBuilder, IRRegs, dest, packed)

    @lift_for("UNPACK64")
    def lift_unpack64(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dests = Inst.get_defs()
        src = Inst.get_uses()[0]
        val = _get_val(src)
        lo32 = IRBuilder.trunc(val, ir.IntType(32), "unpack64_lo")
        hi32_shift = IRBuilder.lshr(val, ir.Constant(ir.IntType(64), 32), "unpack64_hi_shift")
        hi32 = IRBuilder.trunc(hi32_shift, ir.IntType(32), "unpack64_hi")
        self._set_val(IRBuilder, IRRegs, dests[0], lo32)
        self._set_val(IRBuilder, IRRegs, dests[1], hi32)

    @lift_for("CAST64")
    def lift_cast64(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        op1 = Inst.get_uses()[0]
        if not dest.is_reg:
            raise UnsupportedInstructionException(f"CAST64 expects a register operand, got: {dest}")

        val = _get_val(op1)

        val64 = IRBuilder.sext(val, ir.IntType(64), "cast64")
        self._set_val(IRBuilder, IRRegs, dest, val64)

    @lift_for("BITCAST")
    def lift_bitcast(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        dest_type = dest.get_ir_type(self)

        val = _get_val(src)

        cast_val = IRBuilder.bitcast(val, dest_type, "cast")
        self._set_val(IRBuilder, IRRegs, dest, cast_val)

    @lift_for("POPC", "UPOPC")
    def lift_popc(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src)

        funcName = "llvm.ctpop.i32"

        popc_fn = self.get_intrinsic(
            funcName, ret_ty=ir.IntType(32), arg_tys=[ir.IntType(32)]
        )

        popcVal = IRBuilder.call(popc_fn, [val], "popc")
        self._set_val(IRBuilder, IRRegs, dest, popcVal)

    @lift_for("RED")
    def lift_red(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        uses = Inst.get_uses()
        src1, src2 = uses[0], uses[1]
        val1 = _get_val(src1)
        val2 = _get_val(src2)
        mode = Inst.opcodes[2]
        order = "seq_cst"
        IRBuilder.atomic_rmw(mode, val1, val2, order)

    @lift_for("HADD2")
    def lift_hadd2(self, Inst, IRBuilder, IRRegs, ConstMem=None, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src1, src2 = Inst.get_uses()[0], Inst.get_uses()[1]
        val1 = _get_val(src1)
        val2 = _get_val(src2)

        hadd_fn = self.get_intrinsic(
            "hadd2",
            ret_ty=ir.IntType(32),
            arg_tys=[ir.IntType(32), ir.IntType(32)],
        )

        haddVal = IRBuilder.call(hadd_fn, [val1, val2], "hadd2")
        self._set_val(IRBuilder, IRRegs, dest, haddVal)
        
    def _set_val(self, IRBuilder, IRRegs, operand, val):
        # check value type matches operand type
        target_ty = operand.get_ir_type(self)
        name = operand.name
        # cast if needed
        if "i" in target_ty.intrinsic_name and "i" in val.type.intrinsic_name:
            if val.type.width < target_ty.width:
                val = IRBuilder.zext(val, target_ty, f"{name}_zext" if name else "reg_zext")
            elif val.type.width > target_ty.width:
                val = IRBuilder.trunc(val, target_ty, f"{name}_trunc" if name else "reg_trunc")
                
        elif "f" in target_ty.intrinsic_name and "f" in val.type.intrinsic_name:
            valfpwidth = int(val.type.intrinsic_name[1:])
            targetfpwidth = int(target_ty.intrinsic_name[1:]) 
            if valfpwidth < targetfpwidth:
                val = IRBuilder.fpext(val, target_ty, f"{name}_fpext" if name else "reg_fpext")
            elif valfpwidth > targetfpwidth:
                val = IRBuilder.fptrunc(val, target_ty, f"{name}_fptrunc" if name else "reg_fptrunc")
                
        else:
            val = IRBuilder.bitcast(val, target_ty, f"{name}_bitcast" if name else "reg_bitcast")
        
        IRRegs[operand.get_ir_name(self)] = val

    def get_ir_type(self, type_desc):
        if type_desc == "Int32":
            return self.ir.IntType(32)
        if type_desc == "Float32":
            return self.ir.FloatType()
        if type_desc == "Int32_PTR":
            return self.ir.PointerType(self.ir.IntType(32), 1)
        if type_desc == "Float32_PTR":
            return self.ir.PointerType(self.ir.FloatType(), 1)
        if type_desc == "Int64":
            return self.ir.IntType(64)
        if type_desc == "Int64_PTR":
            return self.ir.PointerType(self.ir.IntType(64), 1)
        if type_desc == "Float64":
            return self.ir.DoubleType()
        if type_desc == "Float64_PTR":
            return self.ir.PointerType(self.ir.DoubleType(), 1)
        if type_desc == "Bool":
            return self.ir.IntType(1)
        if type_desc == "PTR":
            return self.ir.PointerType(self.ir.IntType(8), 1)
        if type_desc == "NOTYPE":
            return self.ir.IntType(32)

        raise ValueError(f"Unknown type: {type_desc}")

    def shutdown(self):
        binding.shutdown()
