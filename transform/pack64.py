from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand


class Pack64(SaSSTransform):

    def _next_hi_name(self, reg_name):
        if reg_name.startswith('UR'):
            prefix = 'UR'
            rest = reg_name[2:]
        else:
            prefix = 'R'
            rest = reg_name[1:]
        return f"{prefix}{int(rest)+1}"

    def _prev_hi_name(self, reg_name):
        if reg_name.startswith('UR'):
            prefix = 'UR'
            rest = reg_name[2:]
        else:
            prefix = 'R'
            rest = reg_name[1:]
        return f"{prefix}{int(rest)-1}"

    def _is_simple_hw_reg(self, reg_name):
        if reg_name.startswith('UR'):
            rest = reg_name[2:]
        elif reg_name.startswith('R'):
            rest = reg_name[1:]
        else:
            return False
        return rest.isdigit()

    def create_pack64(self, addr_op, parentBB):
        src_op_lower = addr_op.Clone()
        src_op_lower.IsMemAddr = False

        src_op_upper = src_op_lower.Clone()
        src_op_upper.SetReg(self._next_hi_name(addr_op.Reg))

        dest_op_name = src_op_lower.Reg + "_int64"
        dst_op = Operand.fromReg(dest_op_name, dest_op_name)

        pack64_inst = Instruction(
            id=f"pack64_{addr_op.Name}",
            opcodes=["PACK64"],
            operands=[
                dst_op,
                src_op_lower,
                src_op_upper
            ],
            parentBB=parentBB
        )

        addr_op.SetReg(dest_op_name)

        return pack64_inst

    def create_iadd64(self, addr_op, offset_op, inst):
        src_op_base = Operand.fromReg(addr_op.Name, addr_op.Name)
        src_op_offset = Operand.fromReg(offset_op.Name, offset_op.Name)
        dest_op_name = f"{addr_op.Name}_iadd64"
        dest_op = Operand.fromReg(dest_op_name, dest_op_name)

        iadd64_inst = Instruction(
            id=f"iadd64_{addr_op.Name}",
            opcodes=["IADD64"],
            operands=[
                dest_op,
                src_op_base,
                src_op_offset
            ],
            parentBB=inst.parent
        )

        addr_op.Replace(dest_op)

        return iadd64_inst

    def apply(self, module):
        print("=== Start of Pack64 Transformation ===")


        count = 0
        for func in module.functions:
            for block in func.blocks:
                new_insts = []
                for inst in block.instructions:
                #     for op in inst.operands:
                #         if not op.IsMemAddr:
                #             continue

                #         if op.IsRZ:
                #             continue

                #         ur_offset = True if op.MemAddrOffset and "UR" in op.MemAddrOffset else False
                #         r_base = "R" in op.Reg

                #         if (not ur_offset) or (ur_offset and r_base and "64" in op.Suffix) and (not op.IsRZ):
                #             new_insts.append(self.create_pack64(op, inst.parent))
                #             count += 1


                #         if ur_offset:
                #             offsetOp = Operand(op.MemAddrOffset, op.MemAddrOffset, None, -1, True, False, False)
                #             inst1 = self.create_pack64(offsetOp, inst.parent)
                #             inst2 = self.create_iadd64(op, offsetOp, inst)
                #             new_insts.append(inst1)
                #             new_insts.append(inst2)
                #             count += 1

                #     new_insts.append(inst)

                # block.instructions = new_insts
                    # IMAD.WIDE.U32 R3, R11, R16, R8, where R8 is a 64 bit value
                    if inst.opcodes[0] == "IMAD" and "WIDE" in inst.opcodes:
                        uses = inst.GetUses()
                        if not uses:
                            new_insts.append(inst)
                            continue

                        valOp = uses[-1]
                        if not valOp.IsWritableReg:
                            new_insts.append(inst)
                            continue

                        # Insert PACK64 instruction
                        src_op_lower = valOp.Clone()
                        src_op_upper_name = self._next_hi_name(valOp.Reg)
                        src_op_upper = Operand.fromReg(
                            src_op_upper_name,
                            src_op_upper_name
                        )

                        new_reg = src_op_lower.Name + "_int64_" + str(count)
                        dst_op = Operand.fromReg(new_reg, new_reg)

                        pack_val_inst = Instruction(
                            id=f"{inst.id}_pack64_val",
                            opcodes=["PACK64"],
                            operands=[
                                dst_op,
                                src_op_lower,
                                src_op_upper
                            ],
                            parentBB=inst.parent
                        )
                        count += 1
                        new_insts.append(pack_val_inst)
                        valOp.SetReg(new_reg)

                    # MATCH.ANY.U64 R35 = R13, where R13 is a 64 bit value
                    if inst.opcodes[0] == "MATCH":

                        if "U64" not in inst.opcodes:
                            new_insts.append(inst)
                            continue

                        uses = inst.GetUses()
                        if not uses:
                            new_insts.append(inst)
                            continue

                        valOp = uses[0]
                        if not valOp.IsWritableReg:
                            new_insts.append(inst)
                            continue

                        # Insert PACK64 instruction
                        src_op_lower = valOp.Clone()
                        src_op_upper_name = self._next_hi_name(valOp.Reg)
                        src_op_upper = Operand.fromReg(
                            src_op_upper_name,
                            src_op_upper_name
                        )

                        new_reg = src_op_lower.Name + "_int64_" + str(count)
                        dst_op = Operand.fromReg(new_reg, new_reg)

                        pack_val_inst = Instruction(
                            id=f"{inst.id}_pack64_val",
                            opcodes=["PACK64"],
                            operands=[
                                dst_op,
                                src_op_lower,
                                src_op_upper
                            ],
                            parentBB=inst.parent
                        )
                        count += 1
                        new_insts.append(pack_val_inst)
                        valOp.SetReg(new_reg)


                    # RED.E.ADD.STRONG.GPU [R2], R5 ;
                    if inst.opcodes[0] == "RED":
                        uses = inst.GetUses()
                        if not uses:
                            new_insts.append(inst)
                            continue

                        addrOp = uses[0]
                        if not addrOp.IsWritableReg:
                            new_insts.append(inst)
                            continue

                        # Insert PACK64 instruction
                        src_op_lower = Operand.fromReg(
                            addrOp.Reg,
                            addrOp.Reg,
                            addrOp.Suffix
                        )
                        src_op_upper_name = self._next_hi_name(addrOp.Reg)
                        src_op_upper = Operand.fromReg(
                            src_op_upper_name,
                            src_op_upper_name,
                            src_op_lower.Suffix
                        )

                        new_reg = src_op_lower.Name + "_int64_" + str(count)
                        dst_op = Operand.fromReg(new_reg, new_reg)

                        pack_addr_inst = Instruction(
                            id=f"{inst.id}_pack64_addr",
                            opcodes=["PACK64"],
                            operands=[
                                dst_op,
                                src_op_lower,
                                src_op_upper
                            ],
                            parentBB=inst.parent
                        )
                        count += 1
                        new_insts.append(pack_addr_inst)
                        addrOp.SetReg(new_reg)

                    if (inst.IsGlobalLoad() or inst.IsGlobalStore()):

                        uses = inst.GetUses()
                        if not uses:
                            new_insts.append(inst)
                            continue

                        addrOp = uses[0]

                        src_op_lower = Operand.fromReg(
                            addrOp.Reg,
                            addrOp.Reg,
                            addrOp.Suffix
                        )

                        src_op_upper_name = self._next_hi_name(addrOp.Reg)

                        src_op_upper = Operand.fromReg(
                            src_op_upper_name,
                            src_op_upper_name,
                            src_op_lower.Suffix
                        )

                        new_reg = src_op_lower.Name + "_int64_" + str(count)
                        addrOp.SetReg(new_reg)
                        dst_op = Operand.fromReg(new_reg, new_reg)

                        cast_inst = Instruction(
                            id=f"{inst.id}_pack64",
                            opcodes=["PACK64"],
                            operands=[
                                dst_op,
                                src_op_lower,
                                src_op_upper
                            ],
                            parentBB=inst.parent
                        )
                        count += 1
                        new_insts.append(cast_inst)

                        if addrOp.IsMemAddr and isinstance(addrOp.MemAddrOffset, str) and 'UR' in addrOp.MemAddrOffset:

                            off = addrOp.MemAddrOffset
                            ur_lo = Operand.fromReg(off, off)
                            ur_hi_name = self._next_hi_name(off)
                            ur_hi = Operand.fromReg(ur_hi_name, ur_hi_name)
                            ur64_name = f"{off}_int64"
                            ur64 = Operand.fromReg(ur64_name, ur64_name)

                            new_insts.append(Instruction(
                                id=f"{inst.id}_pack64_{off}",
                                opcodes=["PACK64"],
                                operands=[ur64, ur_lo, ur_hi],
                                parentBB=inst.parent
                            ))
                            count += 1

                            sub_op_name = f"{src_op_lower.Name}_usub"
                            sub_op = Operand.fromReg(sub_op_name, sub_op_name)
                            base_op = Operand.fromReg(new_reg, new_reg)
                            offset_op = Operand.fromReg(ur64_name, ur64_name)

                            new_insts.append(Instruction(
                                id=f"{inst.id}_iadd64_usub",
                                opcodes=["IADD64"],
                                operands=[sub_op, base_op, offset_op],
                                parentBB=inst.parent
                            ))

                            # Rewrite pointer to the uniform-substituted 64-bit base and clear offset
                            addrOp.Replace(sub_op)

                    # If this is a 64-bit store, pack the value operand STG.64 [R2], R6 => PACK64 R6_int64, R6 R5
                    if inst.IsStore() and ('64' in inst.opcodes):
                        uses = inst.GetUses()
                        if len(uses) < 2:
                            new_insts.append(inst)
                            continue

                        valOp = uses[-1]
                        if valOp.IsReg and not valOp.IsRZ and self._is_simple_hw_reg(valOp.Reg):
                            src_op_lower = valOp.Clone()
                            src_op_upper_name = self._next_hi_name(valOp.Reg)
                            src_op_upper = Operand.fromReg(
                                src_op_upper_name,
                                src_op_upper_name
                            )

                            new_reg = src_op_lower.Name + "_int64_" + str(count)
                            dst_op = Operand.fromReg(new_reg, new_reg)

                            pack_val_inst = Instruction(
                                id=f"{inst.id}_pack64_val",
                                opcodes=["PACK64"],
                                operands=[
                                    dst_op,
                                    src_op_lower,
                                    src_op_upper
                                ],
                                parentBB=inst.parent
                            )
                            count += 1
                            new_insts.append(pack_val_inst)
                            valOp.SetReg(new_reg)
                    new_insts.append(inst)
                block.instructions = new_insts

        print(f"Total PACK64 instructions added: {count}")
        print("=== End of Pack64 Transformation ===")
