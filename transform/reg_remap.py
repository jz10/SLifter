from transform.transform import SaSSTransform


class RegRemap(SaSSTransform):
    def apply(self, module):
        print("=== Start of RegRemap ===")
        for func in module.functions:
            if not func.blocks:
                continue
            self.remapToSimpleNames(func)
        print("=== End of RegRemap ===")

    def remapToSimpleNames(self, function):
        r_name_map = {}
        p_name_map = {}
        r_counter = 1
        p_counter = 1

        r_names = set()
        p_names = set()
        for block in function.blocks:
            for inst in block.instructions:
                for op in inst.operands:
                    if not op.IsWritableReg:
                        continue
                    if op.IsPredicateReg:
                        p_names.add(op.Reg)
                    else:
                        r_names.add(op.Reg)

        for name in sorted(r_names):
            r_name_map[name] = f"R{r_counter}"
            r_counter += 1
        for name in sorted(p_names):
            p_name_map[name] = f"P{p_counter}"
            p_counter += 1

        for block in function.blocks:
            for inst in block.instructions:
                if inst.Predicated() and inst.pflag.Reg in p_name_map:
                    inst.pflag.SetReg(p_name_map[inst.pflag.Reg])
                for op in inst.operands:
                    if op.IsPredicateReg and op.Reg in p_name_map:
                        op.SetReg(p_name_map[op.Reg])
                    elif op.Reg in r_name_map:
                        op.SetReg(r_name_map[op.Reg])