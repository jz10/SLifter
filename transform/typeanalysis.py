from transform.transform import SaSSTransform

class TypeAnalysis(SaSSTransform):
    def apply(self, module):
        for func in module.functions:
            # Identify type seeds
            Seeds = self.IdentifySeeds(func)
        
            # Propagate type information
            self.PropagateTypes(func, Seeds)

    # Identify the type seeds
    def IdentifySeeds(self, Func):
        Seeds = []
        for BB in Func.blocks:
            for Inst in BB.instructions:
                if Inst.IsBinary() or Inst.IsBranch():
                    Seeds.append([Inst, BB])

        return Seeds

    # Propagate type infromation
    def PropagateTypes(self, Func, Seeds):
        VisitedInsts = []
        while len(Seeds) > 0:
            Inst, BB = Seeds.pop(0)
            # Resolve type in instruction
            Inst.ResolveType()

            # Set visited
            VisitedInsts.append(Inst)
            
            # Propagate the uses for the value defined from this instruction
            Uses = self.PropagateUsesForDef(Inst, BB, Func, VisitedInsts)
            # Propagate the defs for the value used from this instruction
            Defs = self.PropagateDefForUses(Inst, BB, Func, VisitedInsts)

            # Add the uses into worklist
            for Use in Uses:
                if Use not in Seeds:
                    Seeds.append(Use)

            # Add the defs into worklist
            for Def in Defs:
                if Def not in Seeds:
                    Seeds.append(Def)

    # Collect the uses for the value defined in the given instruction
    def PropagateUsesForDef(self, Inst, BB, Func, VisitedInsts):
        Uses = []

        Def = Inst.GetDef()
        if Def == None:
            return Uses
            
        # Search uses in same basic block
        Insts = BB.instructions
        NeedCheck = False
        for i in range(len(Insts)):
            CurrInst = Insts[i]
            if NeedCheck:
                if CurrInst.IsStore():
                    # Check store instruction
                    if CurrInst.CheckAndUpdateUseType(Def) and not CurrInst in VisitedInsts:
                        Uses.append([CurrInst, BB])
                        VisitedInsts.append(CurrInst)
            else:
                if CurrInst == Inst:
                    NeedCheck = True

        return Uses
        
        
    # Collect the def for the values used in the given instruction
    def PropagateDefForUses(self, Inst, BB, Func, VisitedInsts):
        Defs = []

        Uses = Inst.GetUses()
    
        # Search defs in same basic block
        Insts = BB.instructions
        NeedCheck = False
        for i in reversed(range(len(Insts))):
            CurrInst = Insts[i]
            if NeedCheck:
                if CurrInst.IsAddrCompute():
                    if CurrInst.CheckAndUpdateDefType(Uses) and not CurrInst in VisitedInsts:
                        Defs.append([CurrInst, BB])
                        VisitedInsts.append(CurrInst)
                if CurrInst.IsLoad():
                    if CurrInst.CheckAndUpdateDefType(Uses) and not CurrInst in VisitedInsts:
                        Defs.append([CurrInst, BB])
                        VisitedInsts.append(CurrInst)
            else:
                if CurrInst == Inst:
                    NeedCheck = True
                        
        return Defs
