from sir.instruction import Instruction
from sir.basicblock import BasicBlock

class DefUse:
    def __init__(self):
        self.Def2Uses = {}
        self.Use2Def = {}
            
    # Add def-use chaiin
    def AddDefUse(self, DefInst, UseInst):
        self.Def2Uses[DefInst].append(UseInst)
        self.Use2Def[UseInst] = DefInst

    # Get uses from the given def
    def GetUses(self, DefInst):
        return self.Def2Uses[DefInst]

    # Get def from the given use
    def GetDef(self, UseInst):
        return self.Use2Def[UseInst]

    # Collect direct def and use set
    def ColectDirectDU(self, Blocks):
        # Collect direct def and use set
        for BB in Blocks:
            if BB.Uses != None:
                continue

            FirstUses = {}
            LastDefs = {}
            for inst in BB.instructions:
                uses = inst.GetUses()
                for use in uses:
                    # Collect the uses, i.e. 1st use
                    if use in FirstUses:
                        continue

                    FirstUses[use] = inst

                defs = inst.GetDefs()
                for Def in defs:
                    # Collect the defs, i.e. last defs
                    LastDefs[Def] = inst

            # Set direct defs uses
            BB.DirectUses.update(FirstUses)
            BB.DirectDefs.update(LastDefs)

    # Build the global def-use information, i.e. per basic block based def-use set
    def BuildGlobalDU(self, Blocks):
        # Collect direct defs and uses
        self.CollectDirectDU(Blocks)

        # Build global def-use via propagating indirect defs and uses
        worklist = list(Blocks)
        while len(worklist) != 0:
            # Popup 1st element
            BB = worklist[0]
            del(worklist[0])
        
            # Check pred BBs for updating indirect defs
            IndDef = {}
            for pred in BB.preds:
                IndDef.update(pred.DirectDef)
                IndDef.update(pred.IndirectDef)
                
            IndDef.difference_update(BB.DirectDef)
            IndDef.difference_update(BB.IndirectDef)
            if len(IndDef) != 0:
                # Merge indirect defs
                BB.IndirectDef.update(IndDef)
                # Add successors into worklist
                for succ in BB.succs:
                    if succ in worklist:
                        continue
                    worklist.append(succ)
                    
            # Check succ BBs for updating indirect uses
            IndUse = {}
            for succ in BB.succs:
                IndUse.update(succ.DirectUse)
                IndUse.update(succ.IndirectUse)

            IndUse.difference_update(BB.DirectUse)
            IndUse.difference_update(BB.IndirectUse)
            if len(IndUse) != 0:
                # Merge indirect uses
                BB.IndirectUse.update(IndUse)
                # Add predecessors into worklist
                for pred in BB.preds:
                    if pred in worklist:
                        continue
                    worklist.append(pred)
