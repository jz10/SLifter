from transform.transform import SaSSTransform
from collections import deque
from sir.instruction import Instruction
from sir.operand import Operand

class SSA(SaSSTransform):
    def apply(self, module):
        for func in module.functions:
            self.ProcessFunc(func)


    def ProcessFunc(self, function):
        print("=== Start of SSA ===")
        print()

        WorkList = self.TraverseCFG(function)


        InRegsMap = {}
        OutRegsMap = {}
        PhiNodesMap = {}

        for BB in WorkList:
            InRegsMap[BB] = {}
            OutRegsMap[BB] = {}
            PhiNodesMap[BB] = {}

        Changed = True
        
        while Changed:
            Changed = False

            for BB in WorkList:
                Changed |= self.ProcessBB(BB, InRegsMap, OutRegsMap, PhiNodesMap, function)
        self.InsertPhiNodes(WorkList, PhiNodesMap)

        self.RemapRegisters(WorkList)

        # self.PrintRenamedInstructions(WorkList)

        self.UpdateInstContent(WorkList)

        print("SSA done")
        print("=== End of SSA ===")

    def TraverseCFG(self, function):
        EntryBB = function.blocks[0]
        Visited = set()
        Queue = deque([EntryBB])
        WorkList = []

        while Queue:
            CurrBB = Queue.popleft()
            
            if CurrBB in Visited:
                continue

            Visited.add(CurrBB)
            WorkList.append(CurrBB)

            for SuccBB in CurrBB._succs:
                if SuccBB not in Visited:
                    Queue.append(SuccBB)

        return WorkList

    def ExtractBaseRegisterName(self, Reg):
        return Reg.split('@')[0]
    
    def CreateVersionedRegisterName(self, Reg, Inst):
        BaseRegName = self.ExtractBaseRegisterName(Reg)
        VersionedReg = BaseRegName + "@" + str(Inst.id)
        return VersionedReg


    def ProcessBB(self, BB, InRegsMap, OutRegsMap, PhiNodesMap, function):
        NewInRegs = self.GenerateInRegs(BB, InRegsMap, OutRegsMap, PhiNodesMap)

        if BB != function.blocks[0] and NewInRegs == InRegsMap[BB]:
            return False
        
        InRegsMap[BB] = NewInRegs
        CurrRegs = NewInRegs.copy()

        self.ProcessInstructionsInBlock(BB, CurrRegs)
        
        return self.UpdateOutRegisterSet(BB, CurrRegs, OutRegsMap)

    def ProcessInstructionsInBlock(self, BB, CurrRegs):
        for Inst in BB.instructions:
            self.UpdateUseOperands(Inst, CurrRegs)
            self.UpdateDefinitionOperand(Inst, CurrRegs)

    def UpdateUseOperands(self, Inst, CurrRegs):
        Uses = Inst.GetUses()

        Def = Inst.GetDef()

        for Operand in Uses:
            if Operand.IsReg and Operand.Reg:
                # Don't rename predicate registers or RZ register
                if Inst.IsPredicateReg(Operand.Reg) or Operand.Reg == "RZ":
                    continue
                RegName = self.ExtractBaseRegisterName(Operand.Reg)
                if RegName in CurrRegs:
                    Operand._Name = CurrRegs[RegName]
                    Operand._Reg = CurrRegs[RegName]

    def UpdateDefinitionOperand(self, Inst, CurrRegs):
        Def = Inst.GetDef()
        if not Def or not Def.IsReg:
            return

        # Don't rename predicate registers or RZ register
        if Inst.IsPredicateReg(Def.Reg) or Def.Reg == "RZ":
            return
        RegName = self.ExtractBaseRegisterName(Def.Reg)
        NewDef = self.CreateVersionedRegisterName(Def.Reg, Inst)
        CurrRegs[RegName] = NewDef
        Def._Name = NewDef
        Def._Reg = NewDef

    def UpdateOutRegisterSet(self, BB, CurrRegs, OutRegsMap):
        OldOutRegs = OutRegsMap[BB].copy() if BB in OutRegsMap else {}
        OutRegsMap[BB] = CurrRegs.copy()
        return OldOutRegs != OutRegsMap[BB]

    def GenerateInRegs(self, BB, InRegsMap, OutRegsMap, PhiNodesMap):
        if not BB._preds:
            return {}
        
        PhiNodesMap[BB].clear()
        
        register_versions = self.CollectPredecessorVersions(BB, OutRegsMap)
        
        return self.GenerateInSetWithPhiNodes(BB, register_versions, PhiNodesMap)

    def CollectPredecessorVersions(self, BB, OutRegsMap):
        register_versions = {}
        for predecessor_bb in BB._preds:
            if predecessor_bb in OutRegsMap:
                for reg_name, reg_version in OutRegsMap[predecessor_bb].items():
                    if reg_name not in register_versions:
                        register_versions[reg_name] = []
                    register_versions[reg_name].append((predecessor_bb, reg_version))
        return register_versions

    def GenerateInSetWithPhiNodes(self, BB, register_versions, PhiNodesMap):
        incoming_registers = {}
        for reg_name, predecessor_versions in register_versions.items():
            
            unique_versions = {ver for _, ver in predecessor_versions}
            if len(unique_versions) == 1:
                incoming_registers[reg_name] = next(iter(unique_versions))
            elif len(unique_versions) > 1:
                phi_version = reg_name + "@phi_" + str(BB.addr)
                incoming_registers[reg_name] = phi_version
                PhiNodesMap[BB][reg_name] = (phi_version, predecessor_versions)
        return incoming_registers

    def InsertPhiNodes(self, WorkList, PhiNodesMap):
        for basic_block in WorkList:
            
            if not PhiNodesMap[basic_block]:
                continue
                
            phi_instructions = self.CreatePhiInstructions(basic_block, PhiNodesMap[basic_block])
            basic_block.instructions = phi_instructions + basic_block.instructions

    def ClearExistingPhiInstructions(self, basic_block):
        while (basic_block.instructions and 
               basic_block.instructions[0].opcodes == ["PHI"]):
            basic_block.instructions.pop(0)

    def CreatePhiInstructions(self, basic_block, phi_nodes):
        phi_instructions = []
        for reg_name, (phi_version, predecessor_versions) in phi_nodes.items():
            phi_inst = self.CreateSinglePhiInstruction(
                basic_block, reg_name, phi_version, predecessor_versions
            )
            phi_instructions.append(phi_inst)
        return phi_instructions

    def CreateSinglePhiInstruction(self, basic_block, reg_name, phi_version, predecessor_versions):
        phi_operands = []
        
        def_operand = Operand(phi_version, phi_version, None, 0, True, False, False)
        phi_operands.append(def_operand)
        
        sorted_pred_versions = sorted(predecessor_versions, 
                                    key=lambda t: basic_block._preds.index(t[0]))
        
        for pred_bb, version in sorted_pred_versions:
            use_operand = Operand(version, version, None, 0, True, False, False)
            phi_operands.append(use_operand)
        
        incoming_versions = ' '.join([v for _, v in sorted_pred_versions])
        inst_content = f"PHI {phi_version} {incoming_versions}"

        return Instruction(
            id=f"phi_{basic_block.addr}_{reg_name}",
            opcodes=["PHI"],
            operands=phi_operands,
            inst_content=inst_content,
            parentBB=basic_block
        )

    def PrintRenamedInstructions(self, WorkList):
        print("\n=== Instructions with renamed registers ===")
        for basic_block in WorkList:
            print(f"\nBasic Block {basic_block.addr}:")
            for inst in basic_block.instructions:
                print(f"  Original: {inst._InstContent}")
                
                print(f"  Transformed: {inst}")
                print()
        print("=== End of renamed instructions ===\n")

    def RemapRegisters(self, WorkList):
        register_mapping = {}
        register_counter = 1
        
        # collect all unique register names and create mapping
        for basic_block in WorkList:
            for inst in basic_block.instructions:
                for operand in inst.operands:
                    if operand.IsReg and operand.Reg:
                        if inst.IsPredicateReg(operand.Reg) or operand.Reg == "RZ" or operand.Reg == "PT":
                            continue
                        
                        reg_name = operand.Reg

                        if reg_name not in register_mapping:
                            register_mapping[reg_name] = f"R{register_counter}"
                            register_counter += 1
        
        # apply mapping
        for basic_block in WorkList:
            for inst in basic_block.instructions:
                for operand in inst.operands:
                    if operand.IsReg and operand.Reg:
                        if inst.IsPredicateReg(operand.Reg) or operand.Reg == "RZ":
                            continue
                        
                        if operand.Reg in register_mapping:
                            operand._Name = register_mapping[operand.Reg]
                            operand._Reg = register_mapping[operand.Reg]
        
        print(f"Remapped {len(register_mapping)} registers to RN format")

    def UpdateInstContent(self, WorkList):
        for basic_block in WorkList:
            for inst in basic_block.instructions:
                inst._InstContent = inst.__str__()