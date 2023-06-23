// Script that iterates over all instructions of a program and identifies parts, where a 
// value is first read and then directly written to a different address. In such cases, the
// script copies corresponding labels and comments either from source to target or from 
// target to source

// @author Michel Jubke
// @category Tricore_Tools_for_Ghidra
// @keybinding
// @menupath
// @toolbar

import java.text.StringCharacterIterator;

import ghidra.app.script.GhidraScript;
import ghidra.program.util.ProgramSelection;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;


public class auto_copy_labels_and_comments extends GhidraScript {
    
    @Override 
    protected void run() throws Exception{
        
        InstructionIterator instrIterator = currentProgram.getListing().getInstructions(true);
        
        Instruction baseInstr = instrIterator.next();
        
        while (instrIterator.hasNext()) {
        
            Instruction instrOne    = baseInstr;
            Instruction instrTwo    = instrIterator.next();
            String nameOne          = instrOne.getMnemonicString();
            String nameTwo          = instrTwo.getMnemonicString();
            String opOneOne         = instrOne.getDefaultOperandRepresentation(0);
            String opOneTwo         = instrOne.getDefaultOperandRepresentation(1);
            String opTwoOne         = instrTwo.getDefaultOperandRepresentation(0);
            String opTwoTwo         = instrTwo.getDefaultOperandRepresentation(1);
            Address instrOneAddress = instrOne.getMinAddress();
            
            if (nameOne.startsWith("ld") && nameTwo.startsWith("st")) { 
                if (opOneTwo.startsWith("0x") && opTwoOne.startsWith("0x") && opOneTwo.length() == 10 && opTwoOne.length() == 10) {
                    if (opOneOne.equals(opTwoTwo)) {

                        Instruction read  = instrOne;
                        Instruction write = instrTwo;
                        Address sourceAddr = currentProgram.getAddressFactory().getAddress(read.getDefaultOperandRepresentation(1));
                        Address targetAddr = currentProgram.getAddressFactory().getAddress(write.getDefaultOperandRepresentation(0));
                        Symbol sourceSymbol = getSymbolAt(sourceAddr);
                        Symbol targetSymbol = getSymbolAt(targetAddr);
                        Integer numUnderscores = 1;
                        Boolean hasDuplicate = true;

                        if (sourceSymbol.toString().startsWith("DAT") && targetSymbol.toString().startsWith("DAT")) {
                            println("0x" + instrOneAddress + ": Both source and target label start with 'DAT' --> No new label");
                        
                        } else if (targetSymbol.toString().startsWith("DAT")) {                
                            while (hasDuplicate) {
                                SymbolIterator sIter = currentProgram.getSymbolTable().getSymbolIterator();
                                while (sIter.hasNext()) {
                                    Symbol symbolToCheck = sIter.next();
                                    if (symbolToCheck.toString() == sourceSymbol.toString() + "_".repeat(numUnderscores)) {
                                        numUnderscores += 1;
                                        break;
                                    }
                                }
                                hasDuplicate = false;
                            }    
                            Symbol newLabel = createLabel(targetAddr, sourceSymbol.toString() + "_".repeat(numUnderscores), true);
                            String comment  = currentProgram.getListing().getComment(3, sourceAddr);
                            if (comment != null) {
                                currentProgram.getListing().setComment(targetAddr, 3, comment);
                            }
                            println("0x" + instrOneAddress + ": New label");
        
                        } else if (sourceSymbol.toString().startsWith("DAT")) {
                            while (hasDuplicate) {
                                SymbolIterator sIter = currentProgram.getSymbolTable().getSymbolIterator();                    
                                while (sIter.hasNext()) {
                                    Symbol symbolToCheck = sIter.next();
                                    if (symbolToCheck.toString() == targetSymbol.toString() + "_".repeat(numUnderscores)) {
                                        numUnderscores += 1;
                                        break;
                                    }
                                }
                                hasDuplicate = false;
                            }
                            Symbol newLabel = createLabel(sourceAddr, targetSymbol.toString() + "_".repeat(numUnderscores), true);
                            String comment  = currentProgram.getListing().getComment(3, targetAddr);
                            if (comment != null) {
                                currentProgram.getListing().setComment(sourceAddr, 3, comment);
                            }
                            println("0x" + instrOneAddress + ": New label");
                                        
                        } else {
                            println("0x" + instrOneAddress + ": Both source and target have a valid label --> No new label");
                        }
                    }
                }
            }

            baseInstr = instrTwo;
            
        }                 
    } // end run()
} // end class
