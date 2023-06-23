// Script that allows to select a part of the programm code that performs copy actions via 
// real_volotile() ... write_volotile() as detected by search_copy_functions.java and to copy 
// corresponding labels and comments either from source to target or from target to source

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


public class copy_labels_and_comments extends GhidraScript {
    
    @Override 
    protected void run() throws Exception{
        
        if (currentSelection == null || currentSelection.isEmpty()) {
            println("No selection available. Please select some code.");
            return;
        }   

        ProgramSelection s = currentSelection;
        Address baseAddr   = s.getMinAddress();
        long numAddresses  = s.getNumAddresses();
        long numCycles     = numAddresses / 8;

        if (numAddresses % 8 != 0) {
            println("Please select exactly one or more read / write operations.");
            println("One read / write operation is 8 addresses long");
            return;
        }

        for (int i = 0; i < numCycles; i++) {

            Instruction read  = getInstructionAt(baseAddr);
            Instruction write = read.getNext();
            Address sourceAddr = currentProgram.getAddressFactory().getAddress(read.getDefaultOperandRepresentation(1));
            Address targetAddr = currentProgram.getAddressFactory().getAddress(write.getDefaultOperandRepresentation(0));
            Symbol sourceSymbol = getSymbolAt(sourceAddr);
            Symbol targetSymbol = getSymbolAt(targetAddr);
            Integer numUnderscores = 1;
            Boolean hasDuplicate = true;

            if (sourceSymbol.toString().startsWith("DAT") && targetSymbol.toString().startsWith("DAT")) {
                println("0x" + baseAddr + ": Both source and target label start with 'DAT' --> No new label");
            
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
                println("0x" + baseAddr + ": New label");
                
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
                println("0x" + baseAddr + ": New label");

            } else {
                println("0x" + baseAddr + ": Both source and target have a valid label --> No new label");
            }
            
            baseAddr = baseAddr.add(8);

        } 
    } // end run()
} // end class
