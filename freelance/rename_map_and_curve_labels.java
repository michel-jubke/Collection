// Script that searches for _MAP und _CUR lables at the beginning of the axis values of a map or curve and 
// sets a new abel at the very beginning of the map or curve.

// @author Michel Jubke
// @category Tricore_Tools_for_Ghidra
// @keybinding
// @menupath
// @toolbar

import ghidra.app.script.GhidraScript;

import ghidra.program.model.listing.*;
import ghidra.program.model.address.*;
import ghidra.program.model.symbol.*;
import ghidra.program.model.data.*;
import ghidra.program.flatapi.*;


public class rename_map_and_curve_labels extends GhidraScript {
    
    @Override 
    protected void run() throws Exception{
        
        Listing        listing         = currentProgram.getListing();  
        SymbolTable    symbolTable     = currentProgram.getSymbolTable();
        SymbolIterator symbolIterator  = symbolTable.getSymbolIterator();

        while(symbolIterator.hasNext()) {
            
            Symbol   symbol        = symbolIterator.next(); 
            String   symbolName    = symbol.getName();            
            Address  symbolAddress = symbol.getAddress();
            
            // relevant for distributes maps
            if(symbolName.endsWith("MAP_COLS")) {
                
                DataType dataType = listing.getDataAt(symbolAddress).getDataType();    
                String newSymbolName  = symbolName.substring(0, symbolName.length() - 4);
                String symbolID       = symbolName.substring(0, symbolName.length() - 9);
                Symbol previousSymbol = getSymbolBefore(symbol); 
                Symbol nextSymbol     = getSymbolAfter(symbol); 

                
                if(nextSymbol != null && previousSymbol != null && ! nextSymbol.getName().contains(symbolID) && ! previousSymbol.getName().contains(symbolID)) {
                    if(dataType.getName().contains("short")) {
                        Address newSymbolAddress = symbolAddress.subtract(2);
                        Boolean isPrimary = symbol.isPrimary();
                        Symbol newLabel = createLabel(newSymbolAddress, newSymbolName + "COLS_", isPrimary);
                        continue;
                    } 
                }
            
            
            } else if(symbolName.endsWith("MAP_ROWS") || symbolName.endsWith("_VW_ROWS") || symbolName.endsWith("_VW_COLS")) {

                // relevant for distributed maps
                DataType dataType = listing.getDataAt(symbolAddress).getDataType();    
                String newSymbolName  = symbolName.substring(0, symbolName.length() - 4);                
                String symbolID       = symbolName.substring(0, symbolName.length() - 9);
                Symbol nextSymbol     = getSymbolAfter(symbol);
                Symbol nextNextSymbol = getSymbolAfter(nextSymbol);

                if(nextSymbol != null && nextNextSymbol != null && ! nextSymbol.getName().contains(symbolID) && ! nextNextSymbol.getName().contains(symbolID)) {
                    if(dataType.getName().contains("short")) {
                        Address newSymbolAddress = symbolAddress.subtract(2);
                        Boolean isPrimary = symbol.isPrimary();
                        if(symbolName.endsWith("_VW_COLS")) {
                            Symbol newLabel = createLabel(newSymbolAddress, newSymbolName + "COLS_", isPrimary);    
                            continue;
                        } else {
                            Symbol newLabel = createLabel(newSymbolAddress, newSymbolName + "ROWS_", isPrimary);
                            continue;
                        }
                    } else if(dataType.getName().contains("byte")) {
                        Address newSymbolAddress = symbolAddress.subtract(1);
                        Boolean isPrimary = symbol.isPrimary();
                        if(symbolName.endsWith("_VW_COLS")) {
                            Symbol newLabel = createLabel(newSymbolAddress, newSymbolName + "COLS_", isPrimary);    
                            continue;
                        } else {
                            Symbol newLabel = createLabel(newSymbolAddress, newSymbolName + "ROWS_", isPrimary);
                            continue;
                        }
                    } else if(dataType.getName().contains("float")) {
                        Address newSymbolAddress = symbolAddress.subtract(4);
                        Boolean isPrimary = symbol.isPrimary();
                        if(symbolName.endsWith("_VW_COLS")) {
                            Symbol newLabel = createLabel(newSymbolAddress, newSymbolName + "COLS_", isPrimary);    
                            continue;
                        } else {
                            Symbol newLabel = createLabel(newSymbolAddress, newSymbolName + "ROWS_", isPrimary);
                            continue;
                        }
                    } else {
                        println("ERROR 9: " + symbol.getName() + " --> Header not found or incorrect");          
                    }
                
                // relevant for non-distributed short maps
                } else if(dataType.getName().contains("short")) {

                    Address newSymbolAddress = symbolAddress.subtract(4);

                    if(symbolName.endsWith("_VW_ROWS")) {
                        newSymbolAddress = newSymbolAddress.add(2);
                        newSymbolName = newSymbolName + "ROWS_"; 
                    }    

                    if(symbolName.endsWith("_VW_COLS")) {
                        newSymbolAddress = newSymbolAddress.add(2);
                        newSymbolName = newSymbolName + "COLS_";                         
                    }
                
                    // relevant if rows and colums are shorts but the data is bytes
                    DataType d1 = listing.getDataAt(symbolAddress.add(dataType.getLength())).getDataType();
                    DataType d2 = listing.getDataAt(symbolAddress.add(dataType.getLength()).add(d1.getLength())).getDataType();
                    if(d2.getName().contains("byte") && ! symbolName.endsWith("_VW_ROWS") && ! symbolName.endsWith("_VW_COLS")) {
                        newSymbolAddress = newSymbolAddress.add(2);
                    }
                    
                    byte realValue     = getByte(newSymbolAddress);
                    int  expectedValue = dataType.getLength() / 2; // one short is two bytes long
                        
                    if(realValue == expectedValue) {
                        Symbol newLabel = createLabel(newSymbolAddress, newSymbolName, true);
                        continue;
                    } else {
                        println("ERROR 1: " + symbol.getName() + " --> Header not found or incorrect");
                    }
                
                // relevant for non-distributed byte maps
                } else if(dataType.getName().contains("byte")) {

                    Address newSymbolAddress = symbolAddress.subtract(2);
                    
                    if(symbolName.endsWith("_VW_ROWS")) {
                        newSymbolAddress = newSymbolAddress.add(1);
                        newSymbolName = newSymbolName + "ROWS_"; 
                    }    

                    if(symbolName.endsWith("_VW_COLS")) {
                        newSymbolAddress = newSymbolAddress.add(1);
                        newSymbolName = newSymbolName + "COLS_";                         
                    }
                
                    byte realValue           = getByte(newSymbolAddress);
                    int  expectedValue       = dataType.getLength();

                    if(realValue == expectedValue) {
                        Symbol newLabel = createLabel(newSymbolAddress, newSymbolName, true);
                        continue;                        
                    } else {
                        println("ERROR 2: " + symbol.getName() + " --> Header not found or incorrect");
                    }
                
                // relevant for non-distributed float maps
                } else if(dataType.getName().contains("float")) {

                    Address newSymbolAddress = symbolAddress.subtract(8);
                    
                    if(symbolName.endsWith("_VW_ROWS")) {
                        newSymbolAddress = newSymbolAddress.add(2);
                        newSymbolName = newSymbolName + "ROWS_"; 
                    }    

                    if(symbolName.endsWith("_VW_COLS")) {
                        newSymbolAddress = newSymbolAddress.add(2);
                        newSymbolName = newSymbolName + "COLS_";                         
                    }

                    byte realValue     = getByte(newSymbolAddress);
                    int  expectedValue = dataType.getLength() / 4; // one float is four bytes long

                    if(realValue == expectedValue) {
                        Symbol newLabel = createLabel(newSymbolAddress, newSymbolName, true);
                        continue;
                    } else {
                        println("ERROR 3: " + symbol.getName() + " --> Header not found or incorrect");
                    }
                    
                } else {
                    println("ERROR 4: " + symbol.getName() + " --> Data type not implemented");
                } 
            
            // relevant for non-distributed short curves
            } else if(symbolName.endsWith("CUR_COLS")) {
                DataType dataType = listing.getDataAt(symbolAddress).getDataType();    
            
                String   newSymbolName = symbolName.substring(0, symbolName.length() - 4);
                    
                if(dataType.getName().contains("short")) {

                    Address newSymbolAddress = symbolAddress.subtract(2); 
                    byte realValue     = getByte(newSymbolAddress);
                    int  expectedValue = dataType.getLength() / 2; // one short is two bytes long
                        
                    if(realValue == expectedValue) {
                        Symbol newLabel = createLabel(newSymbolAddress, newSymbolName, true);
                        continue;
                    } else {
                        println("ERROR 5: " + symbol.getName() + " --> Header not found or incorrect");

                    }

                // relevant for non-distributed byte curves
                } else if(dataType.getName().contains("byte")) {

                    Address newSymbolAddress = symbolAddress.subtract(1);    
                    byte realValue    = getByte(newSymbolAddress);
                    int expectedValue = dataType.getLength();
                        
                    if(realValue == expectedValue) {
                        Symbol newLabel = createLabel(newSymbolAddress, newSymbolName, true);
                        continue;
                    } else {
                        println("ERROR 6: " + symbol.getName() + " --> Header not found or incorrect");
                    }

                // relevant for non-distributed float curves
                } else if(dataType.getName().contains("float")) {

                    Address newSymbolAddress = symbolAddress.subtract(4);    
                    byte realValue    = getByte(newSymbolAddress);
                    int expectedValue = dataType.getLength() / 4; // one float is 4 bytes long
                        
                    if(realValue == expectedValue) {
                        Symbol newLabel = createLabel(newSymbolAddress, newSymbolName, true);
                        continue;
                    } else {
                        println("ERROR 7: " + symbol.getName() + " --> Header not found or incorrect");
                    }

                } else {
                    println("ERROR 8: " + symbol.getName() + " --> Data type not implemented");
                }

            } else {
                continue;
            } // end if
        } // end while 
    } // end run
} // end class