#!/usr/bin/python3

'''
@ author MJubke
@ date 2023-04

Command line tool that takes as input a .binary file and a manually transfered .winols.csv file, 
then reads missing values in the .winols.csv file from the .binary file and then writes a .winols.csv 
file in which the missing values are present
'''

import argparse
import pandas as pd
from pauschalgutachten.auto_analysis import loader

def main():
    '''
    Main routine
    '''

    # parse arguments
    args = parse_arguments()

    # prepare data 
    binary       = loader.Binary(args.binary_path)
    csv          = pd.DataFrame(loader.load_winols_file(args.winols_csv_path))
    outfile_path = args.outfile_path
    
    # check csv and delete columns with identifier "Unnamed: ..."
    num_damaged_values = len(csv[csv["Feldwerte.Werte"] == "map_transferred_with_ghidra"])
    print(f"*** Found {num_damaged_values} damaged values in the input .winols.csv")
    csv.drop(list(csv.filter(regex = 'Unnamed')), axis=1, inplace=True)

    # build outfile
    num_repaired_values = 0
    
    for index, row in csv.iterrows():
        
        if row["Feldwerte.Werte"] == "map_transferred_with_ghidra":

            # in the following lines, the " ".join(str(values)[1:-1].split()) construction cuts off the
            # brackets and reduces any whitespace between two numbers to 1 to match the .winols.csv format
            
            fieldvalues = extract_missing_values_from_binary(row, binary)
            if row["bKehrwert"] == 1:
                fieldvalues = 1 / fieldvalues
            fieldvalues = fieldvalues * row["Feldwerte.Faktor"]
            fieldvalues = fieldvalues + row["Feldwerte.Offset"]
            
            csv.at[index, "Feldwerte.Werte"]    = " ".join(str(fieldvalues)[1:-1].split())
            csv.at[index, "Fieldvalues.Values"] = " ".join(str(fieldvalues)[1:-1].split())

            # most of the axis values for constants (which are actually not needed) already have the value 48
            # so we set the rest of them to 48 as well
            if row["Spalten"] == 1 and row["Zeilen"] == 1:
                csv.at[index, "StuetzX.Werte"] = "48"
                csv.at[index, "AxisX.Values"]  = "48"
                csv.at[index, "StuetzY.Werte"] = "48"
                csv.at[index, "AxisY.Values"]  = "48"

            # x-axis values need to be extracted for curves
            # y-axis values (which are actuelly not needed) are set to 48
            if row["Spalten"] > 1 and row["Zeilen"] == 1:
                
                x_axis = extract_missing_xaxis_from_binary(row, binary)
                if row["StuetzX.bKehrwert"] == 1:
                    x_axis = 1 / x_axis
                x_axis = x_axis * row["StuetzX.Faktor"]
                x_axis = x_axis + row["StuetzX.Offset"]
                
                csv.at[index, "StuetzX.Werte"] = " ".join(str(x_axis)[1:-1].split())
                csv.at[index, "AxisX.Values"]  = " ".join(str(x_axis)[1:-1].split())
                csv.at[index, "StuetzY.Werte"] = "48"
                csv.at[index, "AxisY.Values"]  = "48"

            # x-axis and y-axis values need to be extracted for maps
            if row["Spalten"] > 1 and row["Zeilen"] > 1:
                
                x_axis = extract_missing_xaxis_from_binary(row, binary)
                if row["StuetzX.bKehrwert"] == 1:
                    x_axis = 1 / x_axis
                x_axis = x_axis * row["StuetzX.Faktor"]
                x_axis = x_axis + row["StuetzX.Offset"]
                
                y_axis = extract_missing_yaxis_from_binary(row, binary)
                if row["StuetzY.bKehrwert"] == 1:
                    y_axis = 1 / y_axis
                y_axis = y_axis * row["StuetzY.Faktor"]
                y_axis = y_axis + row["StuetzY.Offset"]

                csv.at[index, "StuetzX.Werte"] = " ".join(str(x_axis)[1:-1].split())
                csv.at[index, "AxisX.Values"]  = " ".join(str(x_axis)[1:-1].split())
                csv.at[index, "StuetzY.Werte"] = " ".join(str(y_axis)[1:-1].split())
                csv.at[index, "AxisY.Values"]  = " ".join(str(y_axis)[1:-1].split())

            num_repaired_values += 1

    print(f"*** Repaired {num_repaired_values}/{num_damaged_values} damaged values")

    # write outfile
    csv.to_csv(outfile_path, sep=';', index=False)
    print(f"*** Saved repaired .winols.csv to {outfile_path}")


def extract_missing_xaxis_from_binary(row, binary):
    '''
    Method to extract missing x-axis values (columns) in the csv from the related binary
    '''

    num_values = row["Spalten"]

    if row["StuetzX.DataOrg"] == "eByte" and row["StuetzX.bVorzeichen"] == 1:
        return binary.sbyte(row["StuetzX.DataAddr"], num_values)
    
    if row["StuetzX.DataOrg"] == "eByte" and row["StuetzX.bVorzeichen"] == 0:
        return binary.ubyte(row["StuetzX.DataAddr"], num_values)
    
    if row["StuetzX.DataOrg"] == "eLoHi" and row["StuetzX.bVorzeichen"] == 1:
        return binary.sint16(row["StuetzX.DataAddr"], num_values)
    
    if row["StuetzX.DataOrg"] == "eLoHi" and row["StuetzX.bVorzeichen"] == 0:
        return binary.uint16(row["StuetzX.DataAddr"], num_values)
    
    if row["StuetzX.DataOrg"] == "eLoHiLoHi" and row["StuetzX.bVorzeichen"] == 1:
        return binary.sint32(row["StuetzX.DataAddr"], num_values)
    
    if row["StuetzX.DataOrg"] == "eLoHiLoHi" and row["StuetzX.bVorzeichen"] == 0:
        return binary.uint32(row["StuetzX.DataAddr"], num_values)
    
    if row["StuetzX.DataOrg"] == "eFloatLoHi":
        return binary.float32(row["StuetzX.DataAddr"], num_values)
    
    raise NotImplementedError(
        f'Not implemented for StuetzX.DataOrg={row["StuetzX.DataOrg"]} and StuetzX.bVorzeichen={row["StuetzX.bVorzeichen"]}'
    )


def extract_missing_yaxis_from_binary(row, binary):
    '''
    Method to extract missing y-axis values (rows) in the csv from the related binary
    '''

    num_values = row["Zeilen"]

    if row["StuetzY.DataOrg"] == "eByte" and row["StuetzY.bVorzeichen"] == 1:
        return binary.sbyte(row["StuetzY.DataAddr"], num_values)
    
    if row["StuetzY.DataOrg"] == "eByte" and row["StuetzY.bVorzeichen"] == 0:
        return binary.ubyte(row["StuetzY.DataAddr"], num_values)
    
    if row["StuetzY.DataOrg"] == "eLoHi" and row["StuetzY.bVorzeichen"] == 1:
        return binary.sint16(row["StuetzY.DataAddr"], num_values)
    
    if row["StuetzY.DataOrg"] == "eLoHi" and row["StuetzY.bVorzeichen"] == 0:
        return binary.uint16(row["StuetzY.DataAddr"], num_values)
    
    if row["StuetzY.DataOrg"] == "eLoHiLoHi" and row["StuetzY.bVorzeichen"] == 1:
        return binary.sint32(row["StuetzY.DataAddr"], num_values)
    
    if row["StuetzY.DataOrg"] == "eLoHiLoHi" and row["StuetzY.bVorzeichen"] == 0:
        return binary.uint32(row["StuetzY.DataAddr"], num_values)
    
    if row["StuetzY.DataOrg"] == "eFloatLoHi":
        return binary.float32(row["StuetzY.DataAddr"], num_values)
    
    raise NotImplementedError(
        f'Not implemented for StuetzY.DataOrg={row["StuetzY.DataOrg"]} and StuetzY.bVorzeichen={row["StuetzY.bVorzeichen"]}'
    )


def extract_missing_values_from_binary(row, binary):
    '''
    Method to extract missing values in the csv from the related binary
    '''

    num_values = row["Spalten"] * row["Zeilen"]
    
    if row["DataOrg"] == "eByte" and row["bVorzeichen"] == 1:
        return binary.sbyte(row["Feldwerte.StartAddr"], num_values)
    
    if row["DataOrg"] == "eByte" and row["bVorzeichen"] == 0:
        return binary.ubyte(row["Feldwerte.StartAddr"], num_values)
    
    if row["DataOrg"] == "eLoHi" and row["bVorzeichen"] == 1:
        return binary.sint16(row["Feldwerte.StartAddr"], num_values)
    
    if row["DataOrg"] == "eLoHi" and row["bVorzeichen"] == 0:
        return binary.uint16(row["Feldwerte.StartAddr"], num_values)
    
    if row["DataOrg"] == "eLoHiLoHi" and row["bVorzeichen"] == 1:
        return binary.sint32(row["Feldwerte.StartAddr"], num_values)
    
    if row["DataOrg"] == "eLoHiLoHi" and row["bVorzeichen"] == 0:
        return binary.uint32(row["Feldwerte.StartAddr"], num_values)
    
    if row["DataOrg"] == "eFloatLoHi":
        return binary.float32(row["Feldwerte.StartAddr"], num_values)
    
    raise NotImplementedError(
        f'Not implemented for DataOrg={row["DataOrg"]} and bVorzeichen={row["bVorzeichen"]}'
    )


def parse_arguments():
    '''
    Method to parse the command line arguments
    '''
    parser = argparse.ArgumentParser(
        description="Command line tool to repair manually transferred .winols.csv files",
        epilog="Synopsis: repair_manually_transfered_csv.py <.binary path> <.winols.csv path> <outfile path>"
    )
    parser.add_argument(
        "binary_path",
        help="Binary file from which to extract the missing values",
        type=str
    )
    parser.add_argument(
        "winols_csv_path",
        help="'.winols.csv' file that contains 'map_transferred_with_ghidra' entries",
        type=str
    )
    parser.add_argument(
        "outfile_path",
        help="Path / name of the new .winols.csv file",
        type=str
    )

    return parser.parse_args()


# call main
if __name__ == "__main__":
    main()