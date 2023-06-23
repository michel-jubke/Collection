#!/usr/bin/python3

'''
@ author MJubke
@ date 2023-02

Command line tool to extract constants from a given binary 
according to the specifications in a given .winols.csv file
'''
import sys
import argparse
import pathlib
import pandas as pd
from pauschalgutachten.auto_analysis import loader

def main():
    '''
    Main routine
    '''

    # set up argument parser
    parser = argparse.ArgumentParser(
        description="Command line tool to extract constants from a .binary + .winols.csv",
        epilog="Synopsis: plot_maps.py <binary-file> <winols.csv-file> <outfile> "
    )
    parser.add_argument(
        "binary_path",
        help="Binary that contains the constants",
        type=str
    )
    parser.add_argument(
        "winols_csv_path",
        help="'.winols.csv'-file, that describes the constants in the given binary",
        type=str
    )
    parser.add_argument(
        "outfile_path",
        help="File to which the extracted constants should be saved",
        type=str
    )
    parser.add_argument(
        "-od",
        "--outfile_delimiter", 
        help="Delimiter used for the outfile",
        default='\t',
        type=str
    )
    parser.add_argument(
        "-i",
        "--id_names",
        help="Last argument: one or more IdNames of the constants to be extracted",
        nargs='*',
        type=str
    )

    # parse arguments
    args = parser.parse_args()

    # prepare data
    binary_path       = pathlib.Path(args.binary_path).resolve()
    winols_csv_path   = pathlib.Path(args.winols_csv_path).resolve()
    outfile_path      = pathlib.Path(args.outfile_path).resolve()
    outfile_delimiter = args.outfile_delimiter 
    id_names          = args.id_names

    binary            = loader.Binary(binary_path)
    winols_csv        = pd.DataFrame(loader.load_winols_file(winols_csv_path))
    winols_csv_consts = winols_csv[winols_csv["IdName"].str.endswith("_C")]

    if id_names is not None:
        winols_csv_consts = winols_csv_consts[winols_csv_consts["IdName"].isin(id_names)]
        if len(winols_csv_consts) < len(id_names):
            id_names_series = pd.Series(id_names)
            found_mask = id_names_series.isin(winols_csv_consts["IdName"])
            not_found_idnames = id_names_series.loc[~found_mask]
            print("\nWARNING: The following IdNames were not found in the given '.winols.csv'-file: \n")
            for idname in not_found_idnames:
                print("\t", idname)
                print("\n")
        if len(winols_csv_consts) == 0:
            sys.exit()

    # functions to extract constants
    def extract_from_binary(row, binary=binary):
        '''
        This function reads bytes from the binary and returns them 
        interpreted according to the specifications given in the .winols.csv
        '''
        if row["DataOrg"] == "eByte" and row["bVorzeichen"] == 1:
            return binary.sbyte(row["Feldwerte.StartAddr"], 1)[0]
        if row["DataOrg"] == "eByte" and row["bVorzeichen"] == 0:
            return binary.ubyte(row["Feldwerte.StartAddr"], 1)[0]
        if row["DataOrg"] == "eLoHi" and row["bVorzeichen"] == 1:
            return binary.sint16(row["Feldwerte.StartAddr"], 1)[0]
        if row["DataOrg"] == "eLoHi" and row["bVorzeichen"] == 0:
            return binary.uint16(row["Feldwerte.StartAddr"], 1)[0]
        if row["DataOrg"] == "eLoHiLoHi" and row["bVorzeichen"] == 1:
            return binary.sint32(row["Feldwerte.StartAddr"], 1)[0]
        if row["DataOrg"] == "eLoHiLoHi" and row["bVorzeichen"] == 0:
            return binary.uint32(row["Feldwerte.StartAddr"], 1)[0]
        if row["DataOrg"] == "eFloatLoHi":
            return binary.float32(row["Feldwerte.StartAddr"], 1)[0]
        raise NotImplementedError(
                f'Not implemented for DataOrg={row["DataOrg"]} and bVorzeichen={row["bVorzeichen"]}'
            )

    def make_reciprocal(row):
        '''
        This function builds the reciprocal of the values extracted from the 
        binary according to the specifications given in the .winols.csv
        '''
        if row["bKehrwert"] == 1:
            return 1 / row["ValueFromBinary"]
        else:
            return row["ValueFromBinary"]

    def apply_factor(row):
        '''
        This function applies the factor to the values extracted from the 
        binary as given in the .winols.csv
        '''
        return row["ValueFromBinary"] * row["Feldwerte.Faktor"]

    def apply_offset(row):
        '''
        This function applies the offset to the values extracted from the 
        binary as given in the .winols.csv
        '''
        return row["ValueFromBinary"] + row["Feldwerte.Offset"]

    # extract constants
    winols_csv_consts["ValueFromBinary"] = winols_csv_consts.apply(lambda row: extract_from_binary(row), axis=1)
    winols_csv_consts["ValueFromBinary"] = winols_csv_consts.apply(lambda row: make_reciprocal(row), axis=1)
    winols_csv_consts["ValueFromBinary"] = winols_csv_consts.apply(lambda row: apply_factor(row), axis=1)
    winols_csv_consts["ValueFromBinary"] = winols_csv_consts.apply(lambda row: apply_offset(row), axis=1)

    # create outfile and save it
    outfile = winols_csv_consts[["IdName", "ValueFromBinary", "Feldwerte.Einheit"]]
    outfile.to_csv(outfile_path, sep=outfile_delimiter, index=False)


# call main
if __name__ == "__main__":
    main()
