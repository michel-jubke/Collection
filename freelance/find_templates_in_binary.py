#!/usr/bin/python3

from pauschalgutachten.auto_analysis import loader
from pauschalgutachten.auto_analysis import dtype_info
import pandas as pd
import argparse
import os
from pathlib import Path


def main():
    '''
    Main routine
    '''

    # parse args
    args = parse_arguments()
    
    binary_path    = Path(args.binary_path).resolve()
    template_paths = get_template_paths(args.template_folder)         
    info_path      = get_info_path(args.template_folder)
    
    #print(template_paths)
    #print(info_path)
    
    binary = binary_path.read_bytes()
    
    for path in template_paths:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        template = Template(path, info_path)
        
        print(f"Template {template.name} was found at addresses")
    
        addresses = check_occurence(template.axis_and_fieldvalues, binary)
        for address in addresses:
            print(address)
        

############################################################################################################### 

def check_occurence(pattern, binary):
    start_address = 0
    while True:
        try:
            address = binary.index(pattern, start_address)
            yield address
            start_address = address + 1
        except ValueError:
            break


###############################################################################################################

class Template():

    name                 = None
    abs_path             = None
    info_abs_path        = None
    dtype_length         = None
    rows                 = None
    cols                 = None
    x_axis_length        = None
    y_axis_length        = None
    fieldvalues_length   = None
    x_axis               = None
    y_axis               = None
    fieldvalues          = None
    axis_and_fieldvalues = None

    def __init__(self, path, info_path):
        self.abs_path = path
        self.info_abs_path = info_path

        template = loader.load_template(path, verbose=False)
        template_info = pd.read_csv(info_path,
                                    sep=';',
                                    engine='c',
                                    encoding='utf-8',
                                    dtype={
                                        **dtype_info.WINOLS_DTYPES,
                                        **dtype_info.ADDITIONAL_TEMPLATE_INFO_DTYPES
                                        }
                                    )
        template_info.drop_duplicates('IdName', keep='first', inplace=True)

        self.name = template.name

        data_org = template_info['DataOrg'][template_info['IdName'] == self.name].iloc[0]
        if data_org in ['eByte', 'eBitLoHi', 'eBitHiLo']:
            self.dtype_length = 1
        if data_org in ['eLoHi', 'eHiLo']:
            self.dtype_length = 2
        if data_org in ['eLoHiLoHi', 'eHiLoHiLo', 'eFloatLoHi', 'eFloatHiLo']:
            self.dtype_length = 4

        self.rows = template_info['Zeilen'][template_info['IdName'] == self.name].iloc[0]
        self.cols = template_info['Spalten'][template_info['IdName'] == self.name].iloc[0]
        
        if self.rows == 1 and self.cols == 1:
            self.fieldvalues_length = self.dtype_length
            self.fieldvalues = template.binary[-self.fieldvalues_length:]
            self.axis_and_fieldvalues = self.fieldvalues

        if self.rows == 1 and self.cols > 1:
            self.y_axis_length      = self.cols * self.dtype_length  
            self.fieldvalues_length = self.cols * self.dtype_length
            self.y_axis = template.binary[-(self.fieldvalues_length + self.y_axis_length):-self.fieldvalues_length]
            self.fieldvalues = template.binary[-self.fieldvalues_length:]
            self.axis_and_fieldvalues = template.binary[-(self.fieldvalues_length + self.y_axis_length):]

        if self.rows > 1 and self.cols > 1:
            self.y_axis_length      = self.cols * self.dtype_length  
            self.x_axis_length      = self.rows * self.dtype_length  
            self.fieldvalues_length = self.rows * self.cols * self.dtype_length        
            self.y_axis = template.binary[-(self.fieldvalues_length + self.y_axis_length + self.x_axis_length):-(self.fieldvalues_length + self.x_axis_length)]
            self.x_axis = template.binary[-(self.fieldvalues_length + self.y_axis_length):-self.fieldvalues_length]
            self.fieldvalues = template.binary[-self.fieldvalues_length:]
            self.axis_and_fieldvalues = template.binary[-(self.fieldvalues_length + self.y_axis_length + self.x_axis_length):]
            

###############################################################################################################

def parse_arguments():
    '''
    Method to parse the command line arguments
    '''
    
    parser = argparse.ArgumentParser(
        description="Command line tool to find a template in a binary",
        epilog="Synopsis: find_template_in_binary.py binary template"
    )
    
    parser.add_argument(
        "binary_path",
        help="Path to .binary file",
        type=str
    )
    parser.add_argument(
        "template_folder",
        help="Path to folder that contains one or more .template files and exactly one template_info.csv file",
        type=str
    )
    
    args = parser.parse_args()
    
    assert os.path.exists(args.binary_path),     f"Path to binary {args.binary_path} could not be resolved"
    assert os.path.exists(args.template_folder), f"Path to template {args.template_folder} could not be resolved"
    
    return args


def get_template_paths(dir_path):
    template_paths = []
    for file in os.listdir(dir_path):
        if file.endswith('.template'):
            template_paths.append(Path(os.path.join(dir_path, file)).resolve())
    return template_paths


def get_info_path(dir_path):
    for file in os.listdir(dir_path):
        if file == 'template_info.csv':    
            return Path(os.path.join(dir_path, file)).resolve()


###############################################################################################################

# call main
if __name__ == "__main__":
    main()