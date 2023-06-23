#!/usr/bin/python3

'''
@ author MJubke
@ date 2023-04

Command line tool that takes as input a source .binary file and one or more target binary files.
The source is then compared chunk wise against the targets.
The grade of similarity is then printed as a percentage to the command line.
'''

import argparse
import os

def main():
    '''
    Main routine
    '''

    # parse arguments
    args = parse_arguments()

    # prepare data 
    source_binary   = args.source_binary
    target_binaries = args.target_binaries
    chunk_size      = args.chunk_size

    # compare binaries
    for i in range(len(target_binaries)):
        
        target_binary    = target_binaries[i]
        total_chunks     = 0
        identical_chunks = 0
    
        with open(source_binary, 'rb') as binary_1, open(target_binary, 'rb') as binary_2:
            while True:
                chunk_1 = binary_1.read(chunk_size)
                chunk_2 = binary_2.read(chunk_size)
                if not chunk_1 or not chunk_2:
                    break
                total_chunks += 1
                if chunk_1 == chunk_2:
                    identical_chunks += 1

        if total_chunks == 0:
            percentage = 0.0
        else:
            percentage = identical_chunks / total_chunks * 100

        print(f"Source and target binary are {percentage:>6.2f}% identical for target {target_binary}")        


def parse_arguments():
    '''
    Method to parse the command line arguments
    '''
    parser = argparse.ArgumentParser(
        description="Command line tool to compare two binaies and give their grade of similarity in percent",
        epilog="Synopsis: compare_binaries.py [--chunk_size int] <source .binary path> <first target .binary path> [<second target .binary path> ... ]"
    )
    parser.add_argument(
        "-cs",
        "--chunk_size",
        help="Size of the binary chunks that are compared",
        default=1024,
        type=int,
    )
    parser.add_argument(
        "source_binary",
        help="Source .binary file",
        type=str
    )
    parser.add_argument(
        "target_binaries",
        help="One or more target .binary files",
        nargs='*',
        type=str
    )
    
    args = parser.parse_args()
    
    assert args.chunk_size > 0, "Chunk size has to be > 0"
    assert os.path.exists(args.source_binary), f"Path to source binary {args.source_binary} could not be resolved"
    for target_binary in args.target_binaries:
        assert os.path.exists(target_binary), f"Path to target binary {target_binary} could not be resolved"
    
    return args


# call main
if __name__ == "__main__":
    main()