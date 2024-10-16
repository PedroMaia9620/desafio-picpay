import sys
import os
sys.path.insert(1, os.getcwd())
import logging
import argparse
import pandas as pd 

def main(raw_dataset_filename):

    df = pd.read_csv(f'data/raw/{raw_dataset_filename}.csv')
    df.to_csv(f'data/interim/{raw_dataset_filename}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=''
    )

    parser.add_argument('-r', '--raw_dataset_filename', help='raw dataset filename')
    
    args = parser.parse_args()
    main(args.raw_dataset_filename)