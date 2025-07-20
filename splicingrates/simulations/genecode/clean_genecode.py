import pandas as pd
import numpy as np
import os
import argparse

import pandas as pd


def parse_attributes(attribute_string):
    # Split the string by ';'
    attributes = attribute_string.split(';')[:-1]
    # Initialize empty dictionary to store attribute-value pairs
    attribute_dict = {}
    # Iterate over each attribute
    for attribute in attributes:
        # Split the attribute into name and value
        name, value = attribute.strip().split(' ')
        # Remove quotes from value
        value = value.strip().strip('"')
        # Add to dictionary
        attribute_dict[name.strip()] = value
    # Convert dictionary to pandas Series
    series = pd.Series(attribute_dict)
    return series

def clean_genecode_data(fn, output_fn=None):
    df = pd.read_csv(fn, sep='\t', comment='#', header=None)
    df.columns = ['chr', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    # the columns attribute contains different sets of attributes separated by ;. We want to transform df such that each attribute is a column. If an attribute is missing in a gene, then we will fill it with NaN
    attributes = df['attribute'].apply(parse_attributes)
    df = pd.concat([df, attributes], axis=1)
    df.drop('attribute', axis=1, inplace=True)
    if output_fn is not None:
        df.to_csv(output_fn, sep='\t', index=False, header=True, compression='gzip')
    return df

def hist_feature_lengths(df):
    # first, fileter so that we only care about protein_coding genes
    df = df[df['gene_type'] == 'protein_coding']
    # first, draw the histogram of the lengths of the exonß
    exon_lengths = df[df['feature'] == 'exon']['end'] - df[df['feature'] == 'exon']['start']
    exon_lengths.hist(bins=100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean Genecode data')
    parser.add_argument('--genecode_fn', help='Genecode file')
    parser.add_argument('--output_fn', help='Output file')
    args = parser.parse_args()
    print('Done getting command-line arguments')
    clear_df = clean_genecode_data(args.genecode_fn, args.output_fn)
