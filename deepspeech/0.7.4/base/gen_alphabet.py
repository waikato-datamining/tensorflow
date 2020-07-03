#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv

def generate_alphabet(csv_files, alphabet, column="transcript", verbose=False):
    chars = set()
    for csv_file in csv_files:
        if verbose:
            print("Processing %s..." % csv_file)
        with open(csv_file) as cf:
            reader = csv.DictReader(cf)
            for row in reader:
                if column in row:
                    chars |= set(row[column])
    sorted = list(chars)
    sorted.sort()
    if verbose:
        print("Alphabet:", sorted)
    with open(alphabet, "w") as af:
        for c in sorted:
            af.write(c)
            af.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Splits audio files on pauses.')
    parser.add_argument('csv_files', metavar='FILE', type=str, nargs='+', help="The CSV files to generate the alphabet from")
    parser.add_argument('--alphabet', required=True, help='The output file for storing the alphabet')
    parser.add_argument('--column', required=False, default="transcript", type=str, help='The column containing the transcript text')
    parser.add_argument('--verbose', required=False, action='store_true', help='Whether to output some debugging information')
    parsed = parser.parse_args()
    generate_alphabet(parsed.csv_files, parsed.alphabet, column=parsed.column, verbose=parsed.verbose)


if __name__ == '__main__':
    main()
