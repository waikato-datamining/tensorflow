import argparse
import csv


def analyze(input, output):
    """
    Analyzes the input CSV files and stores the collected alphabet in the output file.

    :param input: the CSV input files
    :type input: list
    :param output: the alphabet file to save to
    :type output: str
    """
    alphabet = set()
    for input_file in input:
        print("Analyzing %s..." % input_file)
        with open(input_file, "r") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if "transcript" in row:
                    alphabet.update(row["transcript"])
    alphabet = list(alphabet)
    alphabet.sort()
    alphabet.insert(0, "Alphabet generate from CSV files:")
    for i, input_file in enumerate(input):
        alphabet.insert(i+1, "- %s" % input_file)
    alphabet.append("# end of alphabet")
    with open(output, "w") as fp:
        fp.write("\n".join(alphabet))


def main():
    parser = argparse.ArgumentParser(description="Generates an alphabet file from the transcripts stored in Coqui STT CSV files.", prog="stt_alphabet")
    parser.add_argument("-i", "--input", nargs="+", required=True, help="The CSV file(s) to analyze")
    parser.add_argument("-o", "--output", required=True, help="The file name of the alphabet file to output")
    parsed = parser.parse_args()
    analyze(parsed.input, parsed.output)


if __name__ == "__main__":
    main()
