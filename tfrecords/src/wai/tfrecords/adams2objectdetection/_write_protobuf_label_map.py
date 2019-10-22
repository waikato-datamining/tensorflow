from typing import Dict, List


def write_protobuf_label_map(label_map: Dict[str, int], filename: str):
    """
    Writes the label-to-index mapping to the given file, in
    protobuf format.

    :param label_map:   The mapping from labels to indices.
    :param filename:    The file to write the mapping to.
    """
    # Format the label index map
    protobuf: List[str] = ["item {\n" +
                           f"  id: {index}\n" +
                           f"  name: '{label}'\n" +
                           "}\n"
                           for label, index in label_map.items()]

    # Write the lines to the specified file
    with open(filename, 'w') as file:
        file.writelines(protobuf)
