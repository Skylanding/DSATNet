import argparse as ag
import json
import os

def get_parser_with_args(metadata_json='metadata.json'):
    parser = ag.ArgumentParser(description='Training change detection network')
    if not os.path.exists(metadata_json):
        print('No such file or directory: {0}'.format(metadata_json))
        return None
    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        return parser, metadata
