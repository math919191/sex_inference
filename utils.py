import numpy as np
import pandas as pd

from collections import Counter


def read_in_seg_file_as_df(seg_file, column_names, columns_to_drop, sort_order, column_type_casting):
    with open(seg_file) as f:
        file_lines = f.readlines()

    data = [
        line.split()
        for line in file_lines
    ]

    df = pd.DataFrame(np.array(data), columns=column_names)

    df.drop(columns_to_drop, axis='columns')

    df = df.astype(column_type_casting)

    # make sure the pandas df is sorted to how we want it to be sorted
    df.sort_values(by=sort_order)

    # check segment end position is greater than segment start position raise exception if not. (CREST does this)
    for _, segment in df.iterrows():
        if segment["end"] < segment["start"]:
            raise Exception("Physical position end less than physical position start")

    return df

def on_same_chromosome(seg1, seg2):
    return seg1['chromosome'] == seg2['chromosome']


def has_connected_ibd(prev_row, curr_row, next_row, IBD_type):
    start_connected, end_connected = False, False

    if (prev_row is not None and on_same_chromosome(prev_row, curr_row) and prev_row['IBD_type'] == IBD_type
            and prev_row['end'] + 1 == curr_row['start']):
        start_connected = True

    if (next_row is not None and on_same_chromosome(curr_row, next_row) and next_row['IBD_type'] == IBD_type
            and curr_row['end'] + 1 == next_row['start']):
        end_connected = True

    return start_connected, end_connected


def get_chromo_start(chrosome, simmap):
    return simmap[str(chrosome)]['pos'][0]


def get_chromo_end(chrosome, simmap):
    return simmap[str(chrosome)]['pos'][-1]


def is_beginning_or_end_of_chromosome(current_row, simmap):
    starts_chromosome = current_row['start'] == get_chromo_start(current_row['chromosome'], simmap)
    ends_chromosome = current_row['end'] == get_chromo_end(current_row['chromosome'], simmap)

    return starts_chromosome, ends_chromosome

def combine_adjacent_gaps(gaps):
    def is_touching(gap1, gap2):
        return gap1['chromosome'] == gap2['chromosome'] and gap1['end'] + 1 == gap2['start']

    gaps_adjoined = []
    for gap in gaps:
        if not gaps_adjoined:
            gaps_adjoined.append(gap)
        else:
            if is_touching(gaps_adjoined[-1], gap):
                gaps_adjoined[-1]['end'] = gap['end']
            else:
                gaps_adjoined.append(gap)

    return gaps_adjoined



def remove_dict_dups(list_of_dicts):
    return [dict(t) for t in {tuple(d.items()) for d in list_of_dicts}]

def remove_complete_duplicate_dict(list_of_dicts):
    ret = []

    counted_items = Counter([tuple(d.items()) for d in list_of_dicts])
    for dictionary, num_dicts in counted_items.items():
        if num_dicts == 1:
            ret.append(dict(dictionary))
    return ret
