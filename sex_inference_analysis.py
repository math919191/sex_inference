import argparse
from pathlib import Path
from typing import TypedDict

import pandas as pd

from CREST_sex_inference import read_simmap
from sex_inference_analysis_probabilities import calc_lod_probability
from utils import read_in_seg_file_as_df

# ----------------------- utils ----------------------------

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


def is_beginning_or_end_of_chromosome(current_row, simmap):
    def get_chromo_start(chromo, simmap):
        return simmap[str(chromo)]['pos'][0]

    def get_chromo_end(chromo, simmap):
        return simmap[str(chromo)]['pos'][-1]

    starts_chromosome = current_row['start'] == get_chromo_start(current_row['chromosome'], simmap)
    ends_chromosome = current_row['end'] == get_chromo_end(current_row['chromosome'], simmap)

    return starts_chromosome, ends_chromosome

# ----------------------- end utils ----------------------------

def transform_to_co_and_gaps_df(relatives_df, simmap):

    co_and_gaps = []

    for index in range(len(relatives_df)):
        curr_row = relatives_df.iloc[index]

        seg_starts_chromosome, seg_ends_chromosome = is_beginning_or_end_of_chromosome(curr_row, simmap)

        starting_crossover = {'sample_1': curr_row['sample_1'],
                              'sample_2': curr_row['sample_2'],
                              'chromosome': curr_row['chromosome'],
                              'start': curr_row['start'],
                              'end': curr_row['start'],
                              'is_co': True,
                              'breakpoint_overlaps': curr_row['start_overlaps'],
                              'overlaps_anywhere': curr_row['overlaps_p1'],
                              'starts_or_ends_chromosome': seg_starts_chromosome
                              }

        ending_crossover = {'sample_1': curr_row['sample_1'],
                            'sample_2': curr_row['sample_2'],
                            'chromosome': curr_row['chromosome'],
                            'start': curr_row['end'],
                            'end': curr_row['end'],
                            'is_co': True,
                            'breakpoint_overlaps': curr_row['end_overlaps'],
                            'overlaps_anywhere': curr_row['overlaps_p1'],
                            'starts_or_ends_chromosome': seg_ends_chromosome
                            }

        seg_gap = {'sample_1': curr_row['sample_1'],
                   'sample_2': curr_row['sample_2'],
                   'chromosome': curr_row['chromosome'],
                   "start": curr_row['start'],
                   "end": curr_row['end'],
                   'is_co': False,
                   'breakpoint_overlaps': False,
                   'overlaps_anywhere': curr_row['overlaps_p1'],
                   'starts_or_ends_chromosome': False
                   }

        if curr_row['IBD_type'] == 'IBD2':
            continue

        co_and_gaps.append(starting_crossover)
        co_and_gaps.append(ending_crossover)
        co_and_gaps.append(seg_gap)

    return pd.DataFrame(co_and_gaps)


def overlaps(ibd_seg_1, ibd_seg_2):
    return ibd_seg_1['start'] <= ibd_seg_2['end'] and ibd_seg_1['end'] >= ibd_seg_2['start']

def breakpoint_overlaps(breakpoint, other_ibd_seg):
    return other_ibd_seg['start'] < breakpoint < other_ibd_seg['end']

def seg_overlaps_with_other_ibd_df(seg, other_ibd_df):
    other_ibd_df = other_ibd_df[other_ibd_df['chromosome'] == seg['chromosome']]

    for _, other_ibd_seg in other_ibd_df.iterrows():
        # assumes other_ibd_df is sorted
        if seg['end'] < other_ibd_seg['start']:
            break

        if overlaps(seg, other_ibd_seg):
            return True

    return False

def breakpoint_overlaps_with_other_ibd_df(breakpoint, chromosome, other_ibd_df):
    other_ibd_df = other_ibd_df[other_ibd_df['chromosome'] == chromosome]

    for _, other_ibd_seg in other_ibd_df.iterrows():
        if breakpoint_overlaps(breakpoint, other_ibd_seg):
            return True

        # assumes other_ibd_df is sorted
        if breakpoint < other_ibd_seg['start']:
            break

    return False



def add_overlap_col_to_ibd_seg_df(ibd_seg_df, ibd_segs_other, overlaps_col_name='overlaps'):
    # sort the rows by chromosome/start/end

    ibd_seg_df = ibd_seg_df.sort_values(by=['chromosome', 'start', 'end'], ascending=True)
    ibd_segs_other = ibd_segs_other.sort_values(by=['chromosome', 'start', 'end'], ascending=True)

    ibd_segs_other_chromo_dict = {
        chromosome : ibd_segs_other[ibd_segs_other['chromosome'] == chromosome]
        for chromosome in range(1, 23)
    }

    ibd_seg_df[overlaps_col_name] = ibd_seg_df.apply(
        lambda row: bool(seg_overlaps_with_other_ibd_df(row, ibd_segs_other_chromo_dict[row['chromosome']])), axis=1
    )

    ibd_seg_df["start_overlaps"] = ibd_seg_df.apply(
        lambda row: bool(breakpoint_overlaps_with_other_ibd_df(row['start'], row['chromosome'], ibd_segs_other_chromo_dict[row['chromosome']])), axis=1
    )

    ibd_seg_df["end_overlaps"] = ibd_seg_df.apply(
        lambda row: bool(breakpoint_overlaps_with_other_ibd_df(row['end'], row['chromosome'], ibd_segs_other_chromo_dict[row['chromosome']])), axis=1
    )

    return ibd_seg_df

def get_ibd_segs_between_relatives(seg_df, set1_relatives, set2_relatives):
    return seg_df[
        (seg_df['sample_1'].isin(set(set1_relatives)) & seg_df['sample_2'].isin(set(set2_relatives))) |
        (seg_df['sample_1'].isin(set(set2_relatives)) & seg_df['sample_2'].isin(set(set1_relatives)))
      ]

def get_all_relative_ids(pedigree_id, seg_df):
    s1_ids = seg_df.loc[seg_df['sample_1'].str.startswith(f'{pedigree_id}_'), 'sample_1'].tolist()
    s2_ids = seg_df.loc[seg_df['sample_2'].str.startswith(f'{pedigree_id}_'), 'sample_2'].tolist()
    all_ids = set(s1_ids + s2_ids)
    return list(all_ids)


def determine_descendants_and_relatives(pedigree_id, seg_df):

    all_relative_ids = get_all_relative_ids(pedigree_id, seg_df)

    descendants = [f'{pedigree_id}_g5-b1-i1', f'{pedigree_id}_g5-b2-i1', f'{pedigree_id}_g5-b3-i1']
    p1_relatives = list(set(all_relative_ids) - set(descendants))

    return descendants, p1_relatives


def write_df_output(results, output_file):
    print(f"writing to {output_file}")
    results.to_csv(output_file, sep='\t', index=False, header=True)


def get_pedigree_ids(df):
    pedigree_ids = df['sample_1'].str.split("_").str[0].unique()
    return pedigree_ids


COLUMN_NAMES = "sample_1", "sample_2", "chromosome", "start", "end", "IBD_type", "genetic_pos_start", "genetic_pos_end", "genetic_len"
COLUMNS_TO_DROP = ["genetic_pos_start", "genetic_pos_end", "genetic_len"]
SORT_ORDER = ["sample_1", "sample_2", "chromosome", "start"]

column_type_casting = {
    'chromosome': int,
    'start': int,
    'end': int,
}

def main(
        input_file,
        map_file,
        output_file = 'sig_co_gaps_results.csv'
):
    simmap = read_simmap(map_file)
    seg_df = read_in_seg_file_as_df(input_file,
                                    column_names=COLUMN_NAMES,
                                    columns_to_drop=COLUMNS_TO_DROP,
                                    sort_order=SORT_ORDER,
                                    column_type_casting=column_type_casting)

    pedigree_ids = get_pedigree_ids(seg_df)

    df_results = []

    for pedigree_id in pedigree_ids:

        descendants_ids, p1_relatives_ids = determine_descendants_and_relatives(pedigree_id, seg_df)

        pedigree_df = seg_df[seg_df['sample_1'].str.contains(fr'^{pedigree_id}_')]

        # get any ibd between any of the descendants (these are the cousin IBDs)
        descendants_df = get_ibd_segs_between_relatives(pedigree_df, descendants_ids, descendants_ids)

        # get any ibd between any of the descendants and distant relatives (p1 relatives)
        descendants_and_p1_relatives_df = get_ibd_segs_between_relatives(pedigree_df, descendants_ids, p1_relatives_ids)

        descendants_df = add_overlap_col_to_ibd_seg_df(descendants_df, descendants_and_p1_relatives_df, overlaps_col_name='overlaps_p1')

        co_and_gaps_df = transform_to_co_and_gaps_df(descendants_df, simmap=simmap)

        df_results.append(co_and_gaps_df)

        print(pedigree_id)

    df_results = pd.concat(df_results, ignore_index=True)

    write_df_output(df_results, output_file)


class AnalysisParams(TypedDict):
    window: int
    count_ambiguous_co: bool
    remove_dups: bool
    count_p1: bool
    count_p2: bool

def read_in_results_file_as_df(file):
    results_df = pd.read_csv(file, sep='\t')
    return results_df

def get_pedigree_id_from_file(file):
    return file.split('_')[-1].split(".")[0]

def do_data_analysis(input_file, output_file, simmap, details):
    simmap = read_simmap(simmap)

    all_pedigrees_df = read_in_results_file_as_df(input_file)
    all_pedigrees_df = all_pedigrees_df.astype(column_type_casting)

    pedigree_ids = get_pedigree_ids(all_pedigrees_df)

    # determine the subset of pedigree ids we want to analyze
    num_to_analyze = details.get('analysis_count', len(pedigree_ids))
    female_start = len(pedigree_ids) // 2
    pedigree_ids = list(pedigree_ids[:num_to_analyze]) + list(pedigree_ids[female_start:female_start+num_to_analyze])

    data_analysis_results = []

    for pedigree_id in pedigree_ids:

        relative_ids = get_all_relative_ids(pedigree_id, all_pedigrees_df)
        single_pedigree_df = all_pedigrees_df[all_pedigrees_df['sample_1'].isin(relative_ids) | all_pedigrees_df['sample_2'].isin(relative_ids)]

        # remove any instances where it begins/ends the chromosome (it isn't a crossover we care about)
        single_pedigree_df = single_pedigree_df[single_pedigree_df['starts_or_ends_chromosome'] == False]

        # only include any ibd co/gaps that overlap with the relatives of the focal grandparent
        single_pedigree_df = single_pedigree_df[single_pedigree_df['overlaps_anywhere'] == True]

        # if indicated, remove any duplicate crossovers
        if details.get('drop_duplicates', False) == True:
            single_pedigree_df = single_pedigree_df.drop_duplicates(subset=["chromosome", "start", "end", "is_co"])

        # if indicated, only count the crossovers that overlaps a distant relative segment
        if details.get('only_count_overlapping_ends', False):
            # drop only those rows where it is a crossover AND it does not directly overlap
            # keep everything except rows where cond1 is True AND cond2 is False (df = df.loc[(~cond1) | cond2])
            single_pedigree_df = single_pedigree_df[ ~(single_pedigree_df['overlaps_anywhere']) | (single_pedigree_df['breakpoint_overlaps'])]

        # Calculate the score for m/f crossover (higher = more likely female) by adding a column with the LOD
        single_pedigree_df['LOD'] = single_pedigree_df.apply(calc_lod_probability, axis=1, simmap=simmap, window_size=details.get('window_size', 500))

        LOD = sum(single_pedigree_df['LOD'])

        data_analysis_results.append({'pedigree_id': pedigree_id,
                                      'LOD': LOD}
                                     )
        print(pedigree_id)

    results_df = pd.DataFrame(data_analysis_results)

    results_df.to_csv(output_file, sep='\t', index=False, header=True)

    return results_df