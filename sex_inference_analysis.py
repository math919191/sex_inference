import argparse
from pathlib import Path
from typing import TypedDict

import pandas as pd

from CREST_sex_inference import read_simmap
from sex_inference_analysis_probabilities import calc_LOD, calc_probability_mf
from utils import read_in_seg_file_as_df, is_beginning_or_end_of_chromosome

COLUMN_NAMES = "sample_1", "sample_2", "chromosome", "start", "end", "IBD_type", "genetic_pos_start", "genetic_pos_end", "genetic_len"
COLUMNS_TO_DROP = ["genetic_pos_start", "genetic_pos_end", "genetic_len"]
SORT_ORDER = ["sample_1", "sample_2", "chromosome", "start"]

column_type_casting = {
    'chromosome': int,
    'start': int,
    'end': int,
}


def get_significant_crossovers_and_gaps(relatives_df, simmap):
    columns = ["sample_1", "sample_2", "chromosome", "start", "end", "is_co", 'from_p1', 'from_p2', 'ambiguous']

    sig_co_and_gaps = []

    for index in range(len(relatives_df)):
        curr_row = relatives_df.iloc[index]

        seg_starts_chromosome, seg_ends_chromosome = is_beginning_or_end_of_chromosome(curr_row, simmap)

        starting_crossover = {'sample_1': curr_row['sample_1'],
                              'sample_2': curr_row['sample_2'],
                              'chromosome': curr_row['chromosome'],
                              'start': curr_row['start'],
                              'end': curr_row['start'],
                              'is_co': True,
                              'from_p1': False,
                              'from_p2': False,
                              'ambiguous': False
                              }

        ending_crossover = {'sample_1': curr_row['sample_1'],
                            'sample_2': curr_row['sample_2'],
                            'chromosome': curr_row['chromosome'],
                            'start': curr_row['end'],
                            'end': curr_row['end'],
                            'is_co': True,
                            'from_p1': False,
                            'from_p2': False,
                            'ambiguous': False
                            }

        seg_gap = {'sample_1': curr_row['sample_1'],
                   'sample_2': curr_row['sample_2'],
                   'chromosome': curr_row['chromosome'],
                   "start": curr_row['start'],
                   "end": curr_row['end'],
                   'is_co': False,
                   'from_p1': False,
                   'from_p2': False,
                   'ambiguous': False
                   }

        if curr_row['IBD_type'] == 'IBD2':
            continue

        if curr_row['overlaps_p1'] and not curr_row['overlaps_p2']:

            if not seg_starts_chromosome:
                starting_crossover['from_p1'] = True
                sig_co_and_gaps.append(starting_crossover)

            if not seg_ends_chromosome:
                ending_crossover['from_p1'] = True
                sig_co_and_gaps.append(ending_crossover)

            seg_gap['from_p1'] = True
            sig_co_and_gaps.append(seg_gap)


        elif not curr_row['overlaps_p1'] and curr_row['overlaps_p2']:

            if not seg_starts_chromosome:
                starting_crossover['from_p2'] = True
                sig_co_and_gaps.append(starting_crossover)

            if not seg_ends_chromosome:
                ending_crossover['from_p2'] = True
                sig_co_and_gaps.append(ending_crossover)

            seg_gap['from_p2'] = True
            sig_co_and_gaps.append(seg_gap)

    return pd.DataFrame(sig_co_and_gaps)


def overlaps(ibd_seg_1, ibd_seg_2):
    ret = ibd_seg_1['start'] <= ibd_seg_2['end'] and ibd_seg_1['end'] >= ibd_seg_2['start']
    return ret

def seg_overlaps_with_other_ibd_df(seg, other_ibd_df):
    other_ibd_df = other_ibd_df[other_ibd_df['chromosome'] == seg['chromosome']]

    for _, other_ibd_seg in other_ibd_df.iterrows():
        # assumes other_ibd_df is sorted
        if seg['end'] < other_ibd_seg['start']:
            break

        if overlaps(seg, other_ibd_seg):
            return True

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

    return ibd_seg_df

def get_ibd_segs_between_relatives(seg_df, relatives):
    return seg_df[seg_df['sample_1'].isin(set(relatives)) & seg_df['sample_2'].isin(set(relatives))]

def get_ibd_segs_shared(seg_df, set1_relatives, set2_relatives):
    return seg_df[
        seg_df['sample_1'].isin(set(set1_relatives)) & seg_df['sample_2'].isin(set(set2_relatives)) |
        seg_df['sample_1'].isin(set(set2_relatives)) & seg_df['sample_2'].isin(set(set1_relatives))
      ]


def get_ibd_segs_with_relatives(seg_df, relatives):
    return seg_df[seg_df['sample_1'].isin(set(relatives)) | seg_df['sample_2'].isin(set(relatives))]


def get_all_relative_ids(pedigree_id, seg_df):
    s1_ids = seg_df.loc[seg_df['sample_1'].str.startswith(f'{pedigree_id}_'), 'sample_1'].tolist()
    s2_ids = seg_df.loc[seg_df['sample_2'].str.startswith(f'{pedigree_id}_'), 'sample_2'].tolist()
    all_ids = set(s1_ids + s2_ids)
    return list(all_ids)


def determine_descendants_and_relatives(pedigree_id, seg_df):

    all_relative_ids = get_all_relative_ids(pedigree_id, seg_df)

    descendants = [f'{pedigree_id}_g5-b1-i1', f'{pedigree_id}_g5-b2-i1', f'{pedigree_id}_g5-b3-i1']
    p1_relatives = list(set(all_relative_ids) - set(descendants))
    p2_relatives = []

    return descendants, p1_relatives, p2_relatives


def write_df_output(results, output_file):
    print(f"writing to {output_file}")
    results.to_csv(output_file, sep='\t', index=False, header=True)


def get_pedigree_ids(df):
    pedigree_ids = df['sample_1'].str.split("_").str[0].unique()
    return pedigree_ids


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

        descendants_ids, p1_relatives_ids, p2_relatives_ids = determine_descendants_and_relatives(pedigree_id, seg_df)

        descendants_df = get_ibd_segs_between_relatives(seg_df, descendants_ids)

        descendants_and_p1_relatives_df = get_ibd_segs_shared(seg_df, descendants_ids, p1_relatives_ids)
        descendants_and_p2_relatives_df = get_ibd_segs_shared(seg_df, descendants_ids, p2_relatives_ids)

        descendants_df = add_overlap_col_to_ibd_seg_df(descendants_df, descendants_and_p1_relatives_df, overlaps_col_name='overlaps_p1')
        descendants_df = add_overlap_col_to_ibd_seg_df(descendants_df, descendants_and_p2_relatives_df, overlaps_col_name='overlaps_p2')

        sig_co_and_gaps_df = get_significant_crossovers_and_gaps(descendants_df, simmap=simmap)

        print(sig_co_and_gaps_df)
        df_results.append(sig_co_and_gaps_df)

        print(pedigree_id)

    df_results = pd.concat(df_results, ignore_index=True)

    output_file_name = output_file

    write_df_output(df_results, output_file_name)

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

def do_data_analysis(input_file, data_analysis_params_lists, output_file, simmap):

    simmap = read_simmap(simmap)

    pedigrees_df = read_in_results_file_as_df(input_file)
    pedigrees_df = pedigrees_df.astype(column_type_casting)

    pedigree_ids = get_pedigree_ids(pedigrees_df)

    data_analysis_results = []

    for pedigree_id in pedigree_ids:

        relative_ids = get_all_relative_ids(pedigree_id, pedigrees_df)
        pedigree_df = pedigrees_df[pedigrees_df['sample_1'].isin(relative_ids) | pedigrees_df['sample_2'].isin(relative_ids)]

        pedigree_data_analysis_result = {
            "pedigree_id": pedigree_id,
        }


        # for index, params in enumerate(data_analysis_params_lists):
        #
        #     sig_co_gaps_df_copy = pedigree_df.copy()
        #
        #     sig_gaps_df = sig_co_gaps_df_copy[sig_co_gaps_df_copy['is_co'] == False].to_dict(orient="records")
        #     sig_co_df = sig_co_gaps_df_copy[sig_co_gaps_df_copy['is_co'] == True].to_dict(orient="records")
        #     # if params['remove_dups']:
        #     #     sig_co_gaps_df_copy = remove_dups(sig_co_gaps_df_copy)
        #
        #     if not params['count_p1']:
        #         # remove anyone that has p1
        #         sig_co_gaps_df_copy = sig_co_gaps_df_copy[sig_co_gaps_df_copy['from_p1'] == False]
        #
        #     if not params['count_p2']:
        #         # remove anyone that has p2
        #         sig_co_gaps_df_copy = sig_co_gaps_df_copy[sig_co_gaps_df_copy['from_p2'] == False]
        #
        #     if not params['count_ambiguous_co']:
        #         # remove any ambiguous crossovers
        #         sig_co_gaps_df_copy = sig_co_gaps_df_copy[sig_co_gaps_df_copy['ambiguous'] == False]

        index = 0
        params = data_analysis_params_lists[0]
        sig_gaps_df = pedigree_df[pedigree_df['is_co'] == False].to_dict(orient="records")
        sig_co_df = pedigree_df[pedigree_df['is_co'] == True].to_dict(orient="records")

        LOD = calc_probability_mf(sig_gaps_df, sig_co_df, simmap, window_size=params['window'])
        pedigree_data_analysis_result[f'LOD_{index}'] = LOD

        data_analysis_results.append(pedigree_data_analysis_result)

    results_df = pd.DataFrame(data_analysis_results)

    results_df.to_csv(output_file, sep='\t', index=False, header=True)


# if __name__ == "__main__":
#     import sys
#
#     # uncomment and modify to change the arguments here without using the command line
#     # input_name = "sibs-cousin-M"
#     # input_file = "../ped-sim/outputs/sibs-cousins-M.seg"
#     # sys.argv = ['sex_inference_analysis.py',
#     #             '--input', input_file,
#     #             '--output', f'results/{input_name}_results.txt',
#     #             '--simmap', '../ped-sim/refined_mf.simmap',
#     #             '--count_ibd2_co', 'False'
#     #             ]
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('-i', '--input', type=Path, help='Name of the segment input file')
#     parser.add_argument('-o', '--output', type=Path, help='Name of the output file')
#     parser.add_argument('-m', '--simmap', type=Path, help='Name of the genetic map file (should be in .simmap format)')
#     parser.add_argument('-c', '--count_ambiguous_co', type=bool, help='True/False - count ambigious ibd2 crossovers')
#
#     # optional
#     parser.add_argument('-b', '--bim',
#                         help='A PLINK .bim containing the dataset-specific map (should contain 22 autosomes)')
#     parser.add_argument('-w', '--window', metavar='window_size_in_kilobases',
#                         type=int, default=500, help='Window size in kilobases. Default: 500 kb')
#     parser.add_argument('-d', '--count_doubles',
#                         type=bool, default='True', help='Count double crossovers')
#
#     args = parser.parse_args()
#
#     main(
#         input_file=args.input,
#         output_file=args.output,
#         map_file=args.simmap,
#         window=args.window,
#         count_ambiguous_co=args.count_ambiguous_co,
#         count_doubles=args.count_doubles
#     )
