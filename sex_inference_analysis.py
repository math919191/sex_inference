import argparse
from pathlib import Path

import pandas as pd

from CREST_sex_inference import read_simmap
from sex_inference_analysis_probabilities import calc_probability_using_both_parents
from utils import read_in_seg_file_as_df, remove_dict_dups, combine_adjacent_gaps, is_beginning_or_end_of_chromosome, \
    has_connected_ibd

COLUMN_NAMES = "sample_1", "sample_2", "chromosome", "start", "end", "IBD_type", "genetic_pos_start", "genetic_pos_end", "genetic_len"
COLUMNS_TO_DROP = ["genetic_pos_start", "genetic_pos_end", "genetic_len"]
SORT_ORDER = ["sample_1", "sample_2", "chromosome", "start"]

column_type_casting = {
    'chromosome': int,
    'start': int,
    'end': int,
}


def get_significant_crossovers_and_gaps(df, simmap, count_ambiguous_co, count_doubles, count_ibd1, count_ibd2,
                                        count_non_ambi, remove_dups):
    sig_crossovers_p1 = []
    sig_crossovers_p2 = []
    sig_gaps_p1 = []
    sig_gaps_p2 = []

    for index in range(len(df)):
        curr_row = df.iloc[index]
        prev_row = None if index == 0 else df.iloc[index - 1]
        next_row = None if index == len(df) - 1 else df.iloc[index + 1]

        seg_starts_chromosome, seg_ends_chromosome = is_beginning_or_end_of_chromosome(curr_row, simmap)

        starting_crossover = {'chromosome': curr_row['chromosome'], 'position': curr_row['start']}
        ending_crossover = {'chromosome': curr_row['chromosome'], 'position': curr_row['end']}
        seg_gap = {'chromosome': curr_row['chromosome'], "start": curr_row['start'], "end": curr_row['end']}

        if curr_row['IBD_type'] == 'IBD1' and count_ibd1:
            ibd_start_connected_to_ibd2, ibd_end_connected_to_ibd2 = has_connected_ibd(prev_row, curr_row, next_row,
                                                                                       "IBD2")
            # can count start co
            if not seg_starts_chromosome and not ibd_start_connected_to_ibd2:
                if count_non_ambi and curr_row['overlaps_anywhere']:
                    sig_crossovers_p1.append(starting_crossover)
                elif count_ambiguous_co:  # and curr row overlaps with p2 relative
                    sig_crossovers_p2.append(starting_crossover)

            # can count end co
            if not seg_ends_chromosome and not ibd_end_connected_to_ibd2:
                if count_non_ambi and curr_row['overlaps_anywhere']:
                    sig_crossovers_p1.append(ending_crossover)
                elif count_ambiguous_co:  # and curr row overlaps with p2 relative
                    sig_crossovers_p2.append(ending_crossover)

            # count the gap
            if count_non_ambi and curr_row['overlaps_anywhere']:
                sig_gaps_p1.append(seg_gap)
            elif count_ambiguous_co:  # and curr row overlaps with p2 relative
                sig_gaps_p2.append(seg_gap)

        elif curr_row['IBD_type'] == 'IBD2' and count_ibd2:
            ibd_start_connected_to_ibd1, ibd_end_connected_to_ibd1 = has_connected_ibd(prev_row, curr_row, next_row,
                                                                                       "IBD1")
            sig_gaps_p1.append(seg_gap)
            sig_gaps_p2.append(seg_gap)

            # if the start of ibd2 is a crossover, then it is ia significant crossover for both parents
            if count_doubles and not ibd_start_connected_to_ibd1 and not seg_starts_chromosome:
                sig_crossovers_p1.append(starting_crossover)
                sig_crossovers_p2.append(starting_crossover)

            # if the end of the ibd2 is not connect to ibd1, then both are significant crossovers
            if count_doubles and not ibd_end_connected_to_ibd1 and not seg_ends_chromosome:
                sig_crossovers_p1.append(ending_crossover)
                sig_crossovers_p2.append(ending_crossover)

            # if the IBD2 is a continuation of ibd1, then there was a crossover for the ibd2 to start.
            # If the previous row overlaps/the connected ibd1 is shared with the relative, then it is known that the crossover would belong to p2, otherwise p1

            # checking to see if is valid to consider the start of the ibd2 segment connected to ibd1
            if not seg_starts_chromosome and ibd_start_connected_to_ibd1 and prev_row is not None:

                if count_ambiguous_co and not prev_row['overlaps_anywhere']:  # and prev row overlaps with p2 relative
                    sig_crossovers_p1.append(starting_crossover)

                if prev_row['overlaps_anywhere']:  # and prev row does NOT overlap with p2 relative
                    sig_crossovers_p2.append(starting_crossover)

            # checking to see if is valid to consider the end of the ibd2 segment
            if not seg_ends_chromosome and ibd_end_connected_to_ibd1 and next_row is not None:

                if count_ambiguous_co and not next_row['overlaps_anywhere']:  # and next row overlaps with p2 relative
                    sig_crossovers_p1.append(ending_crossover)

                if next_row['overlaps_anywhere']:  # and next row overlaps doesn't overlap with p2 relative
                    sig_crossovers_p2.append(ending_crossover)

    # ibd1 and ibd2 segments that are adjacent and can be considered 1 gap
    sig_gaps_p1 = combine_adjacent_gaps(sig_gaps_p1)
    sig_gaps_p2 = combine_adjacent_gaps(sig_gaps_p2)

    print("Lens before", len(sig_crossovers_p1), len(sig_crossovers_p2), len(sig_gaps_p1), len(sig_gaps_p2))

    if remove_dups:
        sig_crossovers_p1 = remove_dict_dups(sig_crossovers_p1)
        sig_crossovers_p2 = remove_dict_dups(sig_crossovers_p2)
        sig_gaps_p1 = remove_dict_dups(sig_gaps_p1)
        sig_gaps_p2 = remove_dict_dups(sig_gaps_p2)

    print("Lens after", len(sig_crossovers_p1), len(sig_crossovers_p2), len(sig_gaps_p1), len(sig_gaps_p2))

    return sig_crossovers_p1, sig_gaps_p1, sig_crossovers_p2, sig_gaps_p2


def overlaps(ibd_seg_1, ibd_seg_2):
    return ibd_seg_1['start'] <= ibd_seg_2['end'] and ibd_seg_1['end'] >= ibd_seg_2['start']


def add_overlaps_cols(seg_df, sib1, sib2, other_relatives: list, overlaps_col_name='overlaps_anywhere'):
    shared_sibling_ibd_segs = seg_df[
        seg_df['sample_1'].isin({sib1, sib2}) & seg_df['sample_1'].isin({sib1, sib2})].copy()

    shared_sibling_ibd_segs.loc[:, overlaps_col_name] = False

    for other_relative in other_relatives:
        sibs_and_other_shared_ibd = seg_df[
            (
                    ((seg_df['sample_1'] == sib1) & (seg_df['sample_2'] == other_relative)) |
                    ((seg_df['sample_1'] == other_relative) & (seg_df['sample_2'] == sib1)) |
                    ((seg_df['sample_1'] == sib2) & (seg_df['sample_2'] == other_relative)) |
                    ((seg_df['sample_1'] == other_relative) & (seg_df['sample_2'] == sib2))
            )]

        for chromosome in range(1, 23):
            sibs_ibd_at_chromosome = shared_sibling_ibd_segs[shared_sibling_ibd_segs['chromosome'] == chromosome]
            sib_other_ibd_at_chromosome = sibs_and_other_shared_ibd[
                sibs_and_other_shared_ibd['chromosome'] == chromosome]

            for idx, sibs_ibd in sibs_ibd_at_chromosome.iterrows():
                for _, sib_and_other_ibd in sib_other_ibd_at_chromosome.iterrows():
                    if sibs_ibd['end'] < sib_and_other_ibd['start']:
                        break

                    if overlaps(sibs_ibd, sib_and_other_ibd):
                        shared_sibling_ibd_segs.loc[idx, overlaps_col_name] = True
                        break

    return shared_sibling_ibd_segs


def determine_sibs_and_other_ids(run_id, seg_df, run):
    # TODO unhard code this section -- modify so it is not assumed the first two relatives are siblings
    run_id = run_id + "_"
    filtered = seg_df[(seg_df['sample_1'].str.contains(run_id) | seg_df['sample_2'].str.contains(run_id))]
    relative_ids = list(set(filtered[['sample_1', 'sample_2']].values.flatten()))
    relative_ids = sorted(relative_ids)
    # print(relative_ids[0], relative_ids[1], relative_ids[2:])
    # return relative_ids[0], relative_ids[1], relative_ids[2:]
    if run == 1:
        return relative_ids[1], relative_ids[0], [relative_ids[2]]
    elif run == 'sibs_comp':
        print("sib comps")
        return relative_ids[1], relative_ids[2], [relative_ids[0]]
    else:
        return relative_ids[2], relative_ids[0], [relative_ids[1]]


def write_output(results, output_file):
    results.to_csv(output_file, sep='\t', index=False, header=True)


def main(
        input_file,
        output_file,
        map_file,
        window,
        count_ambiguous_co,
        count_doubles,
        run_ids,
        count_ibd1,
        count_ibd2,
        count_non_ambi,
        remove_dups,
        count_p1,
        count_p2
):
    simmap = read_simmap(map_file)
    seg_df = read_in_seg_file_as_df(input_file,
                                    column_names=COLUMN_NAMES,
                                    columns_to_drop=COLUMNS_TO_DROP,
                                    sort_order=SORT_ORDER,
                                    column_type_casting=column_type_casting)

    pedigree_ids = seg_df['sample_1'].str.split("_").str[0].unique()

    results = pd.DataFrame(columns=["pedigree_id", "sib1", "sib2", "LOD_all", "LOD_co", "LOD_gap"])

    for pedigree_id in pedigree_ids:
        sib1, sib2, other_ids = determine_sibs_and_other_ids(pedigree_id, seg_df, 3)

        LOD_sum = 0

        for run in run_ids:
            sib1, sib2, other_ids = determine_sibs_and_other_ids(pedigree_id, seg_df, run)

            sibs_ibd = add_overlaps_cols(seg_df, sib1, sib2, other_ids)  # adds column of T/F overlaps with cousin

            sig_crossovers_p1, sig_gaps_p1, sig_crossovers_p2, sig_gaps_p2 = get_significant_crossovers_and_gaps(
                df=sibs_ibd,
                simmap=simmap,
                count_ambiguous_co=count_ambiguous_co,
                count_doubles=count_doubles,
                count_ibd1=count_ibd1,
                count_ibd2=count_ibd2,
                count_non_ambi=count_non_ambi,
                remove_dups=remove_dups)

            LOD_all, co_LOD, gap_LOD = calc_probability_using_both_parents(sig_crossovers_p1, sig_gaps_p1,
                                                                           sig_crossovers_p2,
                                                                           sig_gaps_p2, simmap, window, count_p1,
                                                                           count_p2)

            LOD_sum += LOD_all

        new_row = pd.DataFrame([{
            "sib1": sib1,
            "sib2": sib2,
            "other": other_ids,
            "LOD_all": LOD_all,
            "LOD_co": co_LOD,
            "LOD_gap": gap_LOD
        }])

        results = pd.concat([results, new_row], ignore_index=True)

        print(f"PedID: {pedigree_id}: {LOD_sum}")

    write_output(results, output_file)

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
