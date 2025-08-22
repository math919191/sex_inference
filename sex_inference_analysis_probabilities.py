from CREST_sex_inference import pos_search, log_product, log_poisson


def get_mf_lengths(chromo, start, end, simmap):
    lengths = {"male": 0, "female": 0}

    gmap = simmap[str(chromo)]

    _, start_pos, start_cM_male, start_cM_female = pos_search(start, gmap)
    _, end_pos, end_cM_male, end_cM_female = pos_search(end, gmap)

    lengths['male'] = (end_cM_male - start_cM_male) / 100
    lengths['female'] = (end_cM_female - start_cM_female) / 100

    return lengths

def get_co_window(chromo, co_position, window_size, simmap):
    gmap = simmap[str(chromo)]
    gmap_ends = (gmap['pos'][0], gmap['pos'][-1])
    # don't go beyond the usable ends
    return max(co_position - (window_size / 2), gmap_ends[0]), min(co_position + (window_size / 2), gmap_ends[1])


def calc_significant_crossover_probability(significant_crossovers, simmap, window_size):
    list_logp_female = []
    list_logp_male = []

    for sig_co in significant_crossovers:
        window_start, window_end = get_co_window(sig_co['chromosome'], sig_co['start'], window_size, simmap=simmap)

        lengths = get_mf_lengths(sig_co['chromosome'], window_start, window_end, simmap)

        list_logp_female.append(log_poisson(1, lengths['female']))
        list_logp_male.append(log_poisson(1, lengths['male']))

    return list_logp_female, list_logp_male

def get_gap_windows(sig_gap, window_size):
    return sig_gap['start'] + (window_size / 2), sig_gap['end'] - (window_size / 2)

def calc_significant_gap_probability(significant_gaps, simmap, window_size):
    list_logp_female = []
    list_logp_male = []

    for sig_gap in significant_gaps:
        adjusted_start, adjusted_end = get_gap_windows(sig_gap, window_size)

        if adjusted_start > adjusted_end:
            print("The start is greater than the end of the segment")
            continue

        lengths = get_mf_lengths(sig_gap['chromosome'], adjusted_start, adjusted_end, simmap)

        list_logp_female.append(log_poisson(0, lengths['female']))
        list_logp_male.append(log_poisson(0, lengths['male']))

    return list_logp_female, list_logp_male


def calc_probability_mf(significant_crossovers, significant_gaps, simmap, window_size):
    co_list_logp_female, co_list_logp_male = calc_significant_crossover_probability(significant_crossovers, simmap, window_size)
    gap_list_logp_female, gap_list_logp_male = calc_significant_gap_probability(significant_gaps, simmap, window_size)

    logp_female = log_product(co_list_logp_female + gap_list_logp_female + gap_list_logp_male)
    logp_male = log_product(co_list_logp_male + gap_list_logp_male + gap_list_logp_female)

    LOD_all = logp_female - logp_male

    return LOD_all
