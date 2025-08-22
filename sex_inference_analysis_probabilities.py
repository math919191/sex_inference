from CREST_sex_inference import pos_search, log_product, log_poisson


def get_mf_lengths(chromo, start, end, simmap):
    lengths = {"male": 0, "female": 0}

    gmap = simmap[str(chromo)]

    _, start_pos, start_cM_male, start_cM_female = pos_search(start, gmap)
    _, end_pos, end_cM_male, end_cM_female = pos_search(end, gmap)

    lengths['male'] = (end_cM_male - start_cM_male) / 100
    lengths['female'] = (end_cM_female - start_cM_female) / 100

    return lengths

def get_co_window(co_position, chromo, window_size, simmap):
    gmap = simmap[str(chromo)]
    gmap_ends = (gmap['pos'][0], gmap['pos'][-1])
    # don't go beyond the usable ends
    return max(co_position - (window_size / 2), gmap_ends[0]), min(co_position + (window_size / 2), gmap_ends[1])

def get_gap_windows(gap, chromo, window_size, simmap):
    gmap = simmap[str(chromo)]
    gmap_ends = (gmap['pos'][0], gmap['pos'][-1])
    # don't go beyond the usable ends
    return max(gap['start'] + (window_size / 2), gmap_ends[0]), min(gap['end'] - (window_size / 2), gmap_ends[1])

def calc_probability(events, simmap, window_size, is_co):
    list_logp_female = []
    list_logp_male = []

    for event in events:
        if is_co:
            adjusted_start, adjusted_end = get_co_window(event['start'], event["chromosome"], window_size, simmap)
            num_events = 1
        else:
            adjusted_start, adjusted_end = get_gap_windows(event, event["chromosome"], window_size, simmap)
            num_events = 0

        if adjusted_start > adjusted_end:
            print("The start is greater than the end of the segment")
            continue

        lengths = get_mf_lengths(event['chromosome'], adjusted_start, adjusted_end, simmap)

        list_logp_female.append(log_poisson(num_events, lengths['female']))
        list_logp_male.append(log_poisson(num_events, lengths['male']))

    return list_logp_female, list_logp_male

def calc_probability_mf(significant_crossovers, significant_gaps, simmap, window_size):
    co_list_logp_female, co_list_logp_male = calc_probability(significant_crossovers, simmap, window_size, is_co=True)
    gap_list_logp_female, gap_list_logp_male = calc_probability(significant_gaps, simmap, window_size, is_co=False)

    logp_female = log_product(co_list_logp_female + gap_list_logp_female + gap_list_logp_female)
    logp_male = log_product(co_list_logp_male + gap_list_logp_male + gap_list_logp_male)

    LOD_all = logp_female - logp_male

    return LOD_all
