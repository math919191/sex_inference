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

def get_gap_windows(gap_start, gap_end, chromo, window_size, simmap):
    gmap = simmap[str(chromo)]
    gmap_ends = (gmap['pos'][0], gmap['pos'][-1])
    # don't go beyond the usable ends
    return max(gap_start + (window_size / 2), gmap_ends[0]), min(gap_end - (window_size / 2), gmap_ends[1])

def calc_lod_probability(event, simmap, window_size):
    is_co = event['is_co']
    if is_co:
        adjusted_start, adjusted_end = get_co_window(event['start'], event["chromosome"], window_size, simmap)
        num_events = 1
    else:
        adjusted_start, adjusted_end = get_gap_windows(event['start'], event['end'], event["chromosome"], window_size, simmap)
        num_events = 0

    if adjusted_start > adjusted_end:
        print("The start is greater than the end of the segment")
        return 0

    lengths = get_mf_lengths(event['chromosome'], adjusted_start, adjusted_end, simmap)

    female_logp = log_poisson(num_events, lengths['female'])
    male_logp = log_poisson(num_events, lengths['male'])

    return (female_logp - male_logp)
