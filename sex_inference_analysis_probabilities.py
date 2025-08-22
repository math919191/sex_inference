from CREST_sex_inference import pos_search, logp_k_crossovers, logp_k_crossovers_for_length, log_product, logp_gaps, \
    log_poisson


def get_co_window(chromo, co_position, window_size, simmap):
    gmap = simmap[str(chromo)]
    gmap_ends = (gmap['pos'][0], gmap['pos'][-1])
    # don't go beyond the usable ends
    return max(co_position - (window_size / 2), gmap_ends[0]), min(co_position + (window_size / 2), gmap_ends[1])


def get_mf_lengths(chromo, start, end, simmap):
    lengths = {"male": 0, "female": 0}

    gmap = simmap[str(chromo)]

    _, start_pos, start_cM_male, start_cM_female = pos_search(start, gmap)
    _, end_pos, end_cM_male, end_cM_female = pos_search(end, gmap)

    lengths['male'] = (end_cM_male - start_cM_male) / 100
    lengths['female'] = (end_cM_female - start_cM_female) / 100

    return lengths


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
    return sig_gap['start'], sig_gap['end']

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

def calc_LOD(sig_gaps_df, sig_co_df, simmap, window_size):
    list_logp_female_co, list_logp_male_co, female_lengths_co, male_lengths_co = calc_significant_crossover_probability(sig_co_df, simmap, window_size)
    list_logp_female_gaps, list_logp_male_gaps, female_lengths_gaps, male_lengths_gaps = calc_significant_gap_probability(sig_gaps_df, simmap, window_size)

    female_prob = log_product(list_logp_female_co + list_logp_female_gaps)
    male_prob = log_product(list_logp_male_co + list_logp_male_gaps)

    LOD = female_prob - male_prob
    return LOD

def calc_probability_using_both_parents(sig_crossovers_p1, sig_gaps_p1, sig_crossovers_p2, sig_gaps_p2, simmap, window, count_p1, count_p2):
    co_prob_f_p1, co_prob_m_p1, female_lengths_co_p1, male_lengths_co_p1 = calc_significant_crossover_probability(sig_crossovers_p1, simmap, window)
    gap_prob_f_p1, gap_prob_m_p1, female_lengths_gap_p1, male_lengths_gap_p1 = calc_significant_gap_probability(sig_gaps_p1, simmap, window)

    co_prob_f_p2, co_prob_m_p2, female_lengths_co_p2, male_lengths_co_p2 = calc_significant_crossover_probability(sig_crossovers_p2, simmap, window)
    gap_prob_f_p2, gap_prob_m_p2, female_lengths_gap_p2, male_lengths_gap_p2 = calc_significant_gap_probability(sig_gaps_p2, simmap, window)


    lengths_p1f = female_lengths_co_p1 + male_lengths_co_p2
    lengths_p1m = male_lengths_co_p1 + female_lengths_co_p2

    LOD, co_LOD, gap_LOD = 0,0,0
    if count_p1 and count_p2:
        co_LOD = sum(co_prob_f_p1 + co_prob_m_p2) - sum(co_prob_m_p1 + co_prob_f_p2)
        gap_LOD = sum(gap_prob_f_p1 + gap_prob_m_p2) - sum(gap_prob_m_p1 + gap_prob_f_p2)

        # log product for p2 being female and p1 being male
        logp_p1m_p2f = log_product(co_prob_m_p1 + gap_prob_m_p1 + co_prob_f_p2 + gap_prob_f_p2)

        # log product for p1 being female and p2 being male
        logp_p1f_p2m = log_product(co_prob_f_p1 + gap_prob_f_p1 + co_prob_m_p2 + gap_prob_m_p2)

        LOD = logp_p1f_p2m - logp_p1m_p2f

    elif count_p1 and not count_p2:
        co_LOD = log_product(co_prob_f_p1) - log_product(co_prob_m_p1)
        gap_LOD = log_product(gap_prob_f_p1) - log_product(gap_prob_m_p1)

        LOD = co_LOD + gap_LOD

    elif count_p2 and not count_p1:
        co_LOD = log_product(co_prob_f_p2) - log_product(co_prob_m_p2)
        gap_LOD = log_product(gap_prob_f_p2) - log_product(gap_prob_m_p2)

        LOD = co_LOD + gap_LOD


    return LOD, co_LOD, gap_LOD
