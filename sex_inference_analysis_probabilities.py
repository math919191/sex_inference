from CREST_sex_inference import pos_search, logp_k_crossovers, logp_k_crossovers_for_length, log_product, logp_gaps, \
    log_poisson


def get_co_window(chromo, co, window_size, simmap):
    gmap = simmap[str(chromo)]
    gmap_ends = (gmap['pos'][0], gmap['pos'][-1])
    # don't go beyond the usable ends
    return max(co - window_size, gmap_ends[0]), min(co + window_size, gmap_ends[1])


def get_mf_lengths(chromo, start, end, simmap):
    lengths = {"male": 0, "female": 0}

    gmap = simmap[str(chromo)]

    start_chrom, start_pos, start_cM_male, start_cM_female = pos_search(start, gmap)
    end_chrom, end_pos, end_cM_male, end_cM_female = pos_search(end, gmap)

    lengths['male'] = (end_cM_male - start_cM_male) / 100
    lengths['female'] = (end_cM_female - start_cM_female) / 100

    return lengths


def calc_significant_crossover_probability(significant_crossovers, simmap, window_size):
    list_logp_female = []
    list_logp_male = []

    female_lengths = []
    male_lengths = []

    for sig_co in significant_crossovers:
        start, end = get_co_window(sig_co['chromosome'], sig_co['position'], window_size, simmap=simmap)

        lengths = get_mf_lengths(sig_co['chromosome'], start, end, simmap)

        list_logp_female.append(log_poisson(1, lengths['female']))
        list_logp_male.append(log_poisson(1, lengths['male']))

        female_lengths.append(lengths['female'])
        male_lengths.append(lengths['male'])

    return list_logp_female, list_logp_male, female_lengths, male_lengths

def get_gap_windows(sig_gap, window_size):
    return sig_gap['start'] + window_size, sig_gap['end'] - window_size


def calc_significant_gap_probability(significant_gaps, simmap, window_size):
    list_logp_female = []
    list_logp_male = []

    female_lengths = []
    male_lengths = []

    for sig_gap in significant_gaps:
        adjusted_start, adjusted_end = get_gap_windows(sig_gap, window_size)

        if adjusted_start > adjusted_end:
            print("The start is greater than the end of the segment")
            continue

        lengths = get_mf_lengths(sig_gap['chromosome'], adjusted_start, adjusted_end, simmap)

        list_logp_female.append(log_poisson(0, lengths['female']))
        list_logp_male.append(log_poisson(0, lengths['male']))

        female_lengths.append(lengths['female'])
        male_lengths.append(lengths['male'])

    return list_logp_female, list_logp_male, female_lengths, male_lengths


def calc_probability_mf(significant_crossovers, significant_gaps, simmap, window):
    co_list_f, co_list_m, female_lengths, male_lengths = calc_significant_crossover_probability(significant_crossovers, simmap, window)
    gap_list_f, gap_list_m, female_lengths_gaps, male_lengths_gaps = calc_significant_gap_probability(significant_gaps, simmap, window)

    logp_female = log_product(co_list_f + gap_list_f)
    logp_male = log_product(co_list_m + gap_list_m)

    LOD_all = logp_female - logp_male
    LOD_co = log_product(co_list_f) - log_product(co_list_m)
    LOD_gaps = log_product(gap_list_f) - log_product(gap_list_m)


    return LOD_all, LOD_co, LOD_gaps, sum(female_lengths), sum(male_lengths)

def calc_probability_using_both_parents(sig_crossovers_p1, sig_gaps_p1, sig_crossovers_p2, sig_gaps_p2, simmap, window):
    co_prob_f_p1, co_prob_m_p1, female_lengths_co_p1, male_lengths_co_p1 = calc_significant_crossover_probability(sig_crossovers_p1, simmap, window)
    gap_prob_f_p1, gap_prob_m_p1, female_lengths_gap_p1, male_lengths_gap_p1 = calc_significant_gap_probability(sig_gaps_p1, simmap, window)

    co_prob_f_p2, co_prob_m_p2, female_lengths_co_p2, male_lengths_co_p2 = calc_significant_crossover_probability(sig_crossovers_p2, simmap, window)
    gap_prob_f_p2, gap_prob_m_p2, female_lengths_gap_p2, male_lengths_gap_p2 = calc_significant_gap_probability(sig_gaps_p2, simmap, window)

    # log product for p1 being female and p2 being male
    logp_p1f_p2m = log_product(co_prob_f_p1 + gap_prob_f_p1 + co_prob_m_p2 + gap_prob_m_p2)

    # log product for p2 being female and p1 being male
    logp_p1m_p2f = log_product(co_prob_m_p1 + gap_prob_m_p1 + co_prob_f_p2 + gap_prob_f_p2)

    LOD = logp_p1f_p2m - logp_p1m_p2f

    lengths_p1f = female_lengths_co_p1 + male_lengths_co_p2
    lengths_p1m = male_lengths_co_p1 + female_lengths_co_p2
    return LOD, lengths_p1f, lengths_p1m
