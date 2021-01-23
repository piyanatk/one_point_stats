def get_mwa_eor_freq(ch):
    if ch > 704:
        print('Maximum frequency channel is 704')
    else:
        return 138.915 + 0.08 * ch
