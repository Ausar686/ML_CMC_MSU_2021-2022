def process(x):
    return sorted(set([a * a for b in x for a in b]))[::-1]
