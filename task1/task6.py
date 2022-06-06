def check(s, filename):
    words = s.lower().split()
    dict_words = {}
    for word in words:
        if word in dict_words:
            dict_words[word] += 1
        else:
            dict_words[word] = 1
    words = sorted(list(set(words)))
    f = open(filename, "w")
    for word in words:
        f.write(word + ' ' + str(dict_words[word]) + '\n')
    f.close()
