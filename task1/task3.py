def longestCommonPrefix(array):
    if len(array) == 0:
        return ''
    res = array[0]
    start = 0
    while start < len(res) and res[start].isspace():
        start += 1
    res = res[start:len(res)]
    if res == '':
        return ''
    for i in range (1, len(array)):
        string = array[i]
        start = 0
        while start < len(string) and string[start].isspace():
            start += 1
        string = string[start:len(string)]
        length = min(len(res), len(string))
        new_res = ''
        for j in range(length):
            if res[j] == string[j]:
                new_res += res[j]
            else:
                break
        res = new_res
        if res == '':
            break
    return res
