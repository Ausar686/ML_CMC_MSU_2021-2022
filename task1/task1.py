def hello(x = None):
    if x is None or x == '':
        return("Hello!")
    return("Hello, " + str(x) + "!")
