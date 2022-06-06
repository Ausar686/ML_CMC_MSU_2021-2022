def is_palindrome(x):
    first_deg = 1
    last_deg = 1
    while x / first_deg >= 10:
        first_deg *= 10
    ans = 1
    while ans and first_deg > last_deg:
        first = x // first_deg % 10
        last = x // last_deg % 10
        if first != last:
            ans = 0
        first_deg //= 10
        last_deg *= 10
    if ans:
        return "YES"
    else:
        return "NO"
