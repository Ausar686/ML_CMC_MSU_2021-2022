def primes():
    def is_prime(n):
        for i in range(2, n):
            if n % i == 0:
                return False
        return True
    cur = 2
    while(1):
        yield cur
        cur += 1
        while not is_prime(cur):
            cur += 1

        
