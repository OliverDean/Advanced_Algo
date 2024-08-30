import numpy as np

def count_primes(N: int) -> int:

    if N < 2:
        return 0
    
    sieve = np.ones(N + 1, dtype=bool)
    sieve[:2] = False 

    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            sieve[i*i:N+1:i] = False
    
    return np.sum(sieve)

if __name__ == "__main__":

    N = int(input().strip())
    
    result = count_primes(N)

    print(result)
