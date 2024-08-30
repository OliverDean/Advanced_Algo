def count_primes(N: int) -> int:
    """
    Counts the number of prime numbers less than or equal to N using the Sieve of Eratosthenes.

    @param N: The upper limit for checking primes.
    @return: The number of primes less than or equal to N.
    """
    if N < 2:
        return 0
    
    is_prime = [True] * (N + 1)
    is_prime[0], is_prime[1] = False, False
    
    p = 2
    while p * p <= N:
        if is_prime[p]:
            for i in range(p * p, N + 1, p):
                is_prime[i] = False
        p += 1
    
    prime_count = sum(is_prime)
    
    return prime_count

if __name__ == "__main__":

    N = int(input().strip())
    
    result = count_primes(N)

    print(result)
