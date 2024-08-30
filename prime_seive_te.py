def count_primes(N: int) -> int:

    if N < 2:
        return 0
    
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False

    p = 2
    while p * p <= N:
        if is_prime[p]:
            for i in range(p * p, N + 1, p):
                is_prime[i] = False
        p += 1

    return sum(is_prime)

if __name__ == "__main__":

    N = int(input().strip())
    
    result = count_primes(N)
    
    print(result)
