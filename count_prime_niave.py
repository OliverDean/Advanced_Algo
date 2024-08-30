import time

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

def count_primes(N: int) -> int:
    """
    Counts the number of prime numbers less than or equal to N.

    @param N: The upper limit for checking primes.
    @return: The number of primes less than or equal to N.
    """
    prime_count = 0
    for i in range(2, N + 1):
        if is_prime(i):
            prime_count += 1
    return prime_count

if __name__ == "__main__":
    # Set the value of N for testing
    N = 2000

    print(f"Testing prime counting up to {N}:\n")

    # Test the prime counting function
    start_time = time.time()
    prime_count = count_primes(N)
    elapsed_time = time.time() - start_time

    print(f"Number of primes less than or equal to {N}: {prime_count}")
    print(f"Time taken: {elapsed_time:.6f} seconds")