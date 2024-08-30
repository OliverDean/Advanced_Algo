#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int count_primes(int N) {
    if (N < 2) return 0;

    bool *is_prime = (bool *)malloc((N + 1) * sizeof(bool));
    for (int i = 0; i <= N; i++) is_prime[i] = true;
    is_prime[0] = is_prime[1] = false;

    for (int p = 2; p * p <= N; p++) {
        if (is_prime[p]) {
            for (int i = p * p; i <= N; i += p) {
                is_prime[i] = false;
            }
        }
    }

    int prime_count = 0;
    for (int i = 2; i <= N; i++) {
        if (is_prime[i]) prime_count++;
    }

    free(is_prime);
    return prime_count;
}

int main() {
    int N;
    scanf("%d", &N);
    printf("%d\n", count_primes(N));
    return 0;
}
