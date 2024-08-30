#include <stdio.h>
#include <math.h>

int count_squares(long long L, long long R) {
    long long start = sqrt(L);
    long long end = sqrt(R);
    if (start * start < L) start++;
    if (start > end) return 0;
    return (int)(end - start + 1);
}

int count_cubes(long long L, long long R) {
    long long start = cbrt(L);
    long long end = cbrt(R);

    // Ensure we correctly include boundary cubes
    if (start * start * start < L) start++;
    if (end * end * end > R) end--;

    if (start > end) return 0;
    return (int)(end - start + 1);
}

int count_squbes(long long L, long long R) {
    int count = 0;
    for (long long k = 1; k * k * k * k * k * k <= R; k++) {
        long long k6 = k * k * k * k * k * k;
        if (k6 >= L && k6 <= R) {
            count++;
        }
    }
    return count;
}

int main() {
    long long L, R;
    scanf("%lld %lld", &L, &R);

    int squares = count_squares(L, R);
    int cubes = count_cubes(L, R);
    int squbes = count_squbes(L, R);

    printf("%d %d %d\n", squares, cubes, squbes);
    return 0;
}
