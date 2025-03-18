#include <stdio.h>
#include <time.h>

int main() {
    struct timespec tp;
    clock_gettime(CLOCK_BOOTTIME, &tp);
    printf("%ld.%09ld\n", tp.tv_sec, tp.tv_nsec);
    return 0;
}
