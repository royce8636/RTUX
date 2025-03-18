#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <target_time>\n", argv[0]);
        return 1;
    }

    char* dot = strchr(argv[1], '.');
    time_t target_sec = atol(argv[1]); // Seconds part
    long target_nsec = 0; // Nanoseconds part
    if (dot != NULL) {
        target_nsec = atol(dot + 1);
        int len = strlen(dot + 1);
        for (int i = 0; i < 9 - len; i++) {
            target_nsec *= 10;
        }
    }
    // printf("Target:    %ld.%09ld\n", target_sec, target_nsec);
    struct timespec realtime;
    struct timespec boottime;

    do {
        clock_gettime(CLOCK_REALTIME, &realtime);
        clock_gettime(CLOCK_BOOTTIME, &boottime);
    } while (realtime.tv_sec < target_sec || (realtime.tv_sec == target_sec && realtime.tv_nsec < target_nsec));

    printf("ANDROID_Realtime:  %ld.%09ld\n", realtime.tv_sec, realtime.tv_nsec);
    printf("ANDROID_Boottime: %ld.%09ld\n", boottime.tv_sec, boottime.tv_nsec);
    
    return 0;
}
