#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int check_file_existence(const char *serial) {
    int result;

    char adb_command[256];
    sprintf(adb_command, "adb -s %s shell '[ -f /data/local/tmp/sync_android ] && echo exists || echo not_exists'", serial);
    result = system(adb_command);
    if (result == 0) {
        printf("Command executed successfully. Check output for file existence.\n");
    } else {
        printf("Failed to execute adb shell command\n");
        return 1;
    }

    FILE* fp;
    char path[1035];
    sprintf(adb_command, "adb -s %s shell '[ -f /data/local/tmp/sync_android ] && echo exists || echo not_exists'", serial);
    fp = popen(adb_command, "r");
    if (fp == NULL) {
        printf("Failed to run command\n");
        exit(1);
    }

    if (fgets(path, sizeof(path) - 1, fp) != NULL) {
        if (strstr(path, "not_exists")) {
            printf("/data/local/tmp/sync_android does not exist, pushing it...\n");
            sprintf(adb_command, "adb -s %s push sync_android /data/local/tmp/sync_android", serial);
            result = system(adb_command);
            if (result != 0) {
                printf("Failed to push sync_android to the device. Check if compiled sync_android exists in rtux_ai first\n");
                pclose(fp);
                return 1;
            } else {
                printf("sync_android successfully pushed to the device.\n");
            }
        } else {
            printf("/data/local/tmp/sync_android exists.\n");
        }
    } else {
        printf("Error reading command output.\n");
        pclose(fp);
        return 1;
    }

    // Close the pipe
    pclose(fp);

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <serial>\n", argv[0]);
        return 1;
    }
    char *serial = argv[1];

    if (check_file_existence(serial) != 0) {
        return 1;
    }

    // Execute adb command to set date
    char adb_command[256];
    // sprintf(adb_command, "adb -s %s exec-out date -s @$(date +"%s.%N")", serial);
    sprintf(adb_command, "adb -s %s exec-out date -s @$(date +\"%%s.%%N\")", serial);
    printf("ADB command: %s\n", adb_command); // Debug
    FILE *pipe;
    pipe = popen(adb_command, "r");
    if (!pipe) {
        perror("popen");
        return 1;
    }
    pclose(pipe);

    // Execute shell commands to display times
    sprintf(adb_command, "date +\"%%T.%%N\"; adb -s %s shell date +\"%%T.%%N\"; date +\"%%T.%%N\"", serial);
    pipe = popen(adb_command, "r");
    if (!pipe) {
        perror("popen");
        return 1;
    }
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        printf("%s", buffer);
    }
    pclose(pipe);

    struct timespec nowtime;
    clock_gettime(CLOCK_REALTIME, &nowtime);

    FILE *fp;
    char path[1035];
    struct timespec target = {nowtime.tv_sec + 5, nowtime.tv_nsec};
    sprintf(adb_command, "adb -s %s shell ./data/local/tmp/sync_android %ld.%09ld & ", serial, target.tv_sec, target.tv_nsec);
    fp = popen(adb_command, "r");
    if (fp == NULL) {
        printf("Failed to run command\n" );
        exit(1);
    }
    struct timespec realtime;
    struct timespec boottime;
    do {
        clock_gettime(CLOCK_REALTIME, &realtime);
        clock_gettime(CLOCK_BOOTTIME, &boottime);
    } while (realtime.tv_sec < target.tv_sec || (realtime.tv_sec == target.tv_sec && realtime.tv_nsec < target.tv_nsec));

    while (fgets(path, sizeof(path)-1, fp) != NULL) {
        printf("%s", path);
    }

    printf("LINUX_Realtime:  %ld.%09ld\n", realtime.tv_sec, realtime.tv_nsec);
    printf("LINUX_Boottime: %ld.%09ld\n", boottime.tv_sec, boottime.tv_nsec);
    
    return 0;
}
