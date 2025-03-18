//This file is based on android-touch-record-replay by Cartucho.
//Original repository: https://github.com/Cartucho/android-touch-record-replay
//Modifications made by Jaeheon Lee on 2024/12/10.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/ioctl.h>
//#include <linux/input.h> // this does not compile
#include <unistd.h>
#include <errno.h>
#include <signal.h>

// from <linux/input.h>

typedef uint32_t        __u32;
typedef uint16_t        __u16;
typedef __signed__ int  __s32;

struct input_event {
    struct timeval time;
    __u16 type;
    __u16 code;
    __u32 value;
};

#define MICROSEC 1000000

#define EVIOCGVERSION		_IOR('E', 0x01, int)			/* get driver version */
#define EVIOCGID		_IOR('E', 0x02, struct input_id)	/* get device ID */
#define EVIOCGKEYCODE		_IOR('E', 0x04, int[2])			/* get keycode */
#define EVIOCSKEYCODE		_IOW('E', 0x04, int[2])			/* set keycode */

#define EVIOCGNAME(len)		_IOC(_IOC_READ, 'E', 0x06, len)		/* get device name */
#define EVIOCGPHYS(len)		_IOC(_IOC_READ, 'E', 0x07, len)		/* get physical location */
#define EVIOCGUNIQ(len)		_IOC(_IOC_READ, 'E', 0x08, len)		/* get unique identifier */

#define EVIOCGKEY(len)		_IOC(_IOC_READ, 'E', 0x18, len)		/* get global keystate */
#define EVIOCGLED(len)		_IOC(_IOC_READ, 'E', 0x19, len)		/* get all LEDs */
#define EVIOCGSND(len)		_IOC(_IOC_READ, 'E', 0x1a, len)		/* get all sounds status */
#define EVIOCGSW(len)		_IOC(_IOC_READ, 'E', 0x1b, len)		/* get all switch states */

#define EVIOCGBIT(ev,len)	_IOC(_IOC_READ, 'E', 0x20 + ev, len)	/* get event bits */
#define EVIOCGABS(abs)		_IOR('E', 0x40 + abs, struct input_absinfo)		/* get abs value/limits */
#define EVIOCSABS(abs)		_IOW('E', 0xc0 + abs, struct input_absinfo)		/* set abs value/limits */

#define EVIOCSFF		_IOC(_IOC_WRITE, 'E', 0x80, sizeof(struct ff_effect))	/* send a force effect to a force feedback device */
#define EVIOCRMFF		_IOW('E', 0x81, int)			/* Erase a force effect */
#define EVIOCGEFFECTS		_IOR('E', 0x84, int)			/* Report number of effects playable at the same time */

#define EVIOCGRAB		_IOW('E', 0x90, int)			/* Grab/Release device */

// end <linux/input.h>

#define MICROSEC 1000000
#define EV_SYN 0x00
#define EV_KEY 0x01
#define KEY_MAX 0x1FF
#define test_bit(bit, array) (array[(bit) / 8] & (1 << ((bit) % 8)))



int fd; // file descriptor for input device


void remove_specific_chars(char* str, char c1, char c2) {
    char *pr = str, *pw = str;
    while (*pr) {
        *pw = *pr++;
        pw += (*pw != c1 && *pw != c2);
    }
    *pw = '\0';
}

void ensure_clean_device_state() {
    unsigned char key_states[KEY_MAX / 8 + 1] = {0};

    if (ioctl(fd, EVIOCGKEY(sizeof(key_states)), key_states) == -1) {
        perror("Failed to get key states");
        return;
    }

    for (int code = 0; code <= KEY_MAX; code++) {
        if (test_bit(code, key_states)) {
            fprintf(stderr, "Device in down state for code %d. Sending release.\n", code);
            
            struct input_event release_event = {0};
            release_event.type = EV_KEY;
            release_event.code = code;
            release_event.value = 0; // Release

            if (write(fd, &release_event, sizeof(release_event)) < sizeof(release_event)) {
                perror("Failed to send release event");
            }

            struct input_event syn_event = {0};
            syn_event.type = EV_SYN;
            syn_event.code = 0;
            syn_event.value = 0;

            if (write(fd, &syn_event, sizeof(syn_event)) < sizeof(syn_event)) {
                perror("Failed to send sync event");
            }
        }
    }
}

int key_down[KEY_MAX + 1] = {0};
void handle_sigusr1(int sig) {
    for (int code = 0; code <= KEY_MAX; code++) {
        if (key_down[code]) {
            struct input_event release_event = {0};
            release_event.type = EV_KEY;
            release_event.code = code;
            release_event.value = 0;

            if (write(fd, &release_event, sizeof(release_event)) < sizeof(release_event)) {
                fprintf(stderr, "Failed to send release event for code %d: %s\n", code, strerror(errno));
            }

            struct input_event syn_event = {0};
            syn_event.type = EV_SYN;
            syn_event.code = 0;
            syn_event.value = 0;

            if (write(fd, &syn_event, sizeof(syn_event)) < sizeof(syn_event)) {
                fprintf(stderr, "Failed to send sync event: %s\n", strerror(errno));
            }

            key_down[code] = 0;
        }
    }

    fprintf(stderr, "SIGNAL RECEIVED: All keys released successfully (EXITING)\n");
    close(fd);
    _exit(0);
}


int main(int argc, char *argv[]) {
    int ret, version;
    struct input_event event;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_device> <input_events>\n", argv[0]);
        return 1;
    }

    fd = open(argv[1], O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Could not open %s: %s\n", argv[1], strerror(errno));
        return 1;
    }

    signal(SIGUSR1, handle_sigusr1);

    ensure_clean_device_state();
    fprintf(stderr, "Device state is clean.\n");

    FILE *fd_in = fopen(argv[2], "r");
    if (fd_in == NULL) {
        fprintf(stderr, "Could not open input file: %s\n", argv[2]);
        return 1;
    }

    char line[128];
    unsigned int sleep_time;
    double timestamp_previous = 0.0;
    double timestamp_now;
    char type[32], code[32], value[32];

    while (fgets(line, sizeof(line), fd_in) != NULL) {
        remove_specific_chars(line, '[', ']');
        sscanf(line, "%lf %s %s %s", &timestamp_now, type, code, value);

        sleep_time = (unsigned int)((timestamp_now - timestamp_previous) * MICROSEC);
        usleep(sleep_time);

        memset(&event, 0, sizeof(event));
        event.type = (int)strtol(type, NULL, 16);
        event.code = (int)strtol(code, NULL, 16);
        event.value = (uint32_t)strtoll(value, NULL, 16);
        ret = write(fd, &event, sizeof(event));
        if (ret < sizeof(event)) {
            fprintf(stderr, "Write event failed: %s\n", strerror(errno));
            return -1;
        }

        if (event.type == EV_KEY) {
            if (event.value == 1) { // Key down
                key_down[event.code] = 1;
            } else if (event.value == 0) { // Key up
                key_down[event.code] = 0;
            }
        }

        timestamp_previous = timestamp_now;

        memset(line, 0, sizeof(line));
        memset(type, 0, sizeof(type));
        memset(code, 0, sizeof(code));
        memset(value, 0, sizeof(value));
    }
    ensure_clean_device_state();
    fclose(fd_in);
    close(fd);

    return 0;
}