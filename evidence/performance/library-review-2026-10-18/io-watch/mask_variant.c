#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/inotify.h>
int inotify_add_watch(int fd, const char *path, unsigned mask) {
  typedef int (*Original)(int, const char *, unsigned);
  Original original = (Original)dlsym(RTLD_NEXT, "inotify_add_watch");
  const unsigned supplied = mask;
  if (getenv("HYLE_WATCH_DROP_CLOSE_WRITE")) mask &= ~IN_CLOSE_WRITE;
  fprintf(stderr, "diagnostic inotify_add_watch input=%u output=%u\n", supplied, mask);
  return original(fd, path, mask);
}
