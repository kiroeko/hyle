# Reader-safe retirement is its own leaf factor. The hot guard stays inline;
# queueing, grace periods and the process-default owner are compiled once.
hyle_add_module(mem_reclaim SOURCES reclamation.cc)
