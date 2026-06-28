# obs-01: structured log production (clay sys/log). Level-gated fan-out to a sink
# registry; default stderr sink. Instrumentation hooks (counters/trace points)
# are a later obs step.
hyle_add_module(obs SOURCES log.cc)
