# timer_queue: caller-owned timer nodes and allocation-free ordering only.
# Callback execution, waiting and cross-thread routing stay in Sched/IO.
hyle_add_module(timer_queue)
