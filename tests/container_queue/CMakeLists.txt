hyle_add_header_self_contained_test(container_queue
  HEADER_DIR container
  HEADERS
    bounded_queue.h
    byte_ring_buffer.h
    sharded_queue.h
    spsc_queue.h
    unbounded_queue.h
    work_stealing_deque.h
    ws_fifo_queue.h)
hyle_add_test(container_queue SOURCES
  bounded_queue_test.cc
  byte_ring_buffer_test.cc
  sharded_queue_test.cc
  spsc_queue_test.cc
  unbounded_queue_test.cc
  work_stealing_deque_test.cc
  ws_fifo_queue_test.cc)
