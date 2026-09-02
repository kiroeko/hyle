#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "hyle/io/io_context.h"
#include "hyle/io/net.h"
#include "hyle/io/net_multishot.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if defined(__linux__) || defined(__APPLE__)
#  include <sys/socket.h>
#  include <unistd.h>
#endif
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#endif

// io-04 integration test (shard-per-core): the co_await socket layer over real
// fds. Each connection is bound to an OWNER worker -- its fds registered on that
// worker's ring, its fibers pinned there via spawn_on -- so submit + reap +
// resume are single-threaded on one core (share-nothing, no ring lock). Runs on
// MULTIPLE workers with connections sharded across them.

using hyle::io::IoContext;
using hyle::io::MultishotRecvStream;
using hyle::io::RecvBuf;
using hyle::io::RecvOp;
using hyle::io::SendOp;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

// A connected pair of fds, bound to an owner worker. On IOCP each fd associates
// with exactly ONE completion port, so we register only on the owner's ring (a
// no-op on io_uring/epoll) -- and the connection's fibers are pinned to that same
// owner, so every op runs on the ring the fd is bound to.
struct SocketPair {
  std::intptr_t a = -1;
  std::intptr_t b = -1;
  bool ok = false;
};

#if defined(__linux__) || defined(__APPLE__)
SocketPair make_pair(IoContext& io, unsigned owner) {
  int sv[2];
  SocketPair sp;
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return sp;
  sp.a = sv[0];
  sp.b = sv[1];
  // register_handle is a no-op on io_uring/epoll but keeps the call site uniform.
  sp.ok = io.bind_fd(owner, sp.a) && io.bind_fd(owner, sp.b);
  return sp;
}
void close_pair(SocketPair& sp) {
  if (sp.a >= 0) ::close(static_cast<int>(sp.a));
  if (sp.b >= 0) ::close(static_cast<int>(sp.b));
}
#endif

#if defined(_WIN32)
SocketPair make_pair(IoContext& io, unsigned owner) {
  SocketPair sp;
  SOCKET ls = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  if (ls == INVALID_SOCKET) return sp;
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = 0;
  ::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  ::listen(ls, 1);
  int slen = sizeof(sa);
  ::getsockname(ls, reinterpret_cast<sockaddr*>(&sa), &slen);
  SOCKET c = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  ::connect(c, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  SOCKET s = ::accept(ls, nullptr, nullptr);
  ::closesocket(ls);
  if (c == INVALID_SOCKET || s == INVALID_SOCKET) return sp;
  sp.a = static_cast<std::intptr_t>(c);
  sp.b = static_cast<std::intptr_t>(s);
  // Bind BOTH ends to the owner worker's port only. IOCP allows a socket to
  // associate with exactly one port, and the connection's fibers are pinned to
  // this same owner, so both ends' ops complete on this one ring.
  sp.ok = io.bind_fd(owner, sp.a) && io.bind_fd(owner, sp.b);
  return sp;
}
void close_pair(SocketPair& sp) {
  if (sp.a != -1) ::closesocket(static_cast<SOCKET>(sp.a));
  if (sp.b != -1) ::closesocket(static_cast<SOCKET>(sp.b));
}
#endif

// Coroutines resolve the CURRENT worker's ring at each co_await. Because the
// connection's fibers are PINNED to their owner worker (spawn_on), the current
// worker is always the owner, so this is the ring the fd is bound to.
struct EchoState {
  IoContext* io;
  std::intptr_t send_fd;
  std::intptr_t recv_fd;
  char recv_buf[64] = {};
  std::int64_t sent = 0;
  std::int64_t received = 0;
};

Task<int> echo_coro(AllocatorRef, EchoState* st) {
  static const char kMsg[] = "co_await-socket-echo";
  const std::uint32_t len = sizeof(kMsg);
  st->sent = co_await SendOp{&st->io->proactor_for_current(), st->send_fd, kMsg, len};
  st->received =
      co_await RecvOp{&st->io->proactor_for_current(), st->recv_fd, st->recv_buf, len};
  co_return 0;
}

struct Drive {
  Task<int> task;
  std::atomic<bool> finished{false};
};
void drive_entry(void* p) {
  auto* d = static_cast<Drive*>(p);
  d->task.resume();
  d->finished.store(true, std::memory_order_release);
}

}  // namespace

// Single fiber, self send+recv, pinned to its owner worker on a multi-worker
// scheduler. The op submits + reaps on the owner's ring.
TEST(IoNet, CoAwaitSendRecvRoundTrip) {
  constexpr unsigned kWorkers = 4;
  constexpr unsigned kOwner = 1;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid()) << "every worker's proactor must come up";

  SocketPair sp = make_pair(io, kOwner);
  ASSERT_TRUE(sp.ok) << "failed to build a connected socket pair";

  EchoState st{};
  st.io = &io;
  st.send_fd = sp.a;
  st.recv_fd = sp.b;

  auto d = std::make_unique<Drive>();
  d->task = echo_coro(system_allocator(), &st);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());

  sched.spawn_on(kOwner, &f);  // pin to the fd's owner worker
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(d->finished.load());
  EXPECT_EQ(st.sent, static_cast<std::int64_t>(sizeof("co_await-socket-echo")));
  EXPECT_EQ(st.received, static_cast<std::int64_t>(sizeof("co_await-socket-echo")));
  EXPECT_STREQ(st.recv_buf, "co_await-socket-echo");

  close_pair(sp);
}

// ---- two-fiber echo, connection pinned to one core -------------------------
// A server fiber co_awaits recv then echoes back with send; a client fiber
// co_awaits send then recv of the echo. Both fibers are pinned to the SAME owner
// worker (the fd's core), so all four ops run share-nothing on one ring.
namespace {
struct EchoPair {
  IoContext* io;
  std::intptr_t server_fd;
  std::intptr_t client_fd;
  char server_buf[64] = {};
  char client_reply[64] = {};
  std::int64_t server_recv = 0;
  std::int64_t client_recv = 0;
};

Task<int> server_coro(AllocatorRef, EchoPair* c) {
  c->server_recv =
      co_await RecvOp{&c->io->proactor_for_current(), c->server_fd, c->server_buf, 64};
  if (c->server_recv > 0) {
    co_await SendOp{&c->io->proactor_for_current(), c->server_fd, c->server_buf,
                    static_cast<std::uint32_t>(c->server_recv)};
  }
  co_return 0;
}

Task<int> client_coro(AllocatorRef, EchoPair* c) {
  static const char kMsg[] = "two-fiber-echo";
  const std::uint32_t len = sizeof(kMsg);
  co_await SendOp{&c->io->proactor_for_current(), c->client_fd, kMsg, len};
  c->client_recv =
      co_await RecvOp{&c->io->proactor_for_current(), c->client_fd, c->client_reply, len};
  co_return 0;
}
}  // namespace

TEST(IoNet, TwoFiberEchoPinnedToCore) {
  constexpr unsigned kWorkers = 4;
  constexpr unsigned kOwner = 2;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());

  SocketPair sp = make_pair(io, kOwner);
  ASSERT_TRUE(sp.ok);

  EchoPair c{};
  c.io = &io;
  c.server_fd = sp.a;
  c.client_fd = sp.b;

  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = server_coro(system_allocator(), &c);
  dc->task = client_coro(system_allocator(), &c);
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());

  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(ds->finished.load());
  EXPECT_TRUE(dc->finished.load());
  EXPECT_EQ(c.server_recv, static_cast<std::int64_t>(sizeof("two-fiber-echo")));
  EXPECT_EQ(c.client_recv, static_cast<std::int64_t>(sizeof("two-fiber-echo")));
  EXPECT_STREQ(c.server_buf, "two-fiber-echo");
  EXPECT_STREQ(c.client_reply, "two-fiber-echo");

  close_pair(sp);
}

// Many concurrent connections SHARDED across cores: N socket pairs, each bound to
// a different owner worker (round-robin) with its fiber pair pinned there. This
// is the shard-per-core payoff -- connections partitioned across cores, each core
// driving its own share-nothing ring in parallel.
TEST(IoNet, ManyConnectionsShardedAcrossCores) {
  constexpr unsigned kWorkers = 4;
  constexpr int kPairs = 32;
  Scheduler sched(kWorkers, system_allocator(), 1024);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());

  std::vector<SocketPair> pairs;
  std::vector<std::unique_ptr<EchoPair>> states;
  std::vector<std::unique_ptr<Drive>> drives;
  std::vector<std::unique_ptr<Fiber>> fibers;
  pairs.reserve(kPairs);

  for (int i = 0; i < kPairs; ++i) {
    unsigned owner = static_cast<unsigned>(i) % kWorkers;  // shard across cores
    SocketPair sp = make_pair(io, owner);
    ASSERT_TRUE(sp.ok);
    pairs.push_back(sp);
    auto c = std::make_unique<EchoPair>();
    c->io = &io;
    c->server_fd = sp.a;
    c->client_fd = sp.b;
    auto ds = std::make_unique<Drive>();
    auto dc = std::make_unique<Drive>();
    ds->task = server_coro(system_allocator(), c.get());
    dc->task = client_coro(system_allocator(), c.get());
    auto fs = std::make_unique<Fiber>(kDefaultStackSize,
                                      &drive_entry, ds.get());
    auto fc = std::make_unique<Fiber>(kDefaultStackSize,
                                      &drive_entry, dc.get());
    sched.spawn_on(owner, fs.get());  // both fibers pinned to the fd's core
    sched.spawn_on(owner, fc.get());
    states.push_back(std::move(c));
    drives.push_back(std::move(ds));
    drives.push_back(std::move(dc));
    fibers.push_back(std::move(fs));
    fibers.push_back(std::move(fc));
  }

  sched.drain();
  sched.shutdown();

  for (auto& c : states) {
    EXPECT_EQ(c->client_recv, static_cast<std::int64_t>(sizeof("two-fiber-echo")));
    EXPECT_STREQ(c->client_reply, "two-fiber-echo");
  }
  for (auto& sp : pairs) close_pair(sp);
}

#if defined(__linux__)
// ---- full-duplex on ONE fd, forced-epoll (red-team L3 regression) -----------
// Two fibers concurrently in flight on the SAME fd: one blocked in recv(a), the
// other doing send(a). On the epoll fallback these arm the same fd for EPOLLIN and
// EPOLLOUT respectively. Before the fix, the single map<fd,op> `pending` had the
// second submit OVERWRITE the first -> the recv fiber was orphaned forever (drain
// hangs = ctest timeout). After the fix (pending_in + pending_out), both complete.
// Forced onto epoll (io_uring has no such issue) via HYLE_IO_FORCE_EPOLL.
namespace {
struct DuplexState {
  IoContext* io;
  std::intptr_t fd;      // the shared fd both fibers operate on
  char recv_buf[64] = {};
  std::int64_t recv_n = 0;
  std::int64_t send_n = 0;
};
Task<int> duplex_recv(AllocatorRef, DuplexState* s) {
  // Blocks until data arrives on `fd` (the peer end is written by the test).
  s->recv_n = co_await RecvOp{&s->io->proactor_for_current(), s->fd, s->recv_buf, 64};
  co_return 0;
}
Task<int> duplex_send(AllocatorRef, DuplexState* s) {
  static const char kMsg[] = "duplex-write";
  s->send_n = co_await SendOp{&s->io->proactor_for_current(), s->fd, kMsg, sizeof(kMsg)};
  co_return 0;
}
}  // namespace

TEST(IoNet, FullDuplexSameFdEpoll) {
  ::setenv("HYLE_IO_FORCE_EPOLL", "1", 1);  // exercise the fallback path
  constexpr unsigned kWorkers = 2;
  constexpr unsigned kOwner = 0;
  {
    Scheduler sched(kWorkers, system_allocator(), 256);
    IoContext io(sched, kWorkers);
    ASSERT_TRUE(io.valid());
    SocketPair sp = make_pair(io, kOwner);
    ASSERT_TRUE(sp.ok);

    DuplexState st{};
    st.io = &io;
    st.fd = sp.a;  // BOTH fibers use fd a: recv(a) blocks, send(a) writes to a->b

    auto dr = std::make_unique<Drive>();
    auto dw = std::make_unique<Drive>();
    dr->task = duplex_recv(system_allocator(), &st);
    dw->task = duplex_send(system_allocator(), &st);
    Fiber fr(kDefaultStackSize, &drive_entry, dr.get());
    Fiber fw(kDefaultStackSize, &drive_entry, dw.get());
    sched.spawn_on(kOwner, &fr);  // recv(a): arms fd a for EPOLLIN, blocks
    sched.spawn_on(kOwner, &fw);  // send(a): arms fd a for EPOLLOUT, concurrent
    // Feed recv(a) by writing to the peer end b (plain blocking write is fine here).
    static const char kFeed[] = "feed-recv";
    ASSERT_EQ(::write(static_cast<int>(sp.b), kFeed, sizeof(kFeed)),
              static_cast<ssize_t>(sizeof(kFeed)));

    sched.drain();  // pre-fix: HANGS here (recv fiber orphaned by send's overwrite)
    sched.shutdown();

    EXPECT_TRUE(dr->finished.load()) << "recv fiber orphaned (L3)";
    EXPECT_TRUE(dw->finished.load()) << "send fiber orphaned (L3)";
    EXPECT_GT(st.recv_n, 0) << "recv on the shared fd never completed";
    EXPECT_GT(st.send_n, 0) << "send on the shared fd never completed";
    close_pair(sp);
  }
  ::unsetenv("HYLE_IO_FORCE_EPOLL");
}

// io-07b multishot recv awaitable: `co_await stream.next()` over a provided-buffer
// ring. ONE arm delivers a chunk per datagram (kBuffer + kMore), each RecvBuf
// returning its ring buffer on destruction; peer close = terminal (more()==false).
// Proves the awaitable layer (heap slot + refcount) surfaces the zero-copy multishot
// path the proactor implemented, without a per-recv submit or a caller buffer.
namespace {
struct MultishotState {
  IoContext* io;
  std::intptr_t recv_fd;
  int bgid;
  std::atomic<int> chunks{0};
  char joined[256] = {};
  std::uint32_t joined_len = 0;
  std::atomic<bool> saw_terminal{false};
};
Task<int> multishot_recv_coro(AllocatorRef alloc, MultishotState* st) {
  auto uring = st->io->proactor_for_current().io_uring_extension();
  auto stream = uring ? MultishotRecvStream{uring, st->recv_fd, st->bgid, alloc}
                      : MultishotRecvStream{};
  for (;;) {
    RecvBuf b = co_await stream.next();
    if (!b.more() && b.size() == 0) {  // terminal chunk
      st->saw_terminal.store(true, std::memory_order_release);
      break;
    }
    // Append the chunk bytes (data lives in the ring buffer; copy out before b's dtor
    // returns it). Datagram boundaries => one chunk per message.
    if (b.size() > 0 && st->joined_len + b.size() < sizeof(st->joined)) {
      std::memcpy(st->joined + st->joined_len, b.data(), b.size());
      st->joined_len += b.size();
    }
    st->chunks.fetch_add(1, std::memory_order_relaxed);
    if (!b.more()) {  // a data chunk that is ALSO terminal
      st->saw_terminal.store(true, std::memory_order_release);
      break;
    }
  }
  co_return 0;
}
}  // namespace

TEST(IoNet, MultishotRecvAwaitable) {
  constexpr unsigned kWorkers = 2;
  constexpr unsigned kOwner = 0;
  constexpr int kBgid = 11;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());
  // Register the provided-buffer ring on the OWNER worker's proactor (io_uring only).
  auto uring = io.proactor(kOwner).io_uring_extension();
  if (!uring ||
      !uring.register_buffer_ring(kBgid, /*entries=*/8, /*buffer_size=*/128))
    GTEST_SKIP() << "provided buffer ring unsupported (non-uring backend/old kernel)";

  // STREAM pair: peer close delivers a terminal (result 0, kMore clear) completion --
  // the real end-of-stream the awaitable targets (TCP echo/proxy). STREAM may coalesce
  // the 3 writes into fewer/one chunk(s); we assert on the JOINED bytes + terminal, not
  // chunk count. Bind the recv end to the owner worker's ring.
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
  ASSERT_TRUE(io.bind_fd(kOwner, sv[0]));
  // Pre-send 3 segments + close the peer so the stream sees data then a clean EOF
  // terminal (avoids a blocking reap waiting on data that never comes).
  const char* msgs[] = {"aa", "bb", "cc"};
  for (const char* m : msgs) ASSERT_GT(::write(sv[1], m, std::strlen(m)), 0);
  ::close(sv[1]);  // peer close -> EOF terminal completion after the data

  MultishotState st{};
  st.io = &io;
  st.recv_fd = sv[0];
  st.bgid = kBgid;
  auto d = std::make_unique<Drive>();
  d->task = multishot_recv_coro(system_allocator(), &st);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn_on(kOwner, &f);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(d->finished.load()) << "multishot fiber never finished (stream hung)";
  EXPECT_GE(st.chunks.load(), 1) << "one arm should deliver the data (>=1 chunk)";
  EXPECT_TRUE(st.saw_terminal.load()) << "peer close must surface a terminal chunk";
  st.joined[st.joined_len] = '\0';
  EXPECT_STREQ(st.joined, "aabbcc") << "all bytes received in order (STREAM coalesces)";
  ::close(sv[0]);
}
#endif  // __linux__

namespace {
struct MultishotAllocState {
  AllocatorRef backing = system_allocator();
  std::atomic<int> live{0};
  static void* alloc(void* p, std::size_t n, std::size_t a) noexcept {
    auto* s = static_cast<MultishotAllocState*>(p);
    void* out = s->backing.allocate(n, a);
    if (out != nullptr) s->live.fetch_add(1, std::memory_order_relaxed);
    return out;
  }
  static void free(void* p, void* q, std::size_t n, std::size_t a) noexcept {
    if (q == nullptr) return;
    auto* s = static_cast<MultishotAllocState*>(p);
    s->backing.deallocate(q, n, a);
    s->live.fetch_sub(1, std::memory_order_relaxed);
  }
  static void* realloc(void* p, void* q, std::size_t old_n, std::size_t new_n,
                       std::size_t a) noexcept {
    return static_cast<MultishotAllocState*>(p)->backing.reallocate(
        q, old_n, new_n, a);
  }
  AllocatorRef handle() noexcept { return AllocatorRef(&alloc, &free, &realloc, this); }
};
}  // namespace

// The terminal CQE owns and spends the multishot's kernel ref. Dropping a stream
// before that CQE must neither wake its stale Waker nor leak the heap slot.
TEST(IoNet, MultishotTerminalBalancesLifetimeAfterEarlyDrop) {
  MultishotAllocState state;
  AllocatorRef a = state.handle();
  hyle::io::Proactor p;  // uninitialized is enough: early-drop cancel returns false

  auto make_slot = [&]() {
    auto* s = static_cast<hyle::io::MultishotSlot*>(
        a.allocate(sizeof(hyle::io::MultishotSlot),
                   alignof(hyle::io::MultishotSlot)));
    EXPECT_NE(s, nullptr);
    if (s == nullptr) return s;
    new (s) hyle::io::MultishotSlot{};
    s->alloc = a;
    s->armed = true;
    return s;
  };

  // Normal terminal: completion spends kernel ref, stream dtor spends fiber ref.
  {
    auto* s = make_slot();
    ASSERT_NE(s, nullptr);
    hyle::io::MultishotRecvStream stream{p.io_uring_extension(), -1, 1, a};
    stream.slot_ = s;
    hyle::io::complete_multishot(
        hyle::io::Completion{hyle::io::to_multishot_token(s), 0, 0, 0});
    EXPECT_EQ(state.live.load(), 1);
  }
  EXPECT_EQ(state.live.load(), 0);

  // Early drop: stream marks the slot abandoned and drops only its own ref.
  // The late terminal CQE must free the remaining kernel ref without a wake.
  auto* late = make_slot();
  ASSERT_NE(late, nullptr);
  {
    hyle::io::MultishotRecvStream stream{p.io_uring_extension(), -1, 1, a};
    stream.slot_ = late;
  }
  EXPECT_EQ(state.live.load(), 1);
  hyle::io::complete_multishot(
      hyle::io::Completion{hyle::io::to_multishot_token(late), -125, 0, 0});
  EXPECT_EQ(state.live.load(), 0);
}
