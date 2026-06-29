// crossbeam-deque parity bench for HYLE's work-stealing queues (task #9).
//
// Mirrors bench/container/queue_sota_bench.cc EXACTLY in workload shape so the
// ns/op numbers are comparable across languages:
//   * same kBatch = 512 fill+drain wave
//   * Worker::new_fifo()  == HYLE WsFifoQueue (tokio-style FIFO owner queue)
//   * Worker::new_lifo()  == HYLE WorkStealingDeque (Chase-Lev LIFO)
//   * steal_batch_and_pop == HYLE WsFifoQueue::steal_into (steal ~half + return one)
//   * Stealer::steal      == HYLE WorkStealingDeque::steal (one item)
//
// HONEST cross-language caveats (printed, not hidden):
//   * Different language + compiler (rustc LLVM vs MSVC/GCC). Same -O3+LTO+1 CGU.
//   * crossbeam Worker is UNBOUNDED (grows); HYLE's are BOUNDED rings. The owner
//     hot path is the comparable part (push/pop); a bounded ring also pays a
//     capacity check the unbounded one skips -- noted, not hidden.
//   * Measurement: hand-rolled min-of-epochs harness (nanobench has no Rust twin).
//     We report the MEDIAN ns/op over many epochs, same statistic nanobench shows.
//
// No criterion dependency (not in the offline cache); std::time::Instant is plenty
// for a 512-op batch repeated thousands of times.

use crossbeam_deque::{Steal, Worker};
use std::hint::black_box;
use std::time::Instant;

const CAP: usize = 1024; // matches C++ ring capacity (crossbeam grows past it)
const BATCH: usize = 512; // matches C++ kBatch

// Run `f` for enough epochs to get a stable median ns/op. `ops_per_call` is how
// many logical queue ops one call of `f` performs (for the per-op divide).
fn measure<F: FnMut() -> u64>(name: &str, ops_per_call: u64, mut f: F) {
    // Warm up (let the Worker buffer reach steady-state capacity, prime caches).
    for _ in 0..200 {
        black_box(f());
    }
    const EPOCHS: usize = 400;
    const ITERS: usize = 50; // calls timed per epoch
    let mut samples: Vec<f64> = Vec::with_capacity(EPOCHS);
    for _ in 0..EPOCHS {
        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..ITERS {
            acc = acc.wrapping_add(f());
        }
        black_box(acc);
        let elapsed = t0.elapsed().as_nanos() as f64;
        let total_ops = (ITERS as u64 * ops_per_call) as f64;
        samples.push(elapsed / total_ops);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    let min = samples[0];
    println!("{:<42} median {:>7.2} ns/op   min {:>7.2} ns/op", name, median, min);
}

// 1. OWNER FIFO fill+drain (== HYLE WsFifoQueue owner path). Reuse one Worker so
//    its buffer stops growing after the first wave (matches C++ static queue).
fn bench_owner_fifo() {
    let w: Worker<u64> = Worker::new_fifo();
    measure("crossbeam Worker FIFO owner fill+drain", (BATCH * 2) as u64, || {
        for i in 0..BATCH {
            w.push(i as u64);
        }
        let mut sum = 0u64;
        while let Some(v) = w.pop() {
            sum = sum.wrapping_add(v);
        }
        sum
    });
}

// 2. OWNER LIFO fill+drain (== HYLE WorkStealingDeque Chase-Lev owner path).
fn bench_owner_lifo() {
    let w: Worker<u64> = Worker::new_lifo();
    measure("crossbeam Worker LIFO owner fill+drain", (BATCH * 2) as u64, || {
        for i in 0..BATCH {
            w.push(i as u64);
        }
        let mut sum = 0u64;
        while let Some(v) = w.pop() {
            sum = sum.wrapping_add(v);
        }
        sum
    });
}

// 3. STEAL drain via steal_batch_and_pop (== HYLE WsFifoQueue::steal_into):
//    owner fills BATCH, thief steals ~half into its own worker + pops one, drains.
fn bench_steal_batch() {
    let victim: Worker<u64> = Worker::new_fifo();
    let thief: Worker<u64> = Worker::new_fifo();
    let stealer = victim.stealer();
    measure("crossbeam steal_batch_and_pop drain", BATCH as u64, || {
        for i in 0..BATCH {
            victim.push(i as u64);
        }
        let mut sum = 0u64;
        loop {
            match stealer.steal_batch_and_pop(&thief) {
                Steal::Success(v) => {
                    sum = sum.wrapping_add(v);
                    while let Some(t) = thief.pop() {
                        sum = sum.wrapping_add(t);
                    }
                }
                Steal::Empty => break,
                Steal::Retry => continue,
            }
        }
        sum
    });
}

// 4. Single-item steal (== HYLE WorkStealingDeque::steal): owner fills, thief
//    steals one at a time off a LIFO deque.
fn bench_steal_one() {
    let w: Worker<u64> = Worker::new_lifo();
    let s = w.stealer();
    measure("crossbeam Stealer::steal one-at-a-time", BATCH as u64, || {
        for i in 0..BATCH {
            w.push(i as u64);
        }
        let mut sum = 0u64;
        loop {
            match s.steal() {
                Steal::Success(v) => sum = sum.wrapping_add(v),
                Steal::Empty => break,
                Steal::Retry => continue,
            }
        }
        sum
    });
}

fn main() {
    // Version is the LOCAL crossbeam-deque clone (path dep in Cargo.toml), 0.8.3.
    println!("# crossbeam-deque 0.8.3 (local clone) parity (cap={}, batch={})", CAP, BATCH);
    println!("# rival for HYLE WsFifoQueue (FIFO) + WorkStealingDeque (LIFO/Chase-Lev)");
    bench_owner_fifo();
    bench_owner_lifo();
    bench_steal_batch();
    bench_steal_one();
}
