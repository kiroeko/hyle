// Native Crossbeam work-stealing API traces beside Hyle's C++ queue benchmark.
// Both fill 512 values and check every output, but they are not identical
// implementations or harnesses: Rust/LLVM differs from the measured C++ build;
// Worker buffers can grow and shrink whereas Hyle's rings are fixed; Crossbeam's
// default steal batch is capped at 32 whereas Hyle steals half the remaining items.
// These are single-thread API traces, not concurrent owner/thief throughput.

use crossbeam_deque::{Steal, Worker};
use std::hint::black_box;
use std::time::Instant;

const BATCH: usize = 512;
const WARMUP: usize = 200;

// Queue/stealer construction precedes this function and destruction follows it.
// Warmup does not stop native Worker::pop from shrinking its buffer: a later fill
// may grow it again. Keep that native work in the callback and disclose it.
fn measure<F: FnMut() -> u64>(name: &str, ops_per_call: u64, mut f: F) {
    for _ in 0..WARMUP {
        black_box(f());
    }
    // A functional-only path: never enter Instant's timing path.
    if std::env::var_os("HYLE_BENCH_SMOKE").is_some() {
        println!("smoke PASS: {name}, {WARMUP} checked waves");
        return;
    }
    const EPOCHS: usize = 400;
    const ITERS: usize = 50;
    let mut samples: Vec<f64> = Vec::with_capacity(EPOCHS);
    let mut elapsed_ns: Vec<u128> = Vec::with_capacity(EPOCHS);
    for _ in 0..EPOCHS {
        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..ITERS {
            acc = acc.wrapping_add(f());
        }
        black_box(acc);
        let elapsed = t0.elapsed().as_nanos();
        let total_ops = (ITERS as u64 * ops_per_call) as f64;
        elapsed_ns.push(elapsed);
        samples.push(elapsed as f64 / total_ops);
    }
    // Preserve execution order. Sort only a copy to calculate summary statistics.
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = (sorted[EPOCHS / 2 - 1] + sorted[EPOCHS / 2]) / 2.0;
    println!(
        "{:<48} median {:>7.2} ns/unit   min {:>7.2} ns/unit",
        name, median, sorted[0]
    );
    for (epoch, ns_per_unit) in samples.iter().enumerate() {
        println!(
            "sample\t{name}\t{epoch}\t{ITERS}\t{ops_per_call}\t{}\t{ns_per_unit:.12}",
            elapsed_ns[epoch]
        );
    }
}

fn owner_wave(w: &Worker<u64>, lifo: bool) -> u64 {
    for i in 0..BATCH {
        w.push(i as u64); // infallible native API (may allocate/grow)
    }
    let mut sum = 0;
    for i in 0..BATCH {
        let expected = if lifo { BATCH - 1 - i } else { i } as u64;
        assert_eq!(w.pop(), Some(expected));
        sum += expected;
    }
    assert_eq!(w.pop(), None);
    sum
}

struct CheckedBatch {
    seen: [bool; BATCH],
    count: usize,
    sum: u64,
    xor_sum: u64,
}

impl CheckedBatch {
    fn new() -> Self {
        Self {
            seen: [false; BATCH],
            count: 0,
            sum: 0,
            xor_sum: 0,
        }
    }

    fn add(&mut self, value: u64) {
        assert!(value < BATCH as u64 && !self.seen[value as usize]);
        self.seen[value as usize] = true;
        self.count += 1;
        self.sum += value;
        self.xor_sum ^= value;
    }

    fn finish(self) -> u64 {
        assert_eq!(self.count, BATCH);
        assert_eq!(self.sum, (BATCH * (BATCH - 1) / 2) as u64);
        assert_eq!(self.xor_sum, 0);
        self.sum
    }
}

fn bench_owner_fifo() {
    let w = Worker::new_fifo();
    measure(
        "crossbeam FIFO owner fill+drain (scalar ops)",
        (BATCH * 2) as u64,
        || owner_wave(&w, false),
    );
}

fn bench_owner_lifo() {
    let w = Worker::new_lifo();
    measure(
        "crossbeam LIFO owner fill+drain (scalar ops)",
        (BATCH * 2) as u64,
        || owner_wave(&w, true),
    );
}

// Owner fill is timed too; denominator is 512 transferred items. The native
// default cap of 32 is retained. Direct-return order differs from Hyle's batch
// operation, so validate exact conservation without imposing a shared order.
fn bench_steal_batch() {
    let victim = Worker::new_fifo();
    let thief = Worker::new_fifo();
    let stealer = victim.stealer();
    measure(
        "crossbeam capped32 fill+steal+drain (items)",
        BATCH as u64,
        || {
            for i in 0..BATCH {
                victim.push(i as u64);
            }
            let mut checked = CheckedBatch::new();
            loop {
                match stealer.steal_batch_and_pop(&thief) {
                    Steal::Success(value) => {
                        checked.add(value);
                        while let Some(value) = thief.pop() {
                            checked.add(value);
                        }
                    }
                    Steal::Empty => break,
                    Steal::Retry => continue,
                }
            }
            assert_eq!(victim.pop(), None);
            assert_eq!(thief.pop(), None);
            checked.finish()
        },
    );
}

fn bench_steal_one() {
    let w = Worker::new_lifo();
    let stealer = w.stealer();
    measure(
        "crossbeam single-item fill+steal+drain (items)",
        BATCH as u64,
        || {
            for i in 0..BATCH {
                w.push(i as u64);
            }
            let mut count = 0;
            loop {
                match stealer.steal() {
                    Steal::Success(value) => {
                        assert!(count < BATCH);
                        assert_eq!(value, count as u64);
                        count += 1;
                    }
                    Steal::Empty => break,
                    Steal::Retry => continue,
                }
            }
            assert_eq!(count, BATCH);
            assert_eq!(w.pop(), None);
            count as u64
        },
    );
}

fn main() {
    println!(
        "# Native crossbeam-deque; record the resolved Cargo.lock/metadata and compiler identity."
    );
    println!("# Single thread, 512 items/wave; owner units=push+pop, steal units=items including owner fill.");
    println!("# Worker may grow/shrink; default batch cap32 differs from Hyle steal-half; no identical-work ranking claim.");
    println!("# sample columns: name, epoch, iterations, units/call, elapsed_ns, ns/unit (execution order)");
    let filter = std::env::var("HYLE_QUEUE_BENCH_FILTER").unwrap_or_default();
    assert!(filter.is_empty() || filter == "ws-owner" || filter == "ws-steal");
    if filter.is_empty() || filter == "ws-owner" {
        bench_owner_fifo();
        bench_owner_lifo();
    }
    if filter.is_empty() || filter == "ws-steal" {
        bench_steal_batch();
        bench_steal_one();
    }
}
