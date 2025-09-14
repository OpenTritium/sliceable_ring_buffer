use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sliceable_ring_buffer::SliceRingBuffer;
use std::{collections::VecDeque, hint::black_box, time::Duration};

// A range of sizes to test various memory footprints.
const SIZES: &[usize] = &[128, 16 * 1024, 512 * 1024];

// ====================================================================================
// Group 1: `push_back` Workloads - Isolating Reallocation Cost
// ====================================================================================
fn bench_push_back_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_back_workloads");
    group.measurement_time(Duration::from_secs(20));

    for &size in SIZES {
        group.throughput(Throughput::Bytes((size * std::mem::size_of::<usize>()) as u64));

        // --- Scenario 1: Growing from empty (Measures push_back + REALLOCATION cost) ---
        group.bench_with_input(BenchmarkId::new("SliceRingBuffer_Grow", size), &size, |b, &s| {
            b.iter(|| {
                let mut srb = SliceRingBuffer::new();
                for i in 0..s {
                    srb.push_back(black_box(i));
                }
            })
        });
        group.bench_with_input(BenchmarkId::new("VecDeque_Grow", size), &size, |b, &s| {
            b.iter(|| {
                let mut vdq = VecDeque::new();
                for i in 0..s {
                    vdq.push_back(black_box(i));
                }
            })
        });

        // --- Scenario 2: Filling a pre-allocated buffer (Measures push_back cost ONLY) ---
        group.bench_with_input(BenchmarkId::new("SliceRingBuffer_NoRealloc", size), &size, |b, &s| {
            b.iter_batched(
                || SliceRingBuffer::with_capacity(s),
                |mut srb| {
                    for i in 0..s {
                        srb.push_back(black_box(i));
                    }
                },
                criterion::BatchSize::SmallInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("VecDeque_NoRealloc", size), &size, |b, &s| {
            b.iter_batched(
                || VecDeque::with_capacity(s),
                |mut vdq| {
                    for i in 0..s {
                        vdq.push_back(black_box(i));
                    }
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

// ====================================================================================
// Group 2: `pop_front` Workloads - Pure Read Performance
// ====================================================================================
fn bench_pop_front_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop_front_workloads");
    group.measurement_time(Duration::from_secs(15));

    for &size in SIZES {
        group.throughput(Throughput::Bytes((size * std::mem::size_of::<usize>()) as u64));

        group.bench_with_input(BenchmarkId::new("SliceRingBuffer", size), &size, |b, &s| {
            b.iter_batched(
                || {
                    let mut srb = SliceRingBuffer::with_capacity(s);
                    for i in 0..s {
                        srb.push_back(i);
                    }
                    srb
                },
                |mut srb| {
                    while srb.pop_front().is_some() {}
                },
                criterion::BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", size), &size, |b, &s| {
            b.iter_batched(
                || {
                    let mut vdq = VecDeque::with_capacity(s);
                    for i in 0..s {
                        vdq.push_back(i);
                    }
                    vdq
                },
                |mut vdq| {
                    while vdq.pop_front().is_some() {}
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

// ====================================================================================
// Group 3: Churn - Simulates a steady-state queue
// ====================================================================================
fn bench_churn_steady_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("churn_steady_state");
    group.measurement_time(Duration::from_secs(15));

    for &size in SIZES {
        // We do one push and one pop per iteration.
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(BenchmarkId::new("SliceRingBuffer", size), &size, |b, &s| {
            b.iter_batched(
                || {
                    let mut srb = SliceRingBuffer::with_capacity(s);
                    for i in 0..(s / 2) {
                        srb.push_back(i);
                    }
                    srb
                },
                |mut srb| {
                    srb.push_back(black_box(0));
                    black_box(srb.pop_front());
                },
                criterion::BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", size), &size, |b, &s| {
            b.iter_batched(
                || {
                    let mut vdq = VecDeque::with_capacity(s);
                    for i in 0..(s / 2) {
                        vdq.push_back(i);
                    }
                    vdq
                },
                |mut vdq| {
                    vdq.push_back(black_box(0));
                    black_box(vdq.pop_front());
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

// ====================================================================================
// Group 4: Iteration - Contiguous vs. Wrapped Layout
// ====================================================================================
fn bench_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration");
    group.measurement_time(Duration::from_secs(20));

    for &size in SIZES {
        group.throughput(Throughput::Bytes((size * std::mem::size_of::<usize>()) as u64));

        // --- Scenario 1: Contiguous layout (best case for VecDeque iterator) ---
        let mut srb_cont = SliceRingBuffer::with_capacity(size);
        let mut vdq_cont = VecDeque::with_capacity(size);
        for i in 0..size {
            srb_cont.push_back(i);
            vdq_cont.push_back(i);
        }

        group.bench_with_input(BenchmarkId::new("SliceRingBuffer_Contiguous", size), &srb_cont, |b, srb| {
            b.iter(|| {
                for item in srb.iter() {
                    black_box(item);
                }
            })
        });
        group.bench_with_input(BenchmarkId::new("VecDeque_Contiguous", size), &vdq_cont, |b, vdq| {
            b.iter(|| {
                for item in vdq.iter() {
                    black_box(item);
                }
            })
        });

        // --- Scenario 2: Wrapped layout (worst case for VecDeque iterator) ---
        let mut srb_wrap = SliceRingBuffer::with_capacity(size);
        let mut vdq_wrap = VecDeque::with_capacity(size);
        for i in 0..size {
            srb_wrap.push_back(i);
            vdq_wrap.push_back(i);
        }
        for _ in 0..(size / 2) {
            srb_wrap.pop_front();
            vdq_wrap.pop_front();
        }
        for i in 0..(size / 2) {
            srb_wrap.push_back(i);
            vdq_wrap.push_back(i);
        }

        group.bench_with_input(BenchmarkId::new("SliceRingBuffer_Wrapped", size), &srb_wrap, |b, srb| {
            b.iter(|| {
                for item in srb.iter() {
                    black_box(item);
                }
            })
        });
        group.bench_with_input(BenchmarkId::new("VecDeque_Wrapped", size), &vdq_wrap, |b, vdq| {
            b.iter(|| {
                for item in vdq.iter() {
                    black_box(item);
                }
            })
        });
    }
    group.finish();
}

// ====================================================================================
// Group 5: `make_contiguous` - The Core Feature Benchmark
// ====================================================================================
fn bench_make_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_contiguous_wrapped");
    group.measurement_time(Duration::from_secs(20));

    for &size in SIZES {
        group.throughput(Throughput::Bytes((size * std::mem::size_of::<usize>()) as u64));

        // Setup a wrapped-around buffer, which is the only state where this matters.
        let mut srb = SliceRingBuffer::with_capacity(size);
        for i in 0..size {
            srb.push_back(i);
        }
        for _ in 0..(size / 2) {
            srb.pop_front();
        }
        for i in 0..(size / 2) {
            srb.push_back(i);
        }

        // For SliceRingBuffer, this is a near zero-cost, O(1) operation.
        group.bench_with_input(BenchmarkId::new("SliceRingBuffer_as_slice", size), &srb, |b, srb| {
            b.iter(|| {
                black_box(srb.as_slice());
            })
        });

        // For VecDeque, this is an O(n) operation involving a memory copy.
        group.bench_with_input(BenchmarkId::new("VecDeque_make_contiguous", size), &size, |b, &s| {
            b.iter_batched(
                || {
                    let mut vdq = VecDeque::with_capacity(s);
                    for i in 0..s {
                        vdq.push_back(i);
                    }
                    for _ in 0..(s / 2) {
                        vdq.pop_front();
                    }
                    for i in 0..(s / 2) {
                        vdq.push_back(i);
                    }
                    vdq
                },
                |mut vdq| {
                    black_box(vdq.make_contiguous());
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

// Register all benchmark groups
criterion_group!(
    benches,
    bench_push_back_workloads,
    bench_pop_front_workloads,
    bench_churn_steady_state,
    bench_iteration,
    bench_make_contiguous
);
criterion_main!(benches);
