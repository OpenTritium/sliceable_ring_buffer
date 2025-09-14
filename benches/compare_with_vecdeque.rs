use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sliceable_ring_buffer::SliceRingBuffer;
use std::{collections::VecDeque, hint::black_box, time::Duration};

// 1KB, 32KB, 8MB, 16MB
const SIZES: [usize; 4] = [0x80, 0x1000, 0x100_000, 0x200_000];

fn bench_push_back(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_back");
    group.measurement_time(Duration::from_secs(20));
    for &size in SIZES.iter() {
        group.throughput(Throughput::Bytes(size as u64 * std::mem::size_of::<usize>() as u64));
        group.bench_with_input(BenchmarkId::new("SliceRingBuffer", size), &size, |b, &size| {
            b.iter(|| {
                let mut srb = SliceRingBuffer::with_capacity(size);
                for i in 0..size {
                    srb.push_back(black_box(i));
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", size), &size, |b, &size| {
            b.iter(|| {
                let mut vdq = VecDeque::with_capacity(size);
                for i in 0..size {
                    vdq.push_back(black_box(i));
                }
            })
        });
    }
    group.finish();
}

fn bench_as_contiguous_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("as_contiguous_slice");
    group.measurement_time(Duration::from_secs(20));
    for &size in SIZES.iter() {
        group.throughput(Throughput::Bytes(size as u64 * std::mem::size_of::<usize>() as u64));
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

        group.bench_with_input(BenchmarkId::new("SliceRingBuffer", size), &srb, |b, srb| {
            b.iter(|| {
                black_box(srb.as_slice());
            })
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut vdq = VecDeque::with_capacity(size);
                    for i in 0..size {
                        vdq.push_back(i);
                    }
                    for _ in 0..(size / 2) {
                        vdq.pop_front();
                    }
                    for i in 0..(size / 2) {
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

fn bench_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration_wrapped");
    group.measurement_time(Duration::from_secs(20));
    for &size in SIZES.iter() {
        group.throughput(Throughput::Bytes(size as u64 * std::mem::size_of::<usize>() as u64));
        let mut srb = SliceRingBuffer::with_capacity(size);
        let mut vdq = VecDeque::with_capacity(size);
        for i in 0..size {
            srb.push_back(i);
            vdq.push_back(i);
        }
        for _ in 0..(size / 2) {
            srb.pop_front();
            vdq.pop_front();
        }
        for i in 0..(size / 2) {
            srb.push_back(i);
            vdq.push_back(i);
        }

        group.bench_with_input(BenchmarkId::new("SliceRingBuffer", size), &srb, |b, srb| {
            b.iter(|| {
                for item in srb.iter() {
                    black_box(item);
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("VecDeque", size), &vdq, |b, vdq| {
            b.iter(|| {
                for item in vdq.iter() {
                    black_box(item);
                }
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_push_back, bench_as_contiguous_slice, bench_iteration);
criterion_main!(benches);
