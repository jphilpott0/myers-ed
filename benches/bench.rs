#![feature(const_index, const_trait_impl)]

use criterion::*;

use myers_ed::prelude::*;

// 512 char DNA strings. Second has a edit distance of 2 from the first.
const BRCA2_C50: &[u8] = &include_bytes!("brca2_c50.txt")[..512];
const BRCA2_C50_MUT: &[u8] = &include_bytes!("brca2_c50_mut.txt")[..512];

pub fn bench_myers_ed_single_avx512_12x12(c: &mut Criterion) {
    c.bench_function("bench_myers_ed_single_avx512_12x12", |b| {
        b.iter(|| black_box(myers_ed_single_avx512(b"GATCAATGACTG", b"GATCAATAACTG")))
    });
}

pub fn bench_myers_ed_single_avx512_with_peq_12x12(c: &mut Criterion) {
    let peq = SingleWordPeq::from_bytes(b"GATCAATGACTG");

    c.bench_function("bench_myers_ed_single_avx512_12x12", |b| {
        b.iter(|| black_box(myers_ed_single_avx512_with_peq(&peq, b"GATCAATAACTG")))
    });
}

pub fn bench_myers_ed_single_avx512_512x512(c: &mut Criterion) {
    c.bench_function("bench_myers_ed_single_avx512_512x512", |b| {
        b.iter(|| black_box(myers_ed_single_avx512(BRCA2_C50, BRCA2_C50_MUT)))
    });
}

pub fn bench_myers_ed_single_avx512_with_peq_512x512(c: &mut Criterion) {
    let peq = SingleWordPeq::from_bytes(BRCA2_C50);

    c.bench_function("bench_myers_ed_single_avx512_512x512", |b| {
        b.iter(|| black_box(myers_ed_single_avx512_with_peq(&peq, BRCA2_C50_MUT)))
    });
}

criterion_group!(
    benches,
    bench_myers_ed_single_avx512_12x12,
    bench_myers_ed_single_avx512_with_peq_12x12,
    bench_myers_ed_single_avx512_512x512,
    bench_myers_ed_single_avx512_with_peq_512x512
);
criterion_main!(benches);
