use blake3::tree::BlockHasher;
use blake3_tree::*;
use criterion::*;
use rand::{thread_rng, Rng};

fn bench_tree(c: &mut Criterion) {
    let mut g = c.benchmark_group("Tree");
    g.sample_size(50);

    for i in 1..16 {
        let size = i * 128;

        let mut tree_builder = blake3::tree::HashTreeBuilder::new();
        (0..size).for_each(|i| tree_builder.update(&block_data(i)));
        let output = tree_builder.finalize();

        // It's always one block.
        g.throughput(Throughput::Bytes(256 * 1024));

        g.bench_with_input(
            BenchmarkId::new("gen-proof-beginning", size),
            &size,
            |b, size| {
                let mut rng = thread_rng();
                b.iter(|| {
                    let proof = ProofBuf::new(&output.tree, rng.gen::<usize>() % size);
                    black_box(proof);
                })
            },
        );

        g.bench_with_input(
            BenchmarkId::new("gen-proof-resume", size),
            &size,
            |b, size| {
                let mut rng = thread_rng();
                b.iter(|| {
                    let i = (rng.gen::<usize>() % size).max(1);
                    let proof = ProofBuf::resume(&output.tree, i);
                    black_box(proof);
                })
            },
        );

        g.bench_with_input(
            BenchmarkId::new("verify-proof-beginning", size),
            &size,
            |b, size| {
                let mut rng = thread_rng();

                let i = rng.gen::<usize>() % size;
                let proof = ProofBuf::new(&output.tree, i);

                b.iter(|| {
                    let mut verifier = IncrementalVerifier::new(*output.hash.as_bytes(), i);
                    verifier.feed_proof(proof.as_slice()).unwrap();
                    black_box(verifier);
                })
            },
        );

        g.bench_with_input(
            BenchmarkId::new("verify-proof-resume", size),
            &size,
            |b, size| {
                let mut rng = thread_rng();

                let i = rng.gen::<usize>() % (size - 1);
                let proof_initial = ProofBuf::new(&output.tree, i);
                let proof = ProofBuf::resume(&output.tree, i + 1);

                let mut block_hasher = BlockHasher::new();
                block_hasher.set_block(i);
                block_hasher.update(&block_data(i));

                b.iter_batched(
                    || {
                        let mut verifier = IncrementalVerifier::new(*output.hash.as_bytes(), i);
                        verifier.feed_proof(proof_initial.as_slice()).unwrap();
                        verifier.verify(block_hasher.clone()).unwrap();
                        verifier
                    },
                    |mut verifier| {
                        verifier.feed_proof(proof.as_slice()).unwrap();
                        black_box(verifier);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    g.finish();
}

#[inline(always)]
fn block_data(n: usize) -> [u8; 256 * 1024] {
    let mut data = [0; 256 * 1024];
    for i in data.chunks_exact_mut(2) {
        i[0] = n as u8;
        i[1] = (n / 256) as u8;
    }
    data
}

criterion_group!(benches, bench_tree);
criterion_main!(benches);
