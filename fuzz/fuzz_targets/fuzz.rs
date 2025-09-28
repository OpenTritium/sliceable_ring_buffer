#![no_main]
use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use sliceable_ring_buffer::SliceRingBuffer;
use std::collections::VecDeque;

#[derive(Debug, Arbitrary)]
enum Op {
    PushBack(u8),
    PushFront(u8),
    PopBack,
    PopFront,
    Insert { index: u8, value: u8 },
    Remove(u8),
    Clear,
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);
    let ops: Vec<Op> = match Vec::<Op>::arbitrary(&mut unstructured) {
        Ok(ops) => ops,
        Err(_) => return,
    };

    let mut srb = SliceRingBuffer::<u8>::new();
    let mut model = VecDeque::<u8>::new();

    for op in ops {
        match op {
            Op::PushBack(val) => {
                srb.push_back(val);
                model.push_back(val);
            }
            Op::PushFront(val) => {
                srb.push_front(val);
                model.push_front(val);
            }
            Op::PopBack => {
                assert_eq!(srb.pop_back(), model.pop_back());
            }
            Op::PopFront => {
                assert_eq!(srb.pop_front(), model.pop_front());
            }
            Op::Insert { index, value } => {
                let index = index as usize % (srb.len() + 1); // 确保 index 在有效范围内
                srb.insert(index, value);
                model.insert(index, value);
            }
            Op::Remove(index) => {
                if !srb.is_empty() {
                    let index = index as usize % srb.len(); // 确保 index 在有效范围内
                    assert_eq!(srb.remove(index), model.remove(index).unwrap());
                }
            }
            Op::Clear => {
                srb.clear();
                model.clear();
            }
        }

        // 5. 在每一步之后，验证不变量 (invariants)
        assert_eq!(srb.len(), model.len());
        assert_eq!(srb.is_empty(), model.is_empty());
        assert_eq!(srb.as_slice(), model.make_contiguous());
    }
});
