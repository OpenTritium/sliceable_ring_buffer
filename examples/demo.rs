use slice_ring_buffer::SliceRingBuffer;
fn main() {
    let mut rb = SliceRingBuffer::with_capacity(8);
    let s = String::from("I love Rust");
    for idx in 0..100000 {
        rb.push_back(format!("{} {} times", s, idx));
    }
    for e in rb.iter() {
        println!("{}", e);
    }
    for e in rb.into_iter() {
        println!("{}", e);
    }
}
