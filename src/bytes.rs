use super::SliceRingBuffer;
use bytes::{Buf, BufMut, buf::UninitSlice};
use std::{mem::MaybeUninit, ptr};

impl Buf for SliceRingBuffer<u8> {
    #[inline]
    fn remaining(&self) -> usize { self.len() }

    #[inline]
    fn chunk(&self) -> &[u8] { self.as_slice() }

    #[inline]
    fn advance(&mut self, cnt: usize) {
        assert!(cnt <= self.len(), "cannot advance past the end of the buffer");
        self.move_head(cnt as isize);
    }
}

unsafe impl BufMut for SliceRingBuffer<u8> {
    #[inline]
    fn remaining_mut(&self) -> usize { self.capacity().saturating_sub(self.len()) }

    #[inline]
    unsafe fn advance_mut(&mut self, cnt: usize) {
        let new_len = self.len() + cnt;
        assert!(new_len <= self.capacity(), "cannot advance past the buffer's capacity");
        self.len = new_len;
    }

    #[inline]
    fn chunk_mut(&mut self) -> &mut bytes::buf::UninitSlice {
        let uninit_slice = self.uninit_slice();
        unsafe { &mut *(uninit_slice as *mut [MaybeUninit<u8>] as *mut UninitSlice) }
    }

    #[inline]
    fn put_slice(&mut self, src: &[u8]) {
        self.reserve(src.len());
        let uninit_slice = self.uninit_slice();
        debug_assert!(uninit_slice.len() >= src.len());
        unsafe {
            let dst = uninit_slice.as_mut_ptr().cast::<u8>();
            ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        self.move_tail(src.len() as isize);
    }

    #[inline]
    fn put<T: Buf>(&mut self, mut src: T)
    where
        Self: Sized,
    {
        while src.has_remaining() {
            let chunk = src.chunk();
            self.put_slice(chunk);
            src.advance(chunk.len());
        }
    }
}

#[cfg(all(test, feature = "bytes"))]
mod tests {
    use crate::SliceRingBuffer;
    use bytes::{Buf, BufMut};

    #[test]
    fn test_put_slice_and_chunk() {
        let mut buf = SliceRingBuffer::<u8>::new();
        assert_eq!(buf.remaining(), 0);
        assert!(buf.chunk().is_empty());

        buf.put_slice(b"hello");
        assert_eq!(buf.remaining(), 5);
        assert_eq!(buf.chunk(), b"hello");

        buf.put_slice(b" world");
        assert_eq!(buf.remaining(), 11);
        assert_eq!(buf.chunk(), b"hello world");
    }

    #[test]
    fn test_advance() {
        let mut buf = SliceRingBuffer::from(b"hello world".to_vec());
        assert_eq!(buf.remaining(), 11);

        buf.advance(6);
        assert_eq!(buf.remaining(), 5);
        assert_eq!(buf.chunk(), b"world");
        assert_eq!(buf.as_slice(), b"world");

        buf.advance(5);
        assert_eq!(buf.remaining(), 0);
        assert!(buf.chunk().is_empty());
        assert!(buf.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_advance_past_end() {
        let mut buf = SliceRingBuffer::from(b"short".to_vec());
        buf.advance(10);
    }

    #[test]
    fn test_put_slice_causes_growth() {
        let mut buf = SliceRingBuffer::<u8>::with_capacity(4);
        // 修正 #1：我们只保证容量至少是请求的大小。
        assert!(buf.capacity() >= 4);

        buf.put_slice(b"1234");
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.remaining(), 4);
        // remaining_mut() 可能为0，也可能不为0，取决于实际分配的大小
        // 所以我们不在这里断言它

        buf.put_slice(b"5678");
        assert_eq!(buf.len(), 8);
        assert_eq!(buf.remaining(), 8);
        assert!(buf.capacity() >= 8);
        assert_eq!(buf.chunk(), b"12345678");
    }

    #[test]
    fn test_ring_buffer_wrapping_behavior() {
        // 创建一个容量至少为5的缓冲区
        let mut buf = SliceRingBuffer::<u8>::with_capacity(5);

        buf.put_slice(b"abcde");
        assert_eq!(buf.chunk(), b"abcde");
        assert_eq!(buf.head(), 0);

        // 修正 #2：移除对 tail() 的不正确断言。
        // tail() 的具体值取决于实际分配的物理容量，
        // 我们不应该在测试中依赖这个实现细节。
        // assert_eq!(buf.tail(), 0); // <-- 移除这一行

        buf.advance(3); // 消耗 'a', 'b', 'c'
        assert_eq!(buf.chunk(), b"de");
        assert_eq!(buf.head(), 3);
        assert_eq!(buf.len(), 2);

        // 写入 'f', 'g', 'h'。这将跨越物理缓冲区的边界
        buf.put_slice(b"fgh");
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.head(), 3);

        // 这才是最重要的断言：无论内部如何环绕，
        // `as_slice()` 提供的逻辑视图必须是正确的。
        assert_eq!(buf.as_slice(), b"defgh");

        buf.advance(5);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_put_and_get_u64() {
        let mut buf = SliceRingBuffer::<u8>::new();
        let number: u64 = 0x0102030405060708;

        buf.put_u64(number);
        assert_eq!(buf.len(), 8);
        assert_eq!(buf.chunk(), &number.to_be_bytes());

        let retrieved_number = buf.get_u64();
        assert_eq!(retrieved_number, number);
        assert!(buf.is_empty(), "get_u64 should consume the bytes");
    }

    #[test]
    fn test_advance_mut_and_chunk_mut() {
        let mut buf = SliceRingBuffer::<u8>::with_capacity(10);
        // 修正 #3：我们只保证容量至少是请求的大小。
        assert!(buf.capacity() >= 10);
        assert!(buf.remaining_mut() >= 10);

        let chunk = buf.chunk_mut();
        let data_to_write = b"test";
        assert!(chunk.len() >= data_to_write.len());

        unsafe {
            let ptr = chunk.as_mut_ptr() as *mut u8;
            ptr.copy_from_nonoverlapping(data_to_write.as_ptr(), data_to_write.len());
            buf.advance_mut(data_to_write.len());
        }

        assert_eq!(buf.len(), 4);
        assert_eq!(buf.remaining(), 4);
        assert!(buf.remaining_mut() >= 6); // 剩余可写空间至少是 10-4=6
        assert_eq!(buf.chunk(), b"test");
    }
}
