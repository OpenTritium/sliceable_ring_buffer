use super::SliceRingBuffer;
use std::{
    io::{self, Read, Write},
    ptr,
};

impl Read for SliceRingBuffer<u8> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let available_data = self.as_slice();
        let len_to_copy = available_data.len().min(buf.len());
        if len_to_copy == 0 {
            return Ok(0);
        }
        buf[..len_to_copy].copy_from_slice(&available_data[..len_to_copy]);
        self.move_head(len_to_copy.cast_signed());
        Ok(len_to_copy)
    }
}

impl Write for SliceRingBuffer<u8> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.reserve(buf.len());
        let uninit_slice = self.uninit_slice();
        unsafe {
            let dst_ptr = uninit_slice.as_mut_ptr().cast::<u8>();
            ptr::copy_nonoverlapping(buf.as_ptr(), dst_ptr, buf.len());
        }
        self.move_tail(buf.len().cast_signed());
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

#[cfg(all(test, feature = "io"))]
mod tests {
    use crate::SliceRingBuffer;
    use std::io::{self, Read, Write};

    #[test]
    fn test_read() {
        let mut source = SliceRingBuffer::from(b"hello".to_vec());
        let mut dest = [0u8; 3];

        // 读满 dest
        let bytes_read = source.read(&mut dest).unwrap();
        assert_eq!(bytes_read, 3);
        assert_eq!(&dest, b"hel");
        assert_eq!(source.as_slice(), b"lo"); // 应该消耗掉了 "hel"

        // 读剩余的
        let bytes_read = source.read(&mut dest).unwrap();
        assert_eq!(bytes_read, 2);
        assert_eq!(&dest[..2], b"lo");
        assert!(source.is_empty());

        // 再次读取应该返回 Ok(0)
        let bytes_read = source.read(&mut dest).unwrap();
        assert_eq!(bytes_read, 0);
    }

    #[test]
    fn test_write() {
        let mut buf = SliceRingBuffer::<u8>::new();
        let bytes_written = buf.write(b"world").unwrap();
        assert_eq!(bytes_written, 5);
        assert_eq!(buf.as_slice(), b"world");
    }

    #[test]
    fn test_io_copy() {
        let mut source = SliceRingBuffer::from(b"copy me".to_vec());
        let mut dest = SliceRingBuffer::<u8>::new();

        let bytes_copied = io::copy(&mut source, &mut dest).unwrap();
        assert_eq!(bytes_copied, 7);
        assert!(source.is_empty());
        assert_eq!(dest.as_slice(), b"copy me");
    }
}
