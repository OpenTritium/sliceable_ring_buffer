use super::SliceRingBuffer;
use std::{
    io,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

impl AsyncRead for SliceRingBuffer<u8> {
    fn poll_read(mut self: Pin<&mut Self>, _cx: &mut Context<'_>, buf: &mut ReadBuf<'_>) -> Poll<io::Result<()>> {
        let available_data = self.as_slice();
        let len_to_copy = available_data.len().min(buf.remaining());
        buf.put_slice(&available_data[..len_to_copy]);
        self.move_head(len_to_copy as isize);
        Poll::Ready(Ok(()))
    }
}

impl AsyncWrite for SliceRingBuffer<u8> {
    fn poll_write(mut self: Pin<&mut Self>, _cx: &mut Context<'_>, buf: &[u8]) -> Poll<Result<usize, io::Error>> {
        self.reserve(buf.len());
        let uninit_slice = self.uninit_slice();
        unsafe {
            let dst_ptr = uninit_slice.as_mut_ptr().cast::<u8>();
            std::ptr::copy_nonoverlapping(buf.as_ptr(), dst_ptr, buf.len());
        }
        self.move_tail(buf.len() as isize);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> { Poll::Ready(Ok(())) }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> { Poll::Ready(Ok(())) }
}

#[cfg(all(test, feature = "tokio-io"))]
mod tokio_tests {
    use super::SliceRingBuffer;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn test_async_read() {
        let mut buf = SliceRingBuffer::from(b"async read test".to_vec());
        let mut dest = Vec::new();

        let bytes_read = buf.read_to_end(&mut dest).await.unwrap();
        assert_eq!(bytes_read, 15);
        assert_eq!(dest, b"async read test");
        assert!(buf.is_empty());
    }

    #[tokio::test]
    async fn test_async_write() {
        let mut buf = SliceRingBuffer::<u8>::new();
        buf.write_all(b"async write").await.unwrap();
        assert_eq!(buf.as_slice(), b"async write");
    }

    #[tokio::test]
    async fn test_tokio_copy() {
        let mut source = SliceRingBuffer::from(b"async copy me".to_vec());
        let mut dest = SliceRingBuffer::<u8>::new();

        let bytes_copied = tokio::io::copy(&mut source, &mut dest).await.unwrap();
        assert_eq!(bytes_copied, 13);
        assert!(source.is_empty());
        assert_eq!(dest.as_slice(), b"async copy me");
    }
}
