use crate::SliceRingBuffer;
use serde::{
    Deserialize, Serialize,
    de::{SeqAccess, Visitor},
    ser::SerializeSeq,
};
use std::fmt;

impl<T: Serialize> Serialize for SliceRingBuffer<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for element in self.iter() {
            seq.serialize_element(element)?;
        }
        seq.end()
    }
}

struct SliceRingBufferVisitor<T>(std::marker::PhantomData<T>);

impl<'de, T: Deserialize<'de>> Visitor<'de> for SliceRingBufferVisitor<T> {
    type Value = SliceRingBuffer<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result { formatter.write_str("a sequence") }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let cap = seq.size_hint().unwrap_or(0);
        let mut buffer = SliceRingBuffer::with_capacity(cap);
        while let Some(elem) = seq.next_element()? {
            buffer.push_back(elem);
        }
        Ok(buffer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for SliceRingBuffer<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(SliceRingBufferVisitor(std::marker::PhantomData))
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::SliceRingBuffer;

    #[test]
    fn test_serialize_and_deserialize() {
        // 创建一个 buffer
        let original_buf = SliceRingBuffer::from([1u8, 2, 3, 4]);

        // 序列化为 JSON 字符串
        let json_data = serde_json::to_string(&original_buf).unwrap();
        assert_eq!(json_data, "[1,2,3,4]");

        // 从 JSON 字符串反序列化回来
        let deserialized_buf: SliceRingBuffer<u8> = serde_json::from_str(&json_data).unwrap();

        assert_eq!(original_buf, deserialized_buf);
    }

    #[test]
    fn test_serde_empty() {
        let original_buf = SliceRingBuffer::<i32>::new();
        let json_data = serde_json::to_string(&original_buf).unwrap();
        assert_eq!(json_data, "[]");

        let deserialized_buf: SliceRingBuffer<i32> = serde_json::from_str(&json_data).unwrap();
        assert!(deserialized_buf.is_empty());
    }
}
