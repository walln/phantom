use crate::{CPUStorage, Error, Result};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }
}

pub trait WithDType: Sized + Copy {
    const DTYPE: DType;

    fn to_cpu_owned(data: Vec<Self>) -> CPUStorage;
    fn to_cpu(data: &[Self]) -> CPUStorage {
        Self::to_cpu_owned(data.to_vec())
    }
    fn storage_slice(storage: &CPUStorage) -> Result<&[Self]>;
}

macro_rules! with_dtype {
    ($type: ty, $dtype:ident) => {
        impl WithDType for $type {
            const DTYPE: DType = DType::$dtype;

            fn to_cpu_owned(data: Vec<Self>) -> CPUStorage {
                CPUStorage::$dtype(data)
            }

            fn storage_slice(storage: &CPUStorage) -> Result<&[Self]> {
                match storage {
                    CPUStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        actual: storage.dtype(),
                    }),
                }
            }
        }
    };
}

with_dtype!(f32, F32);
with_dtype!(f64, F64);
