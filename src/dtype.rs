use crate::backend::backend::BackendStorage;
use crate::{CPUStorage, Error, Result};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    U32,
}

pub trait WithDType: Sized + Copy + 'static + num_traits::NumAssign {
    const DTYPE: DType;

    fn to_cpu_owned(data: Vec<Self>) -> CPUStorage;
    fn to_cpu(data: &[Self]) -> CPUStorage {
        Self::to_cpu_owned(data.to_vec())
    }
    fn cpu_storage_slice(storage: &CPUStorage) -> Result<&[Self]>;
    fn cpu_storage_data(storage: CPUStorage) -> Result<Vec<Self>>;

    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}

macro_rules! with_dtype {
    ($type: ty, $dtype:ident, $from_f64: expr, $to_f64: expr) => {
        impl WithDType for $type {
            const DTYPE: DType = DType::$dtype;

            fn to_cpu_owned(data: Vec<Self>) -> CPUStorage {
                CPUStorage::$dtype(data)
            }

            fn cpu_storage_slice(storage: &CPUStorage) -> Result<&[Self]> {
                match storage {
                    CPUStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        actual: storage.dtype(),
                    }),
                }
            }

            fn cpu_storage_data(storage: CPUStorage) -> Result<Vec<Self>> {
                match storage {
                    CPUStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        actual: storage.dtype(),
                    }),
                }
            }

            fn to_f64(self) -> f64 {
                $to_f64(self)
            }

            fn from_f64(value: f64) -> Self {
                $from_f64(value)
            }
        }
    };
}

with_dtype!(f32, F32, |v: f64| v as f32, |v: f32| v as f64);
with_dtype!(f64, F64, |v: f64| v, |v: f64| v);
with_dtype!(u32, U32, |v: f64| v as u32, |v: u32| v as f64);
