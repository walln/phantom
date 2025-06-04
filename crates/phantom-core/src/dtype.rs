use crate::CPUStorage;

#[derive(Debug)]
pub enum DTypeError {
    UnexpectedDType { expected: DType, actual: DType },
}

impl std::fmt::Display for DTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DTypeError::UnexpectedDType { expected, actual } => {
                write!(
                    f,
                    "unexpected dtype, expected: {:?}, actual: {:?}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for DTypeError {}

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
    fn storage_slice(storage: &CPUStorage) -> std::result::Result<&[Self], DTypeError>;
}

macro_rules! with_dtype {
    ($type: ty, $dtype:ident) => {
        impl WithDType for $type {
            const DTYPE: DType = DType::$dtype;

            fn to_cpu_owned(data: Vec<Self>) -> CPUStorage {
                CPUStorage::$dtype(data)
            }

            fn storage_slice(storage: &CPUStorage) -> std::result::Result<&[Self], DTypeError> {
                match storage {
                    CPUStorage::$dtype(data) => Ok(data),
                    _ => Err(DTypeError::UnexpectedDType {
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
