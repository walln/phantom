use crate::backend::cpu_backend::CPUStorage;
use crate::{DType, Device, Shape};

#[derive(Debug)]
pub enum StorageError {
    BinaryOperationDeviceMismatch {
        lhs: Device,
        rhs: Device,
        op: &'static str,
    },
    BinaryOperationDTypeMismatch {
        lhs: DType,
        rhs: DType,
        op: &'static str,
    },
    BinaryOperationShapeMismatch {
        lhs: Shape,
        rhs: Shape,
        op: &'static str,
    },
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::BinaryOperationDeviceMismatch { lhs, rhs, op } => {
                write!(
                    f,
                    "unexpected device in {}, lhs: {:?}, rhs: {:?}",
                    op, lhs, rhs
                )
            }
            StorageError::BinaryOperationDTypeMismatch { lhs, rhs, op } => {
                write!(
                    f,
                    "unexpected dtype in {}, lhs: {:?}, rhs: {:?}",
                    op, lhs, rhs
                )
            }
            StorageError::BinaryOperationShapeMismatch { lhs, rhs, op } => {
                write!(
                    f,
                    "unexpected shape in {}, lhs: {:?}, rhs: {:?}",
                    op, lhs, rhs
                )
            }
        }
    }
}

impl std::error::Error for StorageError {}

pub enum Storage {
    CPU(CPUStorage),
}

pub(crate) trait UnaryOperation {
    const NAME: &'static str;
    fn f32(value: f32) -> f32;
    fn f64(value: f64) -> f64;
}

pub(crate) trait BinaryOperation {
    const NAME: &'static str;
    fn f32(lhs: f32, rhs: f32) -> f32;
    fn f64(lhs: f64, rhs: f64) -> f64;
}

struct Add;

impl BinaryOperation for Add {
    const NAME: &'static str = "add";
    fn f32(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
    fn f64(lhs: f64, rhs: f64) -> f64 {
        lhs + rhs
    }
}

struct Sub;

impl BinaryOperation for Sub {
    const NAME: &'static str = "sub";
    fn f32(lhs: f32, rhs: f32) -> f32 {
        lhs - rhs
    }
    fn f64(lhs: f64, rhs: f64) -> f64 {
        lhs - rhs
    }
}

struct Mul;

impl BinaryOperation for Mul {
    const NAME: &'static str = "mul";
    fn f32(lhs: f32, rhs: f32) -> f32 {
        lhs * rhs
    }
    fn f64(lhs: f64, rhs: f64) -> f64 {
        lhs * rhs
    }
}

struct Div;

impl BinaryOperation for Div {
    const NAME: &'static str = "div";
    fn f32(lhs: f32, rhs: f32) -> f32 {
        lhs / rhs
    }
    fn f64(lhs: f64, rhs: f64) -> f64 {
        lhs / rhs
    }
}

struct Sqr;

impl UnaryOperation for Sqr {
    const NAME: &'static str = "sqr";
    fn f32(value: f32) -> f32 {
        value * value
    }
    fn f64(value: f64) -> f64 {
        value * value
    }
}

struct Sqrt;

impl UnaryOperation for Sqrt {
    const NAME: &'static str = "sqrt";
    fn f32(value: f32) -> f32 {
        value.sqrt()
    }
    fn f64(value: f64) -> f64 {
        value.sqrt()
    }
}

struct Neg;

impl UnaryOperation for Neg {
    const NAME: &'static str = "neg";
    fn f32(value: f32) -> f32 {
        -value
    }
    fn f64(value: f64) -> f64 {
        -value
    }
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Storage::CPU { .. } => Device::CPU,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Storage::CPU(storage) => storage.dtype(),
        }
    }

    pub(crate) fn matches_device(
        &self,
        rhs: &Self,
        op: &'static str,
    ) -> std::result::Result<(), StorageError> {
        let lhs = self.device();
        let rhs = rhs.device();

        if lhs != rhs {
            Err(StorageError::BinaryOperationDeviceMismatch { lhs, rhs, op })
        } else {
            Ok(())
        }
    }

    pub(crate) fn matches_dtype(
        &self,
        rhs: &Self,
        op: &'static str,
    ) -> std::result::Result<(), StorageError> {
        let lhs = self.dtype();
        let rhs = rhs.dtype();

        if lhs != rhs {
            Err(StorageError::BinaryOperationDTypeMismatch { lhs, rhs, op })
        } else {
            Ok(())
        }
    }

    fn unary_operation<T: UnaryOperation>(
        &self,
        shape: &Shape,
        stride: &[usize],
    ) -> std::result::Result<Self, StorageError> {
        match self {
            Storage::CPU(storage) => {
                let storage = storage.unary_impl::<T>(shape, stride)?;
                Ok(Self::CPU(storage))
            }
        }
    }

    fn binary_operation<T: BinaryOperation>(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> std::result::Result<Self, StorageError> {
        // Check the operands are valid for this operation.
        self.matches_device(rhs, T::NAME)?;
        self.matches_dtype(rhs, T::NAME)?;

        // This will need contiguous layout optimizations later
        match (self, rhs) {
            (Storage::CPU(lhs), Storage::CPU(rhs)) => {
                let storage = lhs.binary_operation::<T>(rhs, shape, lhs_stride, rhs_stride)?;
                Ok(Self::CPU(storage))
            }
        }
    }

    pub(crate) fn add(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> std::result::Result<Self, StorageError> {
        self.binary_operation::<Add>(rhs, shape, lhs_stride, rhs_stride)
    }

    pub(crate) fn sub(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> std::result::Result<Self, StorageError> {
        self.binary_operation::<Sub>(rhs, shape, lhs_stride, rhs_stride)
    }

    pub(crate) fn mul(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> std::result::Result<Self, StorageError> {
        self.binary_operation::<Mul>(rhs, shape, lhs_stride, rhs_stride)
    }

    pub(crate) fn div(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> std::result::Result<Self, StorageError> {
        self.binary_operation::<Div>(rhs, shape, lhs_stride, rhs_stride)
    }

    pub(crate) fn affine(
        &self,
        shape: &Shape,
        stride: &[usize],
        mul: f64,
        add: f64,
    ) -> std::result::Result<Self, StorageError> {
        match self {
            Storage::CPU(storage) => {
                let storage = storage.affine(shape, stride, mul, add)?;
                Ok(Self::CPU(storage))
            }
        }
    }

    pub(crate) fn sqr(
        &self,
        shape: &Shape,
        stride: &[usize],
    ) -> std::result::Result<Self, StorageError> {
        self.unary_operation::<Sqr>(shape, stride)
    }

    pub(crate) fn sqrt(
        &self,
        shape: &Shape,
        stride: &[usize],
    ) -> std::result::Result<Self, StorageError> {
        self.unary_operation::<Sqrt>(shape, stride)
    }

    pub(crate) fn neg(
        &self,
        shape: &Shape,
        stride: &[usize],
    ) -> std::result::Result<Self, StorageError> {
        self.unary_operation::<Neg>(shape, stride)
    }

    pub(crate) fn transpose(
        &self,
        shape: &Shape,
        stride: &[usize],
    ) -> std::result::Result<Self, StorageError> {
        match self {
            Storage::CPU(storage) => {
                let storage = storage.transpose(shape, stride)?;
                Ok(Self::CPU(storage))
            }
        }
    }

    pub(crate) fn matmul(
        &self,
        rhs: &Self,
        lhs_shape: (usize, usize),
        lhs_stride: &[usize],
        rhs_shape: (usize, usize),
        rhs_stride: &[usize],
    ) -> std::result::Result<Self, StorageError> {
        self.matches_device(rhs, "matmul")?;
        self.matches_dtype(rhs, "matmul")?;
        match (self, rhs) {
            (Storage::CPU(lhs), Storage::CPU(rhs)) => {
                let storage = lhs.matmul(lhs_shape, lhs_stride, rhs, rhs_shape, rhs_stride)?;
                Ok(Self::CPU(storage))
            }
        }
    }
}
