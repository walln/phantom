use crate::backend::{backend::BackendStorage, CPUStorage, MPSStorage};
use crate::operation::{BinaryOperation, UnaryOperation};
use crate::{DType, Device, Error, Layout, Result};

pub enum Storage {
    CPU(CPUStorage),
    MPS(MPSStorage),
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Storage::CPU { .. } => Device::CPU,
            Storage::MPS { .. } => Device::MPS,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Storage::CPU(storage) => storage.dtype(),
            Storage::MPS(storage) => storage.dtype(),
        }
    }

    pub(crate) fn matches_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.device();
        let rhs = rhs.device();

        if lhs != rhs {
            Err(Error::BinaryOperationDeviceMismatch {
                lhs: lhs.location(),
                rhs: rhs.location(),
                op,
            })
        } else {
            Ok(())
        }
    }

    pub(crate) fn matches_dtype(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.dtype();
        let rhs = rhs.dtype();

        if lhs != rhs {
            Err(Error::BinaryOperationDTypeMismatch { lhs, rhs, op })
        } else {
            Ok(())
        }
    }

    pub(crate) fn unary_operation<T: UnaryOperation>(&self, layout: &Layout) -> Result<Self> {
        match self {
            Storage::CPU(storage) => {
                let storage = storage.unary_operation::<T>(layout)?;
                Ok(Self::CPU(storage))
            }
            Storage::MPS(storage) => {
                let storage = storage.unary_operation::<T>(layout)?;
                Ok(Self::MPS(storage))
            }
        }
    }

    pub(crate) fn binary_operation<T: BinaryOperation>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        // Check the operands are valid for this operation.
        self.matches_device(rhs, T::NAME)?;
        self.matches_dtype(rhs, T::NAME)?;

        // This will need contiguous layout optimizations later
        match (self, rhs) {
            (Storage::CPU(lhs), Storage::CPU(rhs)) => {
                let storage = lhs.binary_operation::<T>(rhs, lhs_layout, rhs_layout)?;
                Ok(Self::CPU(storage))
            }
            (Storage::MPS(lhs), Storage::MPS(rhs)) => {
                let storage = lhs.binary_operation::<T>(rhs, lhs_layout, rhs_layout)?;
                Ok(Self::MPS(storage))
            }
            (_, _) => Err(Error::BinaryOperationDeviceMismatch {
                lhs: self.device().location(),
                rhs: rhs.device().location(),
                op: T::NAME,
            }),
        }
    }

    pub(crate) fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        match self {
            Storage::CPU(storage) => {
                let storage = storage.affine(layout, mul, add)?;
                Ok(Self::CPU(storage))
            }
            Storage::MPS(storage) => {
                let storage = storage.affine(layout, mul, add)?;
                Ok(Self::MPS(storage))
            }
        }
    }

    pub(crate) fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        match self {
            Storage::CPU(storage) => {
                let storage = storage.to_dtype(layout, dtype)?;
                Ok(Self::CPU(storage))
            }
            Storage::MPS(storage) => {
                let storage = storage.to_dtype(layout, dtype)?;
                Ok(Self::MPS(storage))
            }
        }
    }

    /// The source is stridable and the destination is contiguous.
    pub(crate) fn copy_strided_source(
        &self,
        destination: &mut Self,
        destination_offset: usize,
        source_layout: &Layout,
    ) -> Result<()> {
        match (self, destination) {
            (Self::CPU(source), Self::CPU(destination)) => {
                source.copy_strided_source(destination, destination_offset, source_layout)
            }
            (Self::MPS(source), Self::MPS(destination)) => {
                source.copy_strided_source(destination, destination_offset, source_layout)
            }
            (lhs, rhs) => Err(Error::BinaryOperationDeviceMismatch {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "copy",
            }),
        }
    }
}
