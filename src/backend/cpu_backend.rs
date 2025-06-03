use crate::storage::{BinaryOperation, UnaryOperation};
use crate::{index::StridedIndex, DType, Error, Result, Shape};

#[derive(Debug, Clone)]
pub enum CPUStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CPUStorage {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            CPUStorage::F32(_) => DType::F32,
            CPUStorage::F64(_) => DType::F64,
        }
    }

    pub(crate) fn affine(
        &self,
        shape: &Shape,
        stride: &[usize],
        mul: f64,
        add: f64,
    ) -> Result<Self> {
        match self {
            Self::F32(storage) => {
                let index = StridedIndex::new(shape.dims(), stride);
                let mul = mul as f32;
                let add = add as f32;
                let data = index.map(|i| storage[i] * mul + add).collect();
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let index = StridedIndex::new(shape.dims(), stride);
                let data = index.map(|i| storage[i] * mul + add).collect();
                Ok(Self::F64(data))
            }
        }
    }

    pub(crate) fn unary_impl<T: UnaryOperation>(
        &self,
        shape: &Shape,
        stride: &[usize],
    ) -> Result<Self> {
        match self {
            Self::F32(storage) => {
                let index = StridedIndex::new(shape.dims(), stride);
                let data = index.map(|i| T::f32(storage[i])).collect();
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let index = StridedIndex::new(shape.dims(), stride);
                let data = index.map(|i| T::f64(storage[i])).collect();
                Ok(Self::F64(data))
            }
        }
    }

    pub(crate) fn binary_operation<T: BinaryOperation>(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        match (self, rhs) {
            (CPUStorage::F32(lhs), CPUStorage::F32(rhs)) => {
                let lhs_index = StridedIndex::new(shape.dims(), lhs_stride);
                let rhs_index = StridedIndex::new(shape.dims(), rhs_stride);
                let data = lhs_index
                    .zip(rhs_index)
                    .map(|(lhs_offset, rhs_offset)| T::f32(lhs[lhs_offset], rhs[rhs_offset]))
                    .collect();

                Ok(Self::F32(data))
            }
            (CPUStorage::F64(lhs), CPUStorage::F64(rhs)) => {
                let lhs_index = StridedIndex::new(shape.dims(), lhs_stride);
                let rhs_index = StridedIndex::new(shape.dims(), rhs_stride);
                let data = lhs_index
                    .zip(rhs_index)
                    .map(|(lhs_offset, rhs_offset)| T::f64(lhs[lhs_offset], rhs[rhs_offset]))
                    .collect();

                Ok(Self::F64(data))
            }
            _ => Err(Error::BinaryOperationDTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
                op: T::NAME,
            }),
        }
    }

    pub(crate) fn sum(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        match self {
            Self::F32(storage) => {
                let index = StridedIndex::new(shape.dims(), stride);
                let value: f32 = index.map(|i| storage[i]).sum();
                Ok(Self::F32(vec![value]))
            }
            Self::F64(storage) => {
                let index = StridedIndex::new(shape.dims(), stride);
                let value: f64 = index.map(|i| storage[i]).sum();
                Ok(Self::F64(vec![value]))
            }
        }
    }
}
