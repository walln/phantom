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

    pub(crate) fn transpose(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        let (rows, cols) = shape.rank_two()?;
        match self {
            CPUStorage::F32(storage) => {
                let mut out = vec![0f32; rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        let src = r * stride[0] + c * stride[1];
                        out[c * rows + r] = storage[src];
                    }
                }
                Ok(Self::F32(out))
            }
            CPUStorage::F64(storage) => {
                let mut out = vec![0f64; rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        let src = r * stride[0] + c * stride[1];
                        out[c * rows + r] = storage[src];
                    }
                }
                Ok(Self::F64(out))
            }
        }
    }

    pub(crate) fn matmul(
        &self,
        lhs_shape: (usize, usize),
        lhs_stride: &[usize],
        rhs: &Self,
        rhs_shape: (usize, usize),
        rhs_stride: &[usize],
    ) -> Result<Self> {
        let (m, k) = lhs_shape;
        let (k_rhs, n) = rhs_shape;
        if k != k_rhs {
            return Err(Error::BinaryOperationShapeMismatch {
                lhs: Shape::from((m, k)),
                rhs: Shape::from((k_rhs, n)),
                op: "matmul",
            });
        }
        match (self, rhs) {
            (CPUStorage::F32(lhs), CPUStorage::F32(rhs)) => {
                let mut out = vec![0f32; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0f32;
                        for p in 0..k {
                            let lhs_idx = i * lhs_stride[0] + p * lhs_stride[1];
                            let rhs_idx = p * rhs_stride[0] + j * rhs_stride[1];
                            sum += lhs[lhs_idx] * rhs[rhs_idx];
                        }
                        out[i * n + j] = sum;
                    }
                }
                Ok(Self::F32(out))
            }
            (CPUStorage::F64(lhs), CPUStorage::F64(rhs)) => {
                let mut out = vec![0f64; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0f64;
                        for p in 0..k {
                            let lhs_idx = i * lhs_stride[0] + p * lhs_stride[1];
                            let rhs_idx = p * rhs_stride[0] + j * rhs_stride[1];
                            sum += lhs[lhs_idx] * rhs[rhs_idx];
                        }
                        out[i * n + j] = sum;
                    }
                }
                Ok(Self::F64(out))
            }
            _ => Err(Error::BinaryOperationDTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
                op: "matmul",
            }),
        }
    }
}
