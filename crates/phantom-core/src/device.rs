use crate::backend::cpu_backend::CPUStorage;
use crate::shape::ShapeError;
use crate::{storage::Storage, DType, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
}

impl Device {
    pub fn zeros(&self, shape: &Shape, dtype: DType) -> Storage {
        let elem_count: usize = shape.elem_count();
        match self {
            Device::CPU => {
                let storage = match dtype {
                    DType::F32 => CPUStorage::F32(vec![0f32; elem_count]),
                    DType::F64 => CPUStorage::F64(vec![0f64; elem_count]),
                };
                Storage::CPU(storage)
            }
        }
    }

    pub fn ones(&self, shape: &Shape, dtype: DType) -> Storage {
        let elem_count: usize = shape.elem_count();
        match self {
            Device::CPU => {
                let storage = match dtype {
                    DType::F32 => CPUStorage::F32(vec![1f32; elem_count]),
                    DType::F64 => CPUStorage::F64(vec![1f64; elem_count]),
                };
                Storage::CPU(storage)
            }
        }
    }

    pub fn tensor<A: NDArray>(&self, data: A) -> Storage {
        match self {
            Device::CPU => Storage::CPU(data.to_cpu()),
        }
    }
}

pub trait NDArray {
    fn shape(&self) -> std::result::Result<Shape, ShapeError>;
    fn to_cpu(&self) -> CPUStorage;
}

impl<S: crate::WithDType> NDArray for S {
    fn shape(&self) -> std::result::Result<Shape, ShapeError> {
        Ok(Shape::from(()))
    }

    fn to_cpu(&self) -> CPUStorage {
        S::to_cpu(&[*self])
    }
}

impl<S: crate::WithDType> NDArray for &[S] {
    fn shape(&self) -> std::result::Result<Shape, ShapeError> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu(&self) -> CPUStorage {
        S::to_cpu(self)
    }
}

impl<S: crate::WithDType, const N: usize> NDArray for &[S; N] {
    fn shape(&self) -> std::result::Result<Shape, ShapeError> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu(&self) -> CPUStorage {
        S::to_cpu(self.as_slice())
    }
}

impl<S: crate::WithDType, const N: usize, const M: usize> NDArray for &[[S; N]; M] {
    fn shape(&self) -> std::result::Result<Shape, ShapeError> {
        Ok(Shape::from((M, N)))
    }

    fn to_cpu(&self) -> CPUStorage {
        S::to_cpu_owned(self.concat())
    }
}
