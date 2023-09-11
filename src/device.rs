use crate::backend::backend::BackendDevice;
use crate::backend::cpu_backend::{CPUDevice, CPUStorage};
use crate::WithDType;
use crate::{storage::Storage, DType, Result, Shape};

/// A device location is an actual physical device while a device is a logical
/// device. For example a GPU device means the tensor is loaded on a GPU while a
/// GPU location refers to the specific GPU that the tensor is loaded on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    CPU,
    MPS,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    MPS,
}

impl Device {
    pub fn same_id(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::CPU, Self::CPU) => true,
            (Self::MPS, Self::MPS) => true,
            // For enums that carry values then call .same_device to compare
            _ => false,
        }
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::CPU, Self::CPU) => true,
            (Self::MPS, Self::MPS) => true,
            _ => false,
        }
    }

    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::CPU => DeviceLocation::CPU,
            Self::MPS => DeviceLocation::MPS,
        }
    }

    pub fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::CPU => {
                let storage = CPUDevice.zeros_impl(shape, dtype)?;
                Ok(Storage::CPU(storage))
            }
            Device::MPS => todo!(),
        }
    }

    pub fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::CPU => {
                let storage = CPUDevice.ones_impl(shape, dtype)?;
                Ok(Storage::CPU(storage))
            }
            Device::MPS => todo!(),
        }
    }

    pub fn tensor<A: NDArray>(&self, data: A) -> Storage {
        match self {
            Device::CPU => Storage::CPU(data.to_cpu()),
            Device::MPS => todo!(),
        }
    }

    pub fn storage<A: NDArray>(&self, array: A) -> Result<Storage> {
        match self {
            Device::CPU => Ok(Storage::CPU(array.to_cpu())),
            Device::MPS => todo!(),
        }
    }

    pub fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage> {
        match self {
            Device::CPU => Ok(Storage::CPU(S::to_cpu_owned(data))),
            Device::MPS => todo!(),
        }
    }
}

pub trait NDArray {
    fn shape(&self) -> Result<Shape>;
    fn to_cpu(&self) -> CPUStorage;
}

impl<S: crate::WithDType> NDArray for S {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(()))
    }

    fn to_cpu(&self) -> CPUStorage {
        S::to_cpu(&[*self])
    }
}

impl<S: crate::WithDType> NDArray for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu(&self) -> CPUStorage {
        S::to_cpu(self)
    }
}

impl<S: crate::WithDType, const N: usize> NDArray for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu(&self) -> CPUStorage {
        S::to_cpu(self.as_slice())
    }
}

impl<S: crate::WithDType, const N: usize, const M: usize> NDArray for &[[S; N]; M] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((M, N)))
    }

    fn to_cpu(&self) -> CPUStorage {
        S::to_cpu_owned(self.concat())
    }
}
