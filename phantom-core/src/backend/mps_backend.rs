use crate::device::DeviceLocation;
use crate::operation::{BinaryOperation, UnaryOperation};
use crate::CPUStorage;
use crate::{DType, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub enum MPSStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct MPSDevice;

impl crate::backend::backend::BackendStorage for MPSStorage {
    type Device = MPSDevice;

    fn device(&self) -> &Self::Device {
        todo!()
    }

    fn dtype(&self) -> DType {
        match self {
            MPSStorage::F32(_) => DType::F32,
            MPSStorage::F64(_) => DType::F64,
        }
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        todo!()
    }

    fn to_cpu(&self) -> Result<crate::CPUStorage> {
        todo!()
    }

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        todo!()
    }

    fn copy_strided_source(
        &self,
        destination: &mut Self,
        destination_offset: usize,
        destination_layout: &Layout,
    ) -> Result<()> {
        todo!()
    }

    fn binary_operation<T: BinaryOperation>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        todo!()
    }

    fn unary_operation<T: UnaryOperation>(&self, layout: &Layout) -> Result<Self> {
        todo!()
    }

    fn affine(&self, layout: &Layout, add: f64, mul: f64) -> Result<Self> {
        todo!()
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        todo!()
    }

    fn where_condition(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
    ) -> Result<Self> {
        todo!()
    }

    fn embedding(&self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn sum(&self, _: &Layout, _: &[usize]) -> Result<Self> {
        todo!()
    }
}

impl crate::backend::backend::BackendDevice for MPSDevice {
    type Storage = MPSStorage;

    fn new(_: usize) -> Result<Self> {
        todo!("MPSDevice::new")
    }

    fn location(&self) -> DeviceLocation {
        todo!("MPSDevice::location")
    }

    fn same_device(&self, rhs: &Self) -> bool {
        rhs.location() == self.location()
    }

    fn from_cpu(&self, storage: &CPUStorage) -> Result<Self::Storage> {
        todo!("MPSDevice::from_cpu")
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        todo!("MPSDevice::zeros_impl")
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        todo!("MPSDevice::ones_impl")
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        low: f64,
        high: f64,
    ) -> Result<Self::Storage> {
        todo!("MPSDevice::rand_uniform")
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        todo!("MPSDevice::rand_normal")
    }
}

impl MPSStorage {}
