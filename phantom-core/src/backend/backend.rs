use crate::operation::{BinaryOperation, UnaryOperation};
use crate::{CPUStorage, DType, Layout, Result, Shape};

pub(crate) trait BackendStorage: Sized {
    type Device: BackendDevice;

    fn to_cpu(&self) -> Result<CPUStorage>;

    fn dtype(&self) -> DType;

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self>;

    fn device(&self) -> &Self::Device;

    fn try_clone(&self, layout: &Layout) -> Result<Self>;

    fn copy_strided_source(
        &self,
        destination: &mut Self,
        destination_offset: usize,
        source_layout: &Layout,
    ) -> Result<()>;

    fn unary_operation<T: UnaryOperation>(&self, layout: &Layout) -> Result<Self>;

    fn binary_operation<T: BinaryOperation>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self>;

    fn affine(&self, layout: &Layout, add: f64, mul: f64) -> Result<Self>;

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self>;

    fn where_condition(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
    ) -> Result<Self>;

    // fn conv1d(
    //     &self,
    //     _l: &Layout,
    //     _kernel: &Self,
    //     _kernel_l: &Layout,
    //     _params: &crate::conv::ParamsConv1D,
    // ) -> Result<Self>;

    fn embedding(&self, _: &Layout, _: &Self, _: &Layout) -> Result<Self>;

    fn sum(&self, _: &Layout, _: &[usize]) -> Result<Self>;
}

pub(crate) trait BackendDevice: Sized + std::fmt::Debug + Clone {
    type Storage: BackendStorage;

    fn new(_: usize) -> Result<Self>;

    fn location(&self) -> crate::device::DeviceLocation;

    fn same_device(&self, _: &Self) -> bool;

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage>;

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage>;

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        lower_bound: f64,
        upper_bound: f64,
    ) -> Result<Self::Storage>;

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage>;

    fn from_cpu(&self, storage: &CPUStorage) -> Result<Self::Storage>;
}
