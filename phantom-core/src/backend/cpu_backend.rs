use rand::distributions::uniform;

use crate::backend::backend::BackendStorage;
use crate::operation::{BinaryOperation, UnaryOperation};
use crate::{DType, Error, Layout, Result, Shape, WithDType};

use super::backend::BackendDevice;

#[derive(Debug, Clone)]
pub enum CPUStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
    U32(Vec<u32>),
}

#[derive(Debug, Clone)]
pub struct CPUDevice;

impl BackendStorage for CPUStorage {
    type Device = CPUDevice;

    fn device(&self) -> &Self::Device {
        &CPUDevice
    }

    fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::U32(_) => DType::U32,
        }
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        match (self, dtype) {
            (Self::F32(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F32(data))
            }
            (Self::F32(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::F32(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::F64(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::F64(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F64(data))
            }
            (Self::F64(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }

            (Self::U32(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::U32(data))
            }
            (Self::U32(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::U32(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
        }
    }

    fn to_cpu(&self) -> Result<CPUStorage> {
        Ok(self.clone())
    }

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn binary_operation<T: BinaryOperation>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                let data = binary_map(lhs_layout, rhs_layout, lhs, rhs, T::f32);
                Ok(Self::F32(data))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                let data = binary_map(lhs_layout, rhs_layout, lhs, rhs, T::f64);
                Ok(Self::F64(data))
            }
            _ => Err(Error::BinaryOperationDTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
                op: T::NAME,
            }),
        }
    }

    fn unary_operation<T: UnaryOperation>(&self, layout: &Layout) -> Result<Self> {
        match self {
            Self::F32(storage) => {
                let data = unary_map(storage, layout, T::f32);
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, layout, T::f64);
                Ok(Self::F64(data))
            }
            Self::U32(storage) => {
                let data = unary_map(storage, layout, T::u32);
                Ok(Self::U32(data))
            }
        }
    }

    fn affine(&self, layout: &Layout, add: f64, mul: f64) -> Result<Self> {
        Affine(mul, add).map(self, layout)
    }

    fn sum(&self, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let source_dims = layout.dims();
        let mut destination_dims = source_dims.to_vec();
        for &sum_dim in sum_dims.iter() {
            destination_dims[sum_dim] = 1;
        }
        let destination_shape = Shape::from(destination_dims);
        let mut sum_dims = sum_dims.to_vec();

        // When converting the indicies sort the sum dims as they are processed
        // the dimensions are processed from left to right.
        sum_dims.sort();
        let sum_dims_and_stride: Vec<_> = sum_dims
            .iter()
            .map(|&d| {
                (
                    source_dims[d],
                    source_dims[d + 1..].iter().product::<usize>(),
                )
            })
            .collect();
        Sum {
            destination_shape: &destination_shape,
            sum_dims_and_stride,
        }
        .map(self, layout)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        MatMul(bmnk).map(self, lhs_layout, rhs, rhs_layout)
    }

    fn embedding(&self, lhs_layout: &Layout, rhs: &Self, rhs_layout: &Layout) -> Result<Self> {
        let lhs = self.as_slice::<u32>()?;
        let (vocab_size, hidden_size) = rhs_layout.shape().rank_two()?;
        Embedding {
            vocab_size,
            hidden_size,
            ids: lhs,
            ids_layout: lhs_layout,
        }
        .map(rhs, rhs_layout)
    }

    fn where_condition(
        &self,
        layout: &Layout,
        t: &Self,
        t_layout: &Layout,
        f: &Self,
        f_layout: &Layout,
    ) -> Result<Self> {
        let pred = self.as_slice::<u32>()?;
        WhereCondition(pred, layout).map(t, t_layout, f, f_layout)
    }

    fn copy_strided_source(
        &self,
        destination: &mut Self,
        destination_offset: usize,
        source_layout: &Layout,
    ) -> Result<()> {
        match (self, destination) {
            (Self::F32(src), Self::F32(dst)) => {
                copy_strided_source_(src, dst, destination_offset, source_layout)
            }
            (Self::F64(src), Self::F64(dst)) => {
                copy_strided_source_(src, dst, destination_offset, source_layout)
            }
            (_, destination) => {
                // This should be covered by the dtype check above.
                return Err(Error::BinaryOperationDTypeMismatch {
                    lhs: self.dtype(),
                    rhs: destination.dtype(),
                    op: "copy_strided",
                });
            }
        }
        Ok(())
    }
}

impl BackendDevice for CPUDevice {
    type Storage = CPUStorage;

    fn new(_: usize) -> Result<Self> {
        Ok(Self)
    }

    fn location(&self) -> crate::device::DeviceLocation {
        crate::device::DeviceLocation::CPU
    }

    fn same_device(&self, _: &Self) -> bool {
        true
    }

    fn from_cpu(&self, storage: &CPUStorage) -> Result<Self::Storage> {
        Ok(storage.clone())
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        match dtype {
            DType::F32 => Ok(Self::Storage::F32(vec![0.0; elem_count])),
            DType::F64 => Ok(Self::Storage::F64(vec![0.0; elem_count])),
            DType::U32 => Ok(Self::Storage::U32(vec![0; elem_count])),
        }
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        match dtype {
            DType::F32 => Ok(Self::Storage::F32(vec![1.0; elem_count])),
            DType::F64 => Ok(Self::Storage::F64(vec![1.0; elem_count])),
            DType::U32 => Ok(Self::Storage::U32(vec![1; elem_count])),
        }
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        lower_bound: f64,
        upper_bound: f64,
    ) -> Result<Self::Storage> {
        use rand::prelude::*;
        let elem_count = shape.elem_count();
        let mut rng = rand::thread_rng();

        match dtype {
            DType::F32 => {
                let mut data = Vec::new();
                data.reserve(elem_count);
                let uniform =
                    rand::distributions::Uniform::new(lower_bound as f32, upper_bound as f32);
                for _ in 0..elem_count {
                    data.push(rng.sample::<f32, _>(uniform))
                }
                Ok(CPUStorage::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::new();
                data.reserve(elem_count);
                let uniform = uniform::Uniform::new(lower_bound, upper_bound);
                for _ in 0..elem_count {
                    data.push(rng.sample::<f64, _>(uniform))
                }
                Ok(CPUStorage::F64(data))
            }
            _ => Err(Error::UnsupportedDTypeForOperation {
                dtype,
                op: "rand_normal",
            }),
        }
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        use rand::prelude::*;
        let elem_count = shape.elem_count();
        let mut rng = rand::thread_rng();

        match dtype {
            DType::F32 => {
                let mut data = Vec::new();
                data.reserve(elem_count);
                let mean = mean as f32;
                let std = std as f32;
                for _ in 0..elem_count {
                    data.push(rng.sample::<f32, _>(rand::distributions::Standard) * std + mean)
                }
                Ok(CPUStorage::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::new();
                data.reserve(elem_count);
                for _ in 0..elem_count {
                    data.push(rng.sample::<f64, _>(rand::distributions::Standard) * std + mean)
                }
                Ok(CPUStorage::F64(data))
            }
            _ => Err(Error::UnsupportedDTypeForOperation {
                dtype,
                op: "rand_uniform",
            }),
        }
    }
}

impl CPUStorage {
    pub fn as_slice<D: WithDType>(&self) -> Result<&[D]> {
        D::cpu_storage_slice(self)
    }
}

/// Perform a mapping function of a unary operation on a contiguous slice of data.
fn unary_map<T: Copy, U: Copy, F: FnMut(T) -> U>(
    values: &[T],
    layout: &Layout,
    mut f: F,
) -> Vec<U> {
    match layout.contiguous_offsets() {
        Some((o1, o2)) => values[o1..o2].iter().map(|&v| f(v)).collect(),
        None => layout.strided_index().map(|i| f(values[i])).collect(),
    }
}

/// Perform a mapping function of a binary operation on contiguous slices of data.
fn binary_map<T: Copy, F: FnMut(T, T) -> T>(
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    lhs: &[T],
    rhs: &[T],
    mut f: F,
) -> Vec<T> {
    match (
        lhs_layout.contiguous_offsets(),
        rhs_layout.contiguous_offsets(),
    ) {
        (Some((o_l1, o_l2)), Some((o_r1, o_r2))) => lhs[o_l1..o_l2]
            .iter()
            .zip(rhs[o_r1..o_r2].iter())
            .map(|(&l, &r)| f(l, r))
            .collect(),
        _ => lhs_layout
            .strided_index()
            .zip(rhs_layout.strided_index())
            .map(|(lhs_i, rhs_i)| f(lhs[lhs_i], rhs[rhs_i]))
            .collect(),
    }
}

trait UnaryMappable {
    fn f<T: WithDType>(&self, values: &[T], layout: &Layout) -> Result<Vec<T>>;

    fn map(&self, storage: &CPUStorage, layout: &Layout) -> Result<CPUStorage> {
        match storage {
            CPUStorage::F32(values) => Ok(CPUStorage::F32(self.f(values, layout)?)),
            CPUStorage::F64(values) => Ok(CPUStorage::F64(self.f(values, layout)?)),
            CPUStorage::U32(values) => Ok(CPUStorage::U32(self.f(values, layout)?)),
        }
    }
}

struct Affine(f64, f64);

impl UnaryMappable for Affine {
    fn f<T: WithDType>(&self, values: &[T], layout: &Layout) -> Result<Vec<T>> {
        let mul = T::from_f64(self.0);
        let add = T::from_f64(self.1);
        Ok(unary_map(values, layout, |v| v * mul + add))
    }
}

struct Sum<'a> {
    destination_shape: &'a Shape,
    sum_dims_and_stride: Vec<(usize, usize)>,
}

impl<'a> UnaryMappable for Sum<'a> {
    fn f<T: WithDType>(&self, source: &[T], source_layout: &Layout) -> Result<Vec<T>> {
        let mut destination = vec![T::zero(); self.destination_shape.elem_count()];
        for (unstrided_index, source_index) in source_layout.strided_index().enumerate() {
            let mut destination_index = unstrided_index;
            // Set the sum_dims indexes to 0.
            for &(dim, stride) in self.sum_dims_and_stride.iter() {
                // The compiler is able to optimize the following in a single divmod op.
                let (pre, post) = (destination_index / stride, destination_index % stride);
                destination_index = (pre / dim) * stride + post;
            }
            destination[destination_index] += source[source_index];
        }
        Ok(destination)
    }
}

struct Embedding<'a> {
    vocab_size: usize,
    hidden_size: usize,
    ids: &'a [u32],
    ids_layout: &'a Layout,
}

impl<'a> UnaryMappable for Embedding<'a> {
    fn f<T: WithDType>(&self, values: &[T], layout: &Layout) -> Result<Vec<T>> {
        // TODO: We assume that values is contiguous here.
        let values = &values[layout.start_offset()..];
        let mut vals = Vec::with_capacity(self.ids_layout.shape().elem_count() * self.hidden_size);
        // TODO: Optimize for the case where ids are contiguous.
        for index in self.ids_layout.strided_index() {
            let index = self.ids[index];
            let index = index.try_into().map_err(|_| Error::InvalidIndex {
                index: index.try_into().unwrap(),
                vocab_size: self.vocab_size,
                op: "embedding",
            })?;
            if index >= self.vocab_size {
                return Err(Error::InvalidIndex {
                    index,
                    vocab_size: self.vocab_size,
                    op: "take",
                });
            } else {
                let hidden_size = self.hidden_size;
                vals.extend(&values[hidden_size * index..hidden_size * (index + 1)]);
            }
        }
        Ok(vals)
    }
}

trait BinaryMappable {
    const OP: &'static str;
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, v2: &[T], l2: &Layout) -> Result<Vec<T>>;

    fn map(
        &self,
        lhs: &CPUStorage,
        lhs_layout: &Layout,
        rhs: &CPUStorage,
        rhs_layout: &Layout,
    ) -> Result<CPUStorage> {
        match (lhs, rhs) {
            (CPUStorage::F32(lhs), CPUStorage::F32(rhs)) => {
                Ok(CPUStorage::F32(self.f(lhs, lhs_layout, rhs, rhs_layout)?))
            }
            (CPUStorage::F64(lhs), CPUStorage::F64(rhs)) => {
                Ok(CPUStorage::F64(self.f(lhs, lhs_layout, rhs, rhs_layout)?))
            }
            _ => Err(Error::BinaryOperationDTypeMismatch {
                lhs: lhs.dtype(),
                rhs: rhs.dtype(),
                op: Self::OP,
            }),
        }
    }
}

struct MatMul((usize, usize, usize, usize));

impl MatMul {
    fn striding_error(&self, lhs_layout: &Layout, rhs_layout: &Layout, msg: &'static str) -> Error {
        Error::MatMulUnexpectedStride {
            lhs_layout: lhs_layout.clone(),
            rhs_layout: rhs_layout.clone(),
            bmnk: self.0,
            msg,
        }
    }
}

impl BinaryMappable for MatMul {
    const OP: &'static str = "matmul";

    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_layout: &Layout,
        rhs: &[T],
        rhs_layout: &Layout,
    ) -> Result<Vec<T>> {
        use gemm::{gemm, Parallelism};
        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_layout.start_offset()..];
        let rhs = &rhs[rhs_layout.start_offset()..];

        let lhs_stride = lhs_layout.stride();
        let rhs_stride = rhs_layout.stride();
        let rank = lhs_stride.len();
        let lhs_cs = lhs_stride[rank - 1];
        let lhs_rs = lhs_stride[rank - 2];

        let rhs_cs = rhs_stride[rank - 1];
        let rhs_rs = rhs_stride[rank - 2];

        let a_skip: usize = match lhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * lhs_layout.dims()[1] => stride,
            [stride] => stride,
            [] => m * k,
            _ => Err(self.striding_error(lhs_layout, rhs_layout, "non-contiguous lhs"))?,
        };
        let b_skip: usize = match rhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * rhs_layout.dims()[1] => stride,
            [stride] => stride,
            [] => n * k,
            _ => Err(self.striding_error(lhs_layout, rhs_layout, "non-contiguous rhs"))?,
        };
        let c_skip: usize = m * n;

        let dst_shape: Shape = (m, n).into();
        let dst_strides = dst_shape.stride_contiguous();
        let dst_rs = dst_strides[0];
        let dst_cs = dst_strides[1];

        let mut dst = vec![T::zero(); b * m * n];
        let num_threads = crate::utils::get_num_threads();
        let parallelism = if num_threads > 1 {
            Parallelism::Rayon(num_threads)
        } else {
            Parallelism::None
        };
        for step in 0..b {
            let lhs_p = &lhs[step * a_skip..];
            let rhs_p = &rhs[step * b_skip..];
            let dst_p = &mut dst[step * c_skip..];
            unsafe {
                gemm(
                    /* m: usize = */ m,
                    /* n: usize = */ n,
                    /* k: usize = */ k,
                    /* dst: *mut T = */ dst_p.as_mut_ptr(),
                    /* dst_cs: isize = */ dst_cs as isize,
                    /* dst_rs: isize = */ dst_rs as isize,
                    /* read_dst: bool = */ false,
                    /* lhs: *const T = */ lhs_p.as_ptr(),
                    /* lhs_cs: isize = */ lhs_cs as isize,
                    /* lhs_rs: isize = */ lhs_rs as isize,
                    /* rhs: *const T = */ rhs_p.as_ptr(),
                    /* rhs_cs: isize = */ rhs_cs as isize,
                    /* rhs_rs: isize = */ rhs_rs as isize,
                    /* alpha: T = */ T::zero(),
                    /* beta: T = */ T::one(),
                    /* conj_dst: bool = */ false,
                    /* conj_lhs: bool = */ false,
                    /* conj_rhs: bool = */ false,
                    parallelism,
                )
            }
        }
        Ok(dst)
    }
}

struct WhereCondition<'a>(&'a [u32], &'a Layout);

impl<'a> BinaryMappable for WhereCondition<'a> {
    const OP: &'static str = "where_condition";

    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, v2: &[T], l2: &Layout) -> Result<Vec<T>> {
        let values = match (
            self.1.contiguous_offsets(),
            l1.contiguous_offsets(),
            l2.contiguous_offsets(),
        ) {
            (Some((o1, o2)), Some((o_t1, o_t2)), Some((o_f1, o_f2))) => {
                let pred = &self.0[o1..o2];
                let v1 = &v1[o_t1..o_t2];
                let v2 = &v2[o_f1..o_f2];
                pred.iter()
                    .zip(v1.iter().zip(v2.iter()))
                    .map(|(&p, (&t, &f))| if p > 0 { t } else { f })
                    .collect::<Vec<_>>()
            }
            _ => self
                .1
                .strided_index()
                .zip(l1.strided_index().zip(l2.strided_index()))
                .map(|(i_p, (v1_index, v2_index))| {
                    if self.0[i_p] > 0 {
                        v1[v1_index]
                    } else {
                        v2[v2_index]
                    }
                })
                .collect::<Vec<_>>(),
        };
        Ok(values)
    }
}

fn copy_strided_source_<T: Copy + std::fmt::Display>(
    source: &[T],
    destination: &mut [T],
    destination_offset: usize,
    source_layout: &Layout,
) {
    match source_layout.contiguous_offsets() {
        Some((o_destination1, o_destination2)) => {
            let elem_to_copy =
                (destination.len() - destination_offset).min(o_destination2 - o_destination1);
            destination[destination_offset..destination_offset + elem_to_copy]
                .copy_from_slice(&source[o_destination1..o_destination2])
        }
        None => {
            for (destination_index, source_index) in source_layout.strided_index().enumerate() {
                let destination_index = destination_index + destination_offset;
                if destination_index >= destination.len() {
                    break;
                }
                destination[destination_index] = source[source_index]
            }
        }
    }
}
