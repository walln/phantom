use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::device::{Device, NDArray};
use crate::index::StridedIndex;
use crate::operation::{BackpropOperation, BinaryOperations, Operation, UnaryOperations};
use crate::shape::Dim;
use crate::storage::Storage;
use crate::WithDType;
use crate::{DType, Error, Layout, Result, Shape};

/// Allow each tensor to be uniquely idenified. This makes it cheap to compute if
/// a given tensor is a reference to the same underlying data as another tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorID(usize);

impl TensorID {
    fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

pub struct Tensor_ {
    id: TensorID,
    storage: Arc<Storage>,
    layout: Layout,
    op: BackpropOperation,
    variable: bool,
    dtype: DType,
    device: Device,
}

/// Refcount tensors to make the construction of the graph cheap. Since tensors
/// are reference counted independently of the storage, the storage does not need
/// to be cloned when the operation does not modify the storage.
#[derive(Clone)]
pub struct Tensor(Arc<Tensor_>);

impl std::ops::Deref for Tensor {
    type Target = Tensor_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?},  {:?}]", &self.shape().dims(), self.device())
    }
}

macro_rules! binary_operation {
    ($fn_name:ident, $operation_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.binary_operation_shape_matches(rhs, stringify!($fn_name))?;
            let storage = self
                .storage
                .binary_operation::<crate::operation::$operation_name>(
                    &rhs.storage,
                    self.layout(),
                    rhs.layout(),
                )?;
            let op = BackpropOperation::new_binary(self, rhs, |a, b| {
                Operation::Binary(a, b, BinaryOperations::$operation_name)
            });
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    };
}

macro_rules! unary_operation {
    ($fn_name:ident, $operation_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            let shape = self.shape();
            let storage = self
                .storage
                .unary_operation::<crate::operation::$operation_name>(self.layout())?;
            let op = BackpropOperation::new_unary(self, |arg| {
                Operation::Unary(arg, UnaryOperations::$operation_name)
            });
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    };
}

impl Tensor {
    pub(crate) fn new_impl<A: NDArray>(
        array: A,
        shape: Shape,
        device: &Device,
        variable: bool,
    ) -> Result<Self> {
        let n: usize = shape.elem_count();
        let buffer_size: usize = array.shape()?.elem_count();
        if buffer_size != n {
            return Err(Error::ShapeMismatch { buffer_size, shape });
        }
        let storage = device.storage(array)?;
        Ok(from_storage(
            storage,
            shape,
            BackpropOperation::none(),
            variable,
        ))
    }

    /// Creates a new tensor from a slice of data.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::new(&[0f32, 1., 2., 3., 4., 5.], &Device::CPU)?;
    /// assert_eq!(tensor.shape(), &Shape::from(&[6]));
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn new<A: NDArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?;
        Self::new_impl(array, shape, device, false)
    }

    /// Creates a new variable tensor from a slice of data.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::var(&[0f32, 1., 2., 3., 4., 5.], &Device::CPU)?;
    /// assert_eq!(tensor.is_variable(), true);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn var<A: NDArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?;
        Self::new_impl(array, shape, device, true)
    }

    pub fn from_slice<S: Into<Shape>, D: crate::WithDType>(
        array: &[D],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::new_impl(array, shape.into(), device, false)
    }

    pub(crate) fn zeros_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        variable: bool,
    ) -> Result<Self> {
        if variable {
            let shape = shape.into();
            let storage = device.zeros(&shape, dtype)?;
            Ok(from_storage(
                storage,
                shape,
                BackpropOperation::none(),
                variable,
            ))
        } else {
            let storage = device.zeros(&crate::shape::SCALAR, dtype)?;
            from_storage(
                storage,
                crate::shape::SCALAR,
                BackpropOperation::none(),
                variable,
            )
            .broadcast_as(shape)
        }
    }

    /// Creates a new tensor of zeros with the given shape and data type.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::zeros(&[2, 2], phantom_core::DType::F32, &Device::CPU)?;
    /// assert_eq!(tensor.shape(), &Shape::from(&[2, 2]));
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::zeros_impl(shape, dtype, device, false)
    }

    /// Creates a new variable tensor of zeros in the same shape and data type as the input tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::var(&[[0f32, 1.], [2., 3.]], &Device::CPU)?;
    /// let zeros = tensor.zeros_like()?;
    /// assert_eq!(zeros.shape(), tensor.shape());
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn zeros_like(&self) -> Result<Self> {
        Tensor::zeros(self.shape(), self.dtype(), &self.device())
    }

    /// Creates a new variable tensor of zeros with the given shape and data type.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::zeros_var(&[2, 2], phantom_core::DType::F32, &Device::CPU)?;
    /// assert_eq!(tensor.is_variable(), true);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn zeros_var<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::zeros_impl(shape, dtype, device, true)
    }

    pub fn ones_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        variable: bool,
    ) -> Result<Self> {
        if variable {
            let shape = shape.into();
            let storage = device.ones(&shape, dtype)?;
            Ok(from_storage(
                storage,
                shape,
                BackpropOperation::none(),
                variable,
            ))
        } else {
            let storage = device.ones(&crate::shape::SCALAR, dtype)?;
            from_storage(
                storage,
                crate::shape::SCALAR,
                BackpropOperation::none(),
                variable,
            )
            .broadcast_as(shape)
        }
    }

    /// Creates a new tensor of ones with the given shape and data type.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::ones(&[2, 2], phantom_core::DType::F32, &Device::CPU)?;
    /// assert_eq!(tensor.shape(), &Shape::from(&[2, 2]));
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::ones_impl(shape, dtype, device, false)
    }

    /// Creates a new variable tensor of ones in the same shape and data type as the input tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::var(&[[0f32, 1.], [2., 3.]], &Device::CPU)?;
    /// let ones = tensor.ones_like()?;
    /// assert_eq!(ones.shape(), tensor.shape());
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn ones_like(&self) -> Result<Self> {
        Tensor::ones(self.shape(), self.dtype(), &self.device())
    }

    /// Creates a new variable tensor of ones with the given shape and data type.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::ones_var(&[2, 2], phantom_core::DType::F32, &Device::CPU)?;
    /// assert_eq!(tensor.is_variable(), true);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn ones_var<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::ones_impl(shape, dtype, device, true)
    }

    /// Converts the tensor to a scalar if the tensor is rank 0.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(0f32, &Device::CPU)?;
    /// assert_eq!(tensor.to_scalar::<f32>()?, 0f32);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn to_scalar<S: WithDType>(&self) -> Result<S> {
        if self.rank() != 0 {
            return Err(Error::UnexpectedRank {
                expected: 0,
                actual: self.rank(),
                shape: self.shape().clone(),
            });
        }

        let from_cpu = |cpu_storage: &crate::CPUStorage| {
            let data = S::cpu_storage_slice(cpu_storage)?;
            Ok::<_, Error>(data[self.layout().start_offset()])
        };

        match self.storage.as_ref() {
            Storage::CPU(storage) => from_cpu(storage),
            Storage::MPS(_) => todo!(),
        }
    }

    /// Returns the unique identifier for this tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[0f32], &Device::CPU)?;
    /// assert_eq!(tensor.id(), tensor.id());
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn id(&self) -> TensorID {
        self.id
    }

    /// Returns the data type of this tensor used on the storage backend.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[0f32], &Device::CPU)?;
    /// assert_eq!(tensor.dtype(), phantom_core::DType::F32);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Returns the device that this tensor is stored on.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[0f32], &Device::CPU)?;
    /// assert_eq!(tensor.device(), Device::CPU);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// TODO: Docs
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the shape of the tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::CPU)?;
    /// assert_eq!(tensor.shape(), &Shape::from(&[2, 2]));
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn shape(&self) -> &Shape {
        &self.layout.shape()
    }

    /// Returns the rank of the tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::CPU)?;
    /// assert_eq!(tensor.rank(), 2);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.layout.shape().rank()
    }

    /// Returns the dimension size for each axis of the tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::CPU)?;
    /// assert_eq!(tensor.dims(), &[2, 2]);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn dims(&self) -> &[usize] {
        self.layout.shape().dims()
    }

    /// Returns the total number of values in the tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::CPU)?;
    /// assert_eq!(tensor.elem_count(), 4);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn elem_count(&self) -> usize {
        self.layout.shape().elem_count()
    }

    /// Returns the element-wise stride of the tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::CPU)?;
    /// assert_eq!(tensor.stride(), &[2, 1]);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn stride(&self) -> &[usize] {
        &self.layout.stride()
    }

    /// Returns the operation that created this tensor.
    pub(crate) fn op(&self) -> &Option<Operation> {
        &self.op
    }

    /// Returns true if the computation graph should track this operation or
    /// if this is a variable or one of its dependencies is a variable.
    pub(crate) fn track_op(&self) -> bool {
        self.variable || self.op.is_some()
    }

    /// Returns true if the tensor is a variable that is tracked during backpropagation.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::var(&[[0f32, 1.], [2., 3.]], &Device::CPU)?;
    /// assert!(tensor.is_variable());
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn is_variable(&self) -> bool {
        self.variable
    }

    /// Creates an iterator that yields the offset position of each element in the buffer. Allowing
    /// the elements to be iterated over in lexicographic order.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32], [2.]], &Device::CPU)?;
    /// let mut iter = a.strided_index();
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), None);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn strided_index(&self) -> StridedIndex {
        self.layout.strided_index()
    }

    /// Returns true if the tensor is contiguous in memory.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// // Contigious example
    /// let a = Tensor::new(&[0f32], &Device::CPU)?;
    /// assert!(a.is_contiguous());
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn is_contiguous(&self) -> bool {
        let mut accumulated_stride = 1;
        for (&dim, &stride) in self.shape().dims().iter().zip(self.stride().iter()).rev() {
            if stride != accumulated_stride {
                return false;
            }
            accumulated_stride *= dim;
        }
        true
    }

    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let mut storage = self.device().zeros(shape, self.dtype())?;
            self.storage
                .copy_strided_source(&mut storage, 0, self.layout())?;
            let operation = BackpropOperation::new_unary(self, Operation::Copy);
            Ok(from_storage(storage, shape.clone(), operation, false))
        }
    }

    /// Returns the contents of the rank 1 tensor as a vector.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let a = Tensor::new(&[0f32, 1., 2., 3., 4., 5.], &Device::CPU)?;
    /// assert_eq!(a.to_vector_rank_one::<f32>()?, &[0., 1., 2., 3., 4., 5.]);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn to_vector_rank_one<S: WithDType>(&self) -> Result<Vec<S>> {
        if self.rank() != 1 {
            return Err(Error::UnexpectedRank {
                expected: 1,
                actual: self.rank(),
                shape: self.shape().clone(),
            });
        }
        match &self.storage.as_ref() {
            Storage::CPU(cpu_storage) => {
                let data = S::cpu_storage_slice(cpu_storage)?;
                Ok(self.strided_index().map(|i: usize| data[i]).collect())
            }
            Storage::MPS(_) => todo!(),
        }
    }

    /// Returns the contents of the rank 2 tensor as a vector of vectors in row-major order.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::CPU)?;
    /// assert_eq!(a.to_vector_rank_two::<f32>()?, &[[0., 1.], [2., 3.], [4., 5.]]);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn to_vector_rank_two<S: WithDType>(&self) -> Result<Vec<Vec<S>>> {
        let (dim_one, dim_two) = self.shape().rank_two()?;
        match &self.storage.as_ref() {
            Storage::CPU(storage) => {
                let data = S::cpu_storage_slice(storage)?;
                let mut rows = vec![];
                let mut index = self.strided_index();
                for _idx_row in 0..dim_one {
                    let row = (0..dim_two).map(|_| data[index.next().unwrap()]).collect();
                    rows.push(row)
                }
                assert!(index.next().is_none());
                Ok(rows)
            }
            Storage::MPS(_) => todo!(),
        }
    }

    /// Checks to see if the shapes of two tensors attempting a binary operation match
    /// and the operation can be performed, returning the shape if successful.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::CPU)?;
    /// let b = Tensor::new(&[[0f32, 1.], [2., 3.]], &Device::CPU)?;
    /// assert_eq!(a.binary_operation_shape_matches(&b, "add")?, a.shape());
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn binary_operation_shape_matches(
        &self,
        rhs: &Self,
        operation: &'static str,
    ) -> Result<&Shape> {
        let lhs = self.shape();
        let rhs = rhs.shape();

        if lhs != rhs {
            Err(Error::BinaryOperationShapeMismatch {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                op: operation,
            })
        } else {
            Ok(lhs)
        }
    }

    /// Operation that applies a multiplication and addition to the input tensor. This operation
    /// is equivalent to `mul * input + add` with the difference that the multiplication and
    /// addition is performed in-place on the input tensor. This operation is used to implement
    /// more optimized operations such as `relu`.
    ///
    /// NOTE: This operation casts the input values to the appropriate type so some rounding might
    /// be performed if operating in mixed precision.
    ///
    /// TODO: Add doctests
    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        let storage = self.storage.affine(self.layout(), mul, add)?;
        let operation =
            BackpropOperation::new_unary(self, |node| Operation::Affine { node, mul, add });
        Ok(from_storage(storage, self.shape(), operation, false))
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let tensor = Tensor_ {
            id: TensorID::new(),
            storage: self.storage.clone(),
            layout: self.layout.broadcast_as(shape)?,
            op: BackpropOperation::new_unary(self, Operation::Broadcast),
            variable: false,
            dtype: self.dtype,
            device: self.device,
        };

        Ok(Tensor(Arc::new(tensor)))
    }

    // Shorthand for broadcast_as
    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        self.broadcast_as(shape)
    }

    /// Returns a new tensor duplicating data from the original tensor. New dimensions are inserted
    /// on the left.
    pub fn broadcast_left<S: Into<Shape>>(&self, left_shape: S) -> Result<Self> {
        let left_shape = left_shape.into();
        let mut dims = left_shape.into_dims();
        dims.extend(self.dims());
        self.broadcast_as(dims)
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let storage = self.storage.to_dtype(&self.layout(), dtype)?;
            let operation = BackpropOperation::new_unary(self, Operation::ToDType);
            Ok(from_storage(storage, shape.clone(), operation, false))
        }
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let a_dims = self.dims();
        let b_dims = rhs.dims();

        if a_dims.len() < 2 || b_dims.len() != a_dims.len() {
            Err(Error::BinaryOperationShapeMismatch {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            }
            .backtrace())?
        }

        let dim = a_dims.len();

        let m = a_dims[dim - 2];
        let n = b_dims[dim - 1];
        let k = a_dims[dim - 1];
        let k2 = b_dims[dim - 2];

        let c_shape = Shape::from(&a_dims[..dim - 2]).extend(&[m, n]);
        let batching = a_dims[..dim - 2].iter().product();
        let batching_b = b_dims[..dim - 2].iter().product();
        if k != k2 || batching != batching_b {
            Err(Error::BinaryOperationShapeMismatch {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            }
            .backtrace())?
        }

        let storage = self.storage.matmul(
            &rhs.storage,
            (batching, m, n, k),
            self.layout(),
            rhs.layout(),
        )?;

        let operation = BackpropOperation::new_binary(self, rhs, Operation::Matmul);
        Ok(from_storage(storage, c_shape, operation, false))
    }

    /// Transpose the input tesnor by swapping the dimensions
    ///
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::CPU)?;
    /// let tensor = tensor.t()?;
    /// assert_eq!(tensor.to_vector_rank_two::<f32>()?, &[[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]);
    /// # Ok::<(), phantom_core::Error>(())
    /// ```
    pub fn t(&self) -> Result<Tensor> {
        let rank = self.rank();
        if rank < 2 {
            Err(Error::UnexpectedRank {
                expected: 2,
                actual: rank,
                shape: self.shape().clone(),
            }
            .backtrace())?
        }
        self.transpose(rank - 2, rank - 1)
    }

    /// Transpose the input tesnor by swapping the dimensions
    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Tensor> {
        let dim1 = dim1.to_index(self.shape(), "transpose")?;
        let dim2 = dim2.to_index(self.shape(), "transpose")?;
        let op = BackpropOperation::new_unary(self, |t| Operation::Transpose(t, dim1, dim2));
        let tensor = Tensor_ {
            id: TensorID::new(),
            storage: self.storage.clone(),
            layout: self.layout.transpose(dim1, dim2)?,
            op,
            variable: false,
            dtype: self.dtype,
            device: self.device.clone(),
        };
        Ok(Tensor(Arc::new(tensor)))
    }

    binary_operation!(add, Add);
    binary_operation!(sub, Sub);
    binary_operation!(mul, Mul);
    binary_operation!(div, Div);

    unary_operation!(sqr, Sqr);
    unary_operation!(sqrt, Sqrt);
    unary_operation!(neg, Neg);
}

/// Implement binary operations with operator shorthands.
macro_rules! binary_trait {
    ($trait:ident, $fn1:ident, $mul:expr, $add:expr) => {
        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<B> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Tensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<B> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Tensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<Result<B>> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                Tensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<Result<B>> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                Tensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl std::ops::$trait<f64> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: f64) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }

        impl std::ops::$trait<f64> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: f64) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }
    };
}

binary_trait!(Add, add, |_| 1., |v| v);
binary_trait!(Sub, sub, |_| 1., |v: f64| -v);
binary_trait!(Mul, mul, |v| v, |_| 0.);
binary_trait!(Div, div, |v| 1. / v, |_| 0.);

fn from_storage<S: Into<Shape>>(
    storage: Storage,
    shape: S,
    operation: BackpropOperation,
    variable: bool,
) -> Tensor {
    let dtype = storage.dtype();
    let device = storage.device();

    let tensor = Tensor_ {
        id: TensorID::new(),
        storage: Arc::new(storage),
        layout: Layout::contiguous(shape),
        op: operation,
        variable,
        dtype,
        device,
    };
    Tensor(Arc::new(tensor))
}
