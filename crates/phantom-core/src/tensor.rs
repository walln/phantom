use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::device::{Device, NDArray};
use crate::index::StridedIndex;
use crate::storage::Storage;
use crate::WithDType;
use crate::{DType, Operation, Shape};
use crate::dtype::DTypeError;
use crate::shape::ShapeError;
use crate::storage::StorageError;

#[derive(Debug)]
pub enum TensorError {
    Shape(ShapeError),
    DType(DTypeError),
    Storage(StorageError),
}

impl From<ShapeError> for TensorError {
    fn from(e: ShapeError) -> Self {
        TensorError::Shape(e)
    }
}

impl From<DTypeError> for TensorError {
    fn from(e: DTypeError) -> Self {
        TensorError::DType(e)
    }
}

impl From<StorageError> for TensorError {
    fn from(e: StorageError) -> Self {
        TensorError::Storage(e)
    }
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::Shape(e) => write!(f, "{e}"),
            TensorError::DType(e) => write!(f, "{e}"),
            TensorError::Storage(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TensorError {}

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
    storage: Storage,
    shape: Shape,
    /// Element-wise stride rather than byte-wise stride
    stride: Vec<usize>,
    op: Option<Operation>,
    variable: bool,
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
    ($fn_name:ident, $operation_name:ident, $storage_operation:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> std::result::Result<Self, TensorError> {
            let shape = self.binary_operation_shape_matches(rhs, stringify!($fn_name))?;
            let storage = self.storage.$storage_operation(
                &rhs.storage,
                shape,
                self.stride(),
                rhs.stride(),
            )?;
            let t = Tensor_ {
                id: TensorID::new(),
                storage,
                shape: shape.clone(),
                stride: shape.stride_contiguous(),
                op: Some(Operation::$operation_name(self.clone(), rhs.clone())),
                variable: false,
            };
            Ok(Self(Arc::new(t)))
        }
    };
}

macro_rules! unary_operation {
    ($fn_name:ident, $operation_name:ident, $storage_operation:ident) => {
        pub fn $fn_name(&self) -> std::result::Result<Self, TensorError> {
            let shape = self.shape();
            let storage = self.storage.$storage_operation(shape, self.stride())?;
            let t = Tensor_ {
                id: TensorID::new(),
                storage,
                shape: shape.clone(),
                stride: shape.stride_contiguous(),
                op: Some(Operation::$operation_name(self.clone())),
                variable: false,
            };
            Ok(Self(Arc::new(t)))
        }
    };
}

impl Tensor {
    pub(crate) fn new_impl<A: NDArray>(array: A, device: Device, variable: bool) -> std::result::Result<Self, TensorError> {
        let shape: Shape = array.shape()?;
        let storage: Storage = device.tensor(array);
        let stride: Vec<usize> = shape.stride_contiguous();
        let id: TensorID = TensorID::new();

        let t: Tensor_ = Tensor_ {
            id,
            storage,
            shape,
            stride,
            op: None,
            variable,
        };
        Ok(Self(Arc::new(t)))
    }

    /// Creates a new tensor from a slice of data.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::new(&[0f32, 1., 2., 3., 4., 5.], Device::CPU)?;
    /// assert_eq!(tensor.shape(), &Shape::from(&[6]));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new<A: NDArray>(array: A, device: Device) -> std::result::Result<Self, TensorError> {
        Self::new_impl(array, device, false)
    }

    /// Creates a new variable tensor from a slice of data.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::var(&[0f32, 1., 2., 3., 4., 5.], Device::CPU)?;
    /// assert_eq!(tensor.variable(), true);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn var<A: NDArray>(array: A, device: Device) -> std::result::Result<Self, TensorError> {
        Self::new_impl(array, device, true)
    }

    pub(crate) fn zeros_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: Device,
        variable: bool,
    ) -> Self {
        let shape = shape.into();
        let storage = device.zeros(&shape, dtype);
        let stride = shape.stride_contiguous();
        let id: TensorID = TensorID::new();

        let t = Tensor_ {
            id,
            storage,
            shape,
            stride,
            op: None,
            variable,
        };

        Tensor(Arc::new(t))
    }

    /// Creates a new tensor of zeros with the given shape and data type.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::zeros(&[2, 2], phantom_core::DType::F32, Device::CPU);
    /// assert_eq!(tensor.shape(), &Shape::from(&[2, 2]));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Self {
        Self::zeros_impl(shape, dtype, device, false)
    }

    /// Creates a new variable tensor of zeros in the same shape and data type as the input tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::var(&[[0f32, 1.], [2., 3.]], Device::CPU)?;
    /// let zeros = tensor.zeros_like();
    /// assert_eq!(zeros.shape(), tensor.shape());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn zeros_like(&self) -> Self {
        Tensor::zeros(self.shape(), self.dtype(), self.device())
    }

    /// Creates a new variable tensor of zeros with the given shape and data type.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::zeros_var(&[2, 2], phantom_core::DType::F32, Device::CPU);
    /// assert_eq!(tensor.variable(), true);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn zeros_var<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Self {
        Self::zeros_impl(shape, dtype, device, true)
    }

    pub fn ones_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: Device,
        variable: bool,
    ) -> Self {
        let shape = shape.into();
        let storage = device.ones(&shape, dtype);
        let stride = shape.stride_contiguous();
        let id: TensorID = TensorID::new();

        let t = Tensor_ {
            id,
            storage,
            shape,
            stride,
            op: None,
            variable,
        };

        Tensor(Arc::new(t))
    }

    /// Creates a new tensor of ones with the given shape and data type.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::ones(&[2, 2], phantom_core::DType::F32, Device::CPU);
    /// assert_eq!(tensor.shape(), &Shape::from(&[2, 2]));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Self {
        Self::ones_impl(shape, dtype, device, false)
    }

    /// Creates a new variable tensor of ones in the same shape and data type as the input tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::var(&[[0f32, 1.], [2., 3.]], Device::CPU)?;
    /// let ones = tensor.ones_like();
    /// assert_eq!(ones.shape(), tensor.shape());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn ones_like(&self) -> Self {
        Tensor::ones(self.shape(), self.dtype(), self.device())
    }

    /// Creates a new variable tensor of ones with the given shape and data type.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::ones_var(&[2, 2], phantom_core::DType::F32, Device::CPU);
    /// assert_eq!(tensor.variable(), true);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn ones_var<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Self {
        Self::ones_impl(shape, dtype, device, true)
    }

    /// Converts the tensor to a scalar if the tensor is rank 0.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(0f32, Device::CPU)?;
    /// assert_eq!(tensor.to_scalar::<f32>()?, 0f32);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_scalar<S: WithDType>(&self) -> std::result::Result<S, TensorError> {
        if self.rank() != 0 {
            return Err(TensorError::Shape(ShapeError::UnexpectedRank {
                expected: 0,
                actual: self.rank(),
                shape: self.0.shape.clone(),
            }));
        }
        match &self.0.storage {
            Storage::CPU(storage) => {
                let data = S::storage_slice(storage).map_err(TensorError::from)?;
                Ok(data[0])
            }
        }
    }

    /// Returns the unique identifier for this tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[0f32], Device::CPU)?;
    /// assert_eq!(tensor.id(), tensor.id());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn id(&self) -> TensorID {
        self.id
    }

    /// Returns the data type of this tensor used on the storage backend.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[0f32], Device::CPU)?;
    /// assert_eq!(tensor.dtype(), phantom_core::DType::F32);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Returns the device that this tensor is stored on.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[0f32], Device::CPU)?;
    /// assert_eq!(tensor.device(), Device::CPU);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Returns the shape of the tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device, Shape};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.]], Device::CPU)?;
    /// assert_eq!(tensor.shape(), &Shape::from(&[2, 2]));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the rank of the tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.]], Device::CPU)?;
    /// assert_eq!(tensor.rank(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the dimension size for each axis of the tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.]], Device::CPU)?;
    /// assert_eq!(tensor.dims(), &[2, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Returns the total number of values in the tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.]], Device::CPU)?;
    /// assert_eq!(tensor.elem_count(), 4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    /// Returns the element-wise stride of the tensor.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.]], Device::CPU)?;
    /// assert_eq!(tensor.stride(), &[2, 1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    /// Returns the operation that created this tensor.
    pub(crate) fn op(&self) -> &Option<Operation> {
        &self.op
    }

    /// Returns true if the tensor is a variable that is tracked during backpropagation.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let tensor = Tensor::var(&[[0f32, 1.], [2., 3.]], Device::CPU)?;
    /// assert!(tensor.variable());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn variable(&self) -> bool {
        self.variable
    }

    /// Creates an iterator that yields the offset position of each element in the buffer. Allowing
    /// the elements to be iterated over in lexicographic order.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32], [2.]], Device::CPU)?;
    /// let mut iter = a.strided_index();
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn strided_index(&self) -> StridedIndex {
        StridedIndex::new(self.dims(), self.stride())
    }

    /// Returns true if the tensor is contiguous in memory.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// // Contigious example
    /// let a = Tensor::new(&[0f32], Device::CPU)?;
    /// assert!(a.contiguous());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn contiguous(&self) -> bool {
        let mut accumulated_stride = 1;
        for (&dim, &stride) in self.shape.dims().iter().zip(self.stride.iter()).rev() {
            if stride != accumulated_stride {
                return false;
            }
            accumulated_stride *= dim;
        }
        true
    }

    /// Returns the contents of the rank 1 tensor as a vector.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let a = Tensor::new(&[0f32, 1., 2., 3., 4., 5.], Device::CPU)?;
    /// assert_eq!(a.to_vector_rank_one::<f32>()?, &[0., 1., 2., 3., 4., 5.]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_vector_rank_one<S: WithDType>(&self) -> std::result::Result<Vec<S>, TensorError> {
        if self.rank() != 1 {
            return Err(TensorError::Shape(ShapeError::UnexpectedRank {
                expected: 1,
                actual: self.rank(),
                shape: self.shape().clone(),
            }));
        }
        match &self.storage {
            Storage::CPU(cpu_storage) => {
                let data = S::storage_slice(cpu_storage).map_err(TensorError::from)?;
                Ok(self.strided_index().map(|i: usize| data[i]).collect())
            }
        }
    }

    /// Returns the contents of the rank 2 tensor as a vector of vectors in row-major order.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], Device::CPU)?;
    /// assert_eq!(a.to_vector_rank_two::<f32>()?, &[[0., 1.], [2., 3.], [4., 5.]]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_vector_rank_two<S: WithDType>(&self) -> std::result::Result<Vec<Vec<S>>, TensorError> {
        let (dim_one, dim_two) = self.shape().rank_two()?;
        match &self.storage {
            Storage::CPU(storage) => {
                let data = S::storage_slice(storage).map_err(TensorError::from)?;
                let mut rows = vec![];
                let mut index = self.strided_index();
                for _idx_row in 0..dim_one {
                    let row = (0..dim_two).map(|_| data[index.next().unwrap()]).collect();
                    rows.push(row)
                }
                assert!(index.next().is_none());
                Ok(rows)
            }
        }
    }

    /// Checks to see if the shapes of two tensors attempting a binary operation match
    /// and the operation can be performed, returning the shape if successful.
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.]], Device::CPU)?;
    /// let b = Tensor::new(&[[0f32, 1.], [2., 3.]], Device::CPU)?;
    /// assert_eq!(a.binary_operation_shape_matches(&b, "add")?, a.shape());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn binary_operation_shape_matches(
        &self,
        rhs: &Self,
        operation: &'static str,
    ) -> std::result::Result<&Shape, TensorError> {
        let lhs = self.shape();
        let rhs = rhs.shape();

        if lhs != rhs {
            Err(TensorError::Storage(StorageError::BinaryOperationShapeMismatch {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                op: operation,
            }))
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
    /// ```rust
    /// use phantom_core::{Tensor, Device};
    /// let a = Tensor::new(&[[0f32, 1.], [2., 3.]], Device::CPU)?;
    /// let a = a.affine(4., -2.)?;
    /// assert_eq!(a.to_vector_rank_two::<f32>()?, &[[-2.0, 2.0], [6.0, 10.0]]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn affine(&self, mul: f64, add: f64) -> std::result::Result<Self, TensorError> {
        let shape = self.shape();
        let storage = self
            .storage
            .affine(self.shape(), self.stride(), mul, add)
            .map_err(TensorError::from)?;

        let t = Tensor_ {
            id: TensorID::new(),
            storage,
            shape: shape.clone(),
            stride: shape.stride_contiguous(),
            op: Some(Operation::Affine {
                node: self.clone(),
                mul,
                add,
            }),
            variable: false,
        };
        Ok(Self(Arc::new(t)))
    }

    binary_operation!(add, Add, add);
    binary_operation!(sub, Sub, sub);
    binary_operation!(mul, Mul, mul);
    binary_operation!(div, Div, div);

    unary_operation!(sqr, Sqr, sqr);
    unary_operation!(sqrt, Sqrt, sqrt);
    unary_operation!(neg, Neg, neg);

    pub fn transpose(&self) -> std::result::Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::Shape(ShapeError::UnexpectedRank {
                expected: 2,
                actual: self.rank(),
                shape: self.shape().clone(),
            }));
        }
        let shape = self.shape();
        let storage = self
            .storage
            .transpose(shape, self.stride())
            .map_err(TensorError::from)?;
        let (rows, cols) = shape.rank_two()?;
        let new_shape = Shape::from((cols, rows));
        let t = Tensor_ {
            id: TensorID::new(),
            storage,
            shape: new_shape.clone(),
            stride: new_shape.stride_contiguous(),
            op: Some(Operation::Transpose(self.clone())),
            variable: false,
        };
        Ok(Self(Arc::new(t)))
    }

    pub fn matmul(&self, rhs: &Self) -> std::result::Result<Self, TensorError> {
        if self.rank() != 2 || rhs.rank() != 2 {
            return Err(TensorError::Shape(ShapeError::UnexpectedRank {
                expected: 2,
                actual: if self.rank() != 2 { self.rank() } else { rhs.rank() },
                shape: if self.rank() != 2 { self.shape().clone() } else { rhs.shape().clone() },
            }));
        }
        let (m, k) = self.shape().rank_two()?;
        let (k_rhs, n) = rhs.shape().rank_two()?;
        if k != k_rhs {
            return Err(TensorError::Storage(StorageError::BinaryOperationShapeMismatch {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            }));
        }
        let storage = self
            .storage
            .matmul(
            &rhs.storage,
            (m, k),
            self.stride(),
            (k_rhs, n),
            rhs.stride(),
        )
        .map_err(TensorError::from)?;
        let shape = Shape::from((m, n));
        let t = Tensor_ {
            id: TensorID::new(),
            storage,
            shape: shape.clone(),
            stride: shape.stride_contiguous(),
            op: Some(Operation::MatMul(self.clone(), rhs.clone())),
            variable: false,
        };
        Ok(Self(Arc::new(t)))
    }
}

/// Implement binary operations with operator shorthands.
macro_rules! binary_trait {
    ($trait:ident, $fn1:ident, $mul:expr, $add:expr) => {
        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<B> for Tensor {
            type Output = std::result::Result<Tensor, TensorError>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Tensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<B> for &Tensor {
            type Output = std::result::Result<Tensor, TensorError>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Tensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<std::result::Result<B, TensorError>> for Tensor {
            type Output = std::result::Result<Tensor, TensorError>;

            fn $fn1(self, rhs: std::result::Result<B, TensorError>) -> Self::Output {
                Tensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<std::result::Result<B, TensorError>> for &Tensor {
            type Output = std::result::Result<Tensor, TensorError>;

            fn $fn1(self, rhs: std::result::Result<B, TensorError>) -> Self::Output {
                Tensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl std::ops::$trait<f64> for Tensor {
            type Output = std::result::Result<Tensor, TensorError>;

            fn $fn1(self, rhs: f64) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }

        impl std::ops::$trait<f64> for &Tensor {
            type Output = std::result::Result<Tensor, TensorError>;

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
