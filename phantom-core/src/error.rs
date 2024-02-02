use crate::device::DeviceLocation;
use crate::{DType, Layout, Shape};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    // TODO: consider renaming to unexpected dims despite the fact that rank is more accurate
    #[error("unexpected rank, expected: {expected}, actual: {actual}")]
    UnexpectedRank {
        expected: usize,
        actual: usize,
        shape: Shape,
    },

    #[error("{op}: dimension index {dim} out of range for shape {shape:?}")]
    DimOutOfRange {
        shape: Shape,
        dim: i32,
        op: &'static str,
    },

    #[error("unexpected dtype, expected: {expected:?}, actual: {actual:?}")]
    UnexpectedDType { expected: DType, actual: DType },

    #[error("unexpected device in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    BinaryOperationDeviceMismatch {
        lhs: DeviceLocation,
        rhs: DeviceLocation,
        op: &'static str,
    },

    #[error("unexpected dtype in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    BinaryOperationDTypeMismatch {
        lhs: DType,
        rhs: DType,
        op: &'static str,
    },

    #[error("unexpected shape in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    BinaryOperationShapeMismatch {
        lhs: Shape,
        rhs: Shape,
        op: &'static str,
    },

    #[error("cannot broadcast {source_shape:?} to {destination_shape:?}")]
    BroadcastIncompatibleShapes {
        source_shape: Shape,
        destination_shape: Shape,
    },

    #[error("Shape mismatch, got a buffer of size {buffer_size} which is incompatible with the shape {shape:?}")]
    ShapeMismatch { buffer_size: usize, shape: Shape },

    #[error("backward is not supported for {operation}")]
    BackwardUnsupported { operation: &'static str },

    #[error("unexpected stride in matmul, lhs: {lhs_layout:?}, rhs: {rhs_layout:?}, bmnk: {bmnk:?}, {msg}")]
    MatMulUnexpectedStride {
        lhs_layout: Layout,
        rhs_layout: Layout,
        bmnk: (usize, usize, usize, usize),
        msg: &'static str,
    },

    #[error("{op} invalid index {index} with vocab {vocab_size}")]
    InvalidIndex {
        op: &'static str,
        index: usize,
        vocab_size: usize,
    },

    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDTypeForOperation { dtype: DType, op: &'static str },

    #[error("{inner}\n{backtrace}")]
    WithBacktrace {
        inner: Box<Self>,
        backtrace: Box<std::backtrace::Backtrace>,
    },

    /// User generated error message, typically created via `bail!`.
    #[error("{0}")]
    Message(String),
}

impl Error {
    pub fn backtrace(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled
            | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace {
                inner: Box::new(self),
                backtrace: Box::new(backtrace),
            },
        }
    }

    pub fn message<T>(err: T) -> Self
    where
        T: std::error::Error + Send + Sync + 'static,
    {
        Self::Message(err.to_string()).backtrace()
    }
}

pub type Result<T> = std::result::Result<T, Error>;
