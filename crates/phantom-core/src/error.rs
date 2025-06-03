use crate::{DType, Device, Shape};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("unexpected rank, expected: {expected}, actual: {actual}")]
    UnexpectedRank {
        expected: usize,
        actual: usize,
        shape: Shape,
    },

    #[error("unexpected dtype, expected: {expected:?}, actual: {actual:?}")]
    UnexpectedDType { expected: DType, actual: DType },

    #[error("unexpected device in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    BinaryOperationDeviceMismatch {
        lhs: Device,
        rhs: Device,
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
}

pub type Result<T> = std::result::Result<T, Error>;
