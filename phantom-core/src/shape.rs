use crate::{Error, Result};

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

pub const SCALAR: Shape = Shape(vec![]);

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl From<usize> for Shape {
    fn from(dim_0: usize) -> Self {
        Self(vec![dim_0])
    }
}

impl From<(usize, usize)> for Shape {
    fn from((dim_0, dim_1): (usize, usize)) -> Self {
        Self(vec![dim_0, dim_1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from((dim_0, dim_1, dim_2): (usize, usize, usize)) -> Self {
        Self(vec![dim_0, dim_1, dim_2])
    }
}

impl From<&[usize; 1]> for Shape {
    fn from(dims: &[usize; 1]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize; 2]> for Shape {
    fn from(dims: &[usize; 2]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize; 3]> for Shape {
    fn from(dims: &[usize; 3]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self(shape.0.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

macro_rules! get_rank {
    ($fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty) => {
        pub fn $fn_name(&self) -> Result<$out_type> {
            if self.0.len() != $cnt {
                Err(Error::UnexpectedRank {
                    expected: $cnt,
                    actual: self.0.len(),
                    shape: self.clone(),
                })
            } else {
                Ok($dims(&self.0))
            }
        }
    };
}

impl Shape {
    pub fn from_dims(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn into_dims(self) -> Vec<usize> {
        self.0
    }

    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    get_rank!(rank_zero, 0, |_: &Vec<usize>| (), ());
    get_rank!(rank_one, 1, |dims: &[usize]| dims[0], usize);
    get_rank!(
        rank_two,
        2,
        |dims: &[usize]| (dims[0], dims[1]),
        (usize, usize)
    );
    get_rank!(
        rank_three,
        3,
        |dims: &[usize]| (dims[0], dims[1], dims[2]),
        (usize, usize, usize)
    );

    /// Stride over a contiguous n-dimensional array of this shape
    pub(crate) fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride: Vec<_> = self
            .0
            .iter()
            .rev()
            .scan(1, |product, u| {
                let inital_product = *product;
                *product *= u;
                Some(inital_product)
            })
            .collect();

        stride.reverse();
        stride
    }

    /// Returns true if the shape is contiguous with the given stride
    /// (i.e. no padding between dimensions) in a Row-Major order.
    pub fn is_contiguous(&self, stride: &[usize]) -> bool {
        if self.0.len() != stride.len() {
            return false;
        }

        let mut accumulator = 1;
        for (&stride, &dim) in stride.iter().zip(self.0.iter()).rev() {
            if stride != accumulator {
                return false;
            }
            accumulator *= dim;
        }
        true
    }

    pub fn extend(mut self, additional_dims: &[usize]) -> Self {
        self.0.extend(additional_dims);
        self
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.dims())
    }
}

pub trait Dim {
    fn to_index(&self, shape: &Shape, operation: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim >= shape.dims().len() {
            Err(Error::DimOutOfRange {
                shape: shape.clone(),
                dim: dim as i32,
                op,
            }
            .backtrace())?
        } else {
            Ok(dim)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride() {
        let shape = Shape::from(());
        assert_eq!(shape.stride_contiguous(), Vec::<usize>::new());
        let shape = Shape::from(42);
        assert_eq!(shape.stride_contiguous(), [1]);
        let shape = Shape::from((42, 1337));
        assert_eq!(shape.stride_contiguous(), [1337, 1]);
        let shape = Shape::from((299, 792, 458));
        assert_eq!(shape.stride_contiguous(), [458 * 792, 458, 1]);
    }
}
