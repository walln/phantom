#[derive(Debug)]
pub enum ShapeError {
    UnexpectedRank {
        expected: usize,
        actual: usize,
        shape: Shape,
    },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::UnexpectedRank {
                expected, actual, ..
            } => {
                write!(
                    f,
                    "unexpected rank, expected: {}, actual: {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for ShapeError {}

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(pub(crate) Vec<usize>);

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

macro_rules! get_rank {
    ($fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty) => {
        pub fn $fn_name(&self) -> std::result::Result<$out_type, ShapeError> {
            if self.0.len() != $cnt {
                Err(ShapeError::UnexpectedRank {
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
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.dims())
    }
}
