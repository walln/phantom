use crate::{Error, Result, Shape, StridedIndex};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Layout {
    shape: Shape,
    /// Element-wise stride rather than byte-wise stride
    stride: Vec<usize>,
    start_offset: usize,
}

impl Layout {
    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        Self::contiguous_with_offset(shape, 0)
    }

    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, start_offset: usize) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    /// Returns the appropriate start and stop offset if the data is stored in a C
    /// contiguous (row major) way.
    pub fn contiguous_offsets(&self) -> Option<(usize, usize)> {
        if self.is_contiguous() {
            let start_o = self.start_offset;
            Some((start_o, start_o + self.shape.elem_count()))
        } else {
            None
        }
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    /// Returns true if the data is stored in a contiguous block of memory
    /// (i.e. no padding between dimensions) in a Row-Major order.
    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }

    pub(crate) fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        let dims = self.shape().dims();
        if dim >= dims.len() {
            Err(Error::UnexpectedRank {
                expected: dims.len(),
                actual: dim,
                shape: self.shape().clone(),
            })?
        }

        if start + length > dims[dim] {
            todo!("add a proper error: out of bounds for narrow {dim} {start} {length} {dims:?}")
        }

        let mut dims = dims.to_vec();
        dims[dim] = length;
        Ok(Self {
            shape: Shape::from(dims),
            stride: self.stride.clone(),
            start_offset: self.start_offset + self.stride[dim] * start,
        })
    }

    pub(crate) fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if rank <= dim1 || rank <= dim2 {
            return Err(Error::UnexpectedRank {
                expected: usize::max(dim1, dim2),
                actual: rank,
                shape: self.shape().clone(),
            });
        }
        let mut stride = self.stride().to_vec();
        let mut dims = self.shape().dims().to_vec();
        dims.swap(dim1, dim2);
        stride.swap(dim1, dim2);
        Ok(Self {
            shape: Shape::from(dims),
            stride,
            start_offset: self.start_offset,
        })
    }

    pub(crate) fn strided_index(&self) -> crate::StridedIndex {
        StridedIndex::new(self)
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.rank() < self.shape().rank() {
            Err(Error::BroadcastIncompatibleShapes {
                source_shape: self.shape().clone(),
                destination_shape: shape.clone(),
            })?
        }
        let added_dims = shape.rank() - self.shape().rank();
        let mut stride = vec![0; added_dims];
        for (&destination_dim, (&source_dim, &source_stride)) in shape.dims()[added_dims..]
            .iter()
            .zip(self.dims().iter().zip(self.stride()))
        {
            let s = if destination_dim == source_dim {
                source_stride
            } else if source_dim != 1 {
                return Err(Error::BroadcastIncompatibleShapes {
                    source_shape: self.shape().clone(),
                    destination_shape: shape,
                });
            } else {
                0
            };
            stride.push(s)
        }
        Ok(Self {
            shape,
            stride,
            start_offset: self.start_offset,
        })
    }
}
