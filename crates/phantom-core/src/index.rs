/// A strided index acts as an iterator for the elements of an N-dimensional array stored in a
/// flat buffer using some potential strides. The iterator yields the offset position of each
/// element in the buffer.
#[derive(Debug)]
pub struct StridedIndex<'a> {
    next_index: Option<usize>,
    multi_index: Vec<usize>,
    dims: &'a [usize],
    stride: &'a [usize],
}

impl<'a> StridedIndex<'a> {
    pub(crate) fn new(dims: &'a [usize], stride: &'a [usize]) -> Self {
        let elem_count: usize = dims.iter().product();
        let next_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(0)
        };
        StridedIndex {
            next_index,
            multi_index: vec![0; dims.len()],
            dims,
            stride,
        }
    }
}

impl<'a> Iterator for StridedIndex<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = self.next_index?;
        let mut updated = false;
        for (multi_i, max_i) in self.multi_index.iter_mut().zip(self.dims.iter()).rev() {
            let next_i = *multi_i + 1;
            if next_i < *max_i {
                *multi_i = next_i;
                updated = true;
                break;
            } else {
                *multi_i = 0
            }
        }
        self.next_index = if updated {
            let next_storage_index = self
                .multi_index
                .iter()
                .zip(self.stride.iter())
                .map(|(&x, &y)| x * y)
                .sum();
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }
}
