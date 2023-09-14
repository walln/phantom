use crate::layout::Layout;

/// A strided index acts as an iterator for the elements of an N-dimensional array stored in a
/// flat buffer using some potential strides. The iterator yields the offset position of each
/// element in the buffer.
#[derive(Debug)]
pub struct StridedIndex<'a> {
    next_index: Option<usize>,
    multi_index: Vec<usize>,
    layout: &'a Layout,
}

impl<'a> StridedIndex<'a> {
    pub(crate) fn new(layout: &'a Layout) -> Self {
        let dims = layout.dims();
        let elem_count: usize = dims.iter().product();
        let next_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(layout.start_offset())
        };
        StridedIndex {
            next_index,
            multi_index: vec![0; dims.len()],
            layout,
        }
    }
}

impl<'a> Iterator for StridedIndex<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = match self.next_index {
            None => return None,
            Some(storage_index) => storage_index,
        };
        let mut updated = false;

        for (multi_i, max_i) in self
            .multi_index
            .iter_mut()
            .zip(self.layout.dims().iter())
            .rev()
        {
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
                .zip(self.layout.stride().iter())
                .map(|(&x, &y)| x * y)
                .sum::<usize>()
                + self.layout.start_offset();
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }
}
