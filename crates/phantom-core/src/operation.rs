use crate::Tensor;

pub enum Operation {
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Div(Tensor, Tensor),

    MatMul(Tensor, Tensor),

    Sqr(Tensor),
    Sqrt(Tensor),
    Neg(Tensor),

    Transpose(Tensor),

    Affine { node: Tensor, mul: f64, add: f64 },
}
