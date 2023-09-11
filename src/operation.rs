use crate::Tensor;

#[derive(Debug, Clone)]
pub(crate) enum Operation {
    // Binary Operations
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Div(Tensor, Tensor),

    // Unary Operations
    Sqr(Tensor),
    Sqrt(Tensor),
    Neg(Tensor),

    // Casting and complex ops
    Affine { node: Tensor, mul: f64, add: f64 },
    Broadcast(Tensor),

    ToDType(Tensor),
}

pub(crate) trait UnaryOperation {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn f32(v1: f32) -> f32;
    fn f64(v1: f64) -> f64;
    fn u32(v1: u32) -> u32;
}

pub(crate) trait BinaryOperation {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
    fn u32(v1: u32, v2: u32) -> u32;
}

pub(crate) struct Add;
pub(crate) struct Sub;
pub(crate) struct Mul;
pub(crate) struct Div;
pub(crate) struct Sqr;
pub(crate) struct Sqrt;
pub(crate) struct Neg;

macro_rules! bin_op {
    ($op:ident, $name: literal, $e: expr) => {
        impl BinaryOperation for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("b", $name);
            const V: Self = $op;

            fn f32(v1: f32, v2: f32) -> f32 {
                $e(v1, v2)
            }

            fn f64(v1: f64, v2: f64) -> f64 {
                $e(v1, v2)
            }

            fn u32(v1: u32, v2: u32) -> u32 {
                $e(v1, v2)
            }
        }
    };
}

bin_op!(Add, "add", |v1, v2| v1 + v2);
bin_op!(Sub, "sub", |v1, v2| v1 - v2);
bin_op!(Mul, "mul", |v1, v2| v1 * v2);
bin_op!(Div, "div", |v1, v2| v1 / v2);

macro_rules! unary_op {
    ($op: ident, $name: literal, $a: ident, $e: expr) => {
        impl UnaryOperation for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("u", $name);
            const V: Self = $op;

            fn f32($a: f32) -> f32 {
                $e
            }

            fn f64($a: f64) -> f64 {
                $e
            }

            fn u32($a: u32) -> u32 {
                todo!("no unary function for u32")
            }
        }
    };
}

unary_op!(Sqr, "sqr", a, a * a);
unary_op!(Sqrt, "sqrt", a, a.sqrt());
unary_op!(Neg, "neg", a, -a);
