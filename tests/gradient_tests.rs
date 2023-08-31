use anyhow::{Context, Result};
use phantom::{Device, Tensor};

#[test]
fn simple_grad() -> Result<()> {
    let five = Tensor::new(&[5f32, 5., 5.], Device::CPU)?;

    let x = Tensor::var(&[3f32, 1., 4.], Device::CPU)?;
    let y = x.mul(&x)?.add(&x.mul(&five)?)?.add(&five)?;
    let gradients = y.backward()?;
    let gradient_x = gradients.get(&x.id()).context("x has no gradient")?;

    assert_eq!(x.to_vector_rank_one::<f32>()?, [3., 1., 4.]);
    assert_eq!(y.to_vector_rank_one::<f32>()?, [29., 11., 41.]);
    assert_eq!(gradient_x.to_vector_rank_one::<f32>()?, [11., 7., 13.]);

    let x = Tensor::var(&[4f32, 2., 8.], Device::CPU)?;
    let y = x.mul(&x)?.add(&x.mul(&five)?)?.add(&five)?;
    let gradients = y.backward()?;
    let gradient_x = gradients.get(&x.id()).context("x has no gradient")?;

    assert_eq!(x.to_vector_rank_one::<f32>()?, [4., 2., 8.]);
    assert_eq!(y.to_vector_rank_one::<f32>()?, [41., 19., 109.]);
    assert_eq!(gradient_x.to_vector_rank_one::<f32>()?, [13., 9., 21.]);

    Ok(())
}

#[test]
fn simple_grad_constants() -> Result<()> {
    let x = Tensor::var(&[3f32, 1., 4.], Device::CPU)?;
    let y = (((&x * &x)? + &x * 5f64)? + 4f64)?;
    let gradients = y.backward()?;
    let gradient_x = gradients.get(&x.id()).context("x has no gradient")?;

    assert_eq!(x.to_vector_rank_one::<f32>()?, [3., 1., 4.]);
    assert_eq!(y.to_vector_rank_one::<f32>()?, [28., 10., 40.]);
    assert_eq!(gradient_x.to_vector_rank_one::<f32>()?, [11., 7., 13.]);
    Ok(())
}
