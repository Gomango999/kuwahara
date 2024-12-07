#![allow(dead_code)]

use clap::Parser;
use convolve2d::*;
use image::{ImageReader, ImageResult};
// use indicatif::{ProgressBar, ProgressStyle};
use std::f64::consts::PI;
use std::path::PathBuf;

type RGBImage = DynamicMatrix<SubPixels<f64, 3>>;

mod consts {
    // Number of sectors in the filter
    pub static NUM_SECTORS: i32 = 8;
    // Standard deviation for Gaussian partial derivatives
    pub static PARTIAL_DERIVATIVE_STD: f64 = 1.0;
    // Gaussian spatial derivatives kernel size
    pub static PARTIAL_DERIVATIVE_KERNEL_SIZE: usize = 5;
    // Standard deviation of filter sector smoothing
    pub static FILTER_SMOOTHING_STD: f64 = 1.0;
    // Standard deviation for filter decay
    pub static FILTER_DECAY_STD: f64 = 3.0;
}

#[derive(Parser)]
#[command(about = "The anisotropic Kuwahara filter")]
#[command(
    long_about = "The anisotropic kuwahara filter tries to incorporate the
    directionality of the underlying image into the filter."
)]
pub struct Args {
    #[arg(short, long)]
    input: PathBuf,
}

impl Args {
    /// Generates a default output filename based on the input filename
    fn get_output(&self) -> PathBuf {
        let basename = self.input.parent().unwrap().to_str().unwrap();
        let basename = PathBuf::from(basename);

        let filename = format!(
            "{}_a.png",
            self.input.file_stem().unwrap().to_str().unwrap(),
        );
        let filename = PathBuf::from(filename);

        let filepath = basename.join(filename);
        filepath
    }
}

/// loads the RGB image specified in args
fn load_image(input_file: &PathBuf) -> ImageResult<RGBImage> {
    // Using convolve2d crate. Example usage here:
    // https://lib.rs/crates/convolve2d
    let img = ImageReader::open(input_file)?.decode()?.to_rgb8();
    let img: DynamicMatrix<SubPixels<u8, 3>> = img.into();
    let img: RGBImage = img.map_subpixels(|sp| sp as f64 / 255.0);
    Ok(img)
}

/// The 2D Gaussian
fn gaussian(x: f64, y: f64, std: f64) -> f64 {
    let factor = 1.0 / (2.0 * PI * std.powi(2));
    let exponent = -(x.powi(2) + y.powi(2)) / (2.0 * std.powi(2));
    factor * exponent.exp()
}
/// Returns the partial derivative of the 2D Gaussian WRT x
fn gaussian_derivative_x(x: f64, y: f64, std: f64) -> f64 {
    let factor = -x / std.powi(2);
    factor * gaussian(x, y, std)
}
/// Returns the partial derivative of the 2D Gaussian WRT y
fn gaussian_derivative_y(x: f64, y: f64, std: f64) -> f64 {
    gaussian_derivative_x(y, x, std)
}

/// Creates a kernel defined by f. f receives the (x,y) coordinate corresponding
/// to it's offset from the centre of the matrix. E.g. the middle of the kernel
/// will have value f(0,0), and the element to the right will have value f(1,0).
/// Panics if the kernel length is not odd
fn create_kernel_map(
    kernel_size: usize,
    f: fn(f64, f64) -> f64,
) -> DynamicMatrix<f64> {
    assert!(kernel_size % 2 == 1, "Kernel length must be odd");

    let half_kernel_size = (kernel_size / 2) as i32;
    let kernel_data: Vec<Vec<f64>> = (-half_kernel_size..=half_kernel_size)
        .map(|y| {
            (-half_kernel_size..=half_kernel_size)
                .map(|x| {
                    let y = y as f64;
                    let x = x as f64;
                    f(x, y)
                })
                .collect()
        })
        .collect();
    let kernel_data = kernel_data.iter().flatten().copied().collect();

    DynamicMatrix::new(kernel_size, kernel_size, kernel_data).unwrap()
}

/// Represents the structure tensor with values:
/// |e f|
/// |f g|
struct StructureTensor {
    e: f64,
    f: f64,
    g: f64,
}

impl StructureTensor {
    fn new(x_grad: f64, y_grad: f64) -> Self {
        StructureTensor {
            e: x_grad * x_grad,
            f: x_grad * y_grad,
            g: y_grad * y_grad,
        }
    }
}

/// For each pixel, computes the corresponding structure tensor
fn compute_structure_tensors(
    img: &RGBImage,
    kernel_size: usize,
) -> DynamicMatrix<StructureTensor> {
    // Calculate average gradient across all three channels
    let f_x =
        |x, y| gaussian_derivative_x(x, y, consts::PARTIAL_DERIVATIVE_STD);
    let f_y =
        |x, y| gaussian_derivative_y(x, y, consts::PARTIAL_DERIVATIVE_STD);
    let x_kernel = create_kernel_map(kernel_size, f_x);
    let y_kernel = create_kernel_map(kernel_size, f_y);

    let x_gradients = convolve2d(img, &x_kernel);
    let y_gradients = convolve2d(img, &y_kernel);

    let average_fn = |SubPixels([r, g, b])| (r + g + b) / 3.;
    let x_gradients = x_gradients.map(average_fn);
    let y_gradients = y_gradients.map(average_fn);

    // Compute structure tensors
    let (width, height, x_gradients_data) = x_gradients.into_parts();
    let (_, _, y_gradients_data) = y_gradients.into_parts();
    let structure_tensors_data: Vec<StructureTensor> = x_gradients_data
        .into_iter()
        .zip(y_gradients_data.into_iter())
        .map(|(x_grad, y_grad)| StructureTensor::new(x_grad, y_grad))
        .collect();

    DynamicMatrix::new(width, height, structure_tensors_data).unwrap()
}

pub fn run(args: &Args) -> ImageResult<()> {
    let img = load_image(&args.input)?;

    let _structure_tensors =
        compute_structure_tensors(&img, consts::PARTIAL_DERIVATIVE_KERNEL_SIZE);

    // smooth structure tensors

    // determine eigenvectors and values of each one

    // determine local orientation phi and anistropy

    // compute the scale vector S
    // compute the rotation vector R

    // compute the kernels at different orientations and save them.
    // Calculate K_0
    // Function for K_i
    // Function for w_i

    // for each pixel within a certain area, for each sector, compute
    // m_i and s_i^2

    // find the weighted sum of them and use that for the final output.
    println!("Done!");
    todo!()
}
