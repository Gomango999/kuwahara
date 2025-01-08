#![allow(dead_code)]

use clap::Parser;
use image::{GrayImage, ImageReader, ImageResult, RgbImage};
use ndarray::{array, stack, Array2, Array3, Axis, Zip};
use ndarray_conv::{ConvExt, ConvMode, PaddingMode};
use std::f64::consts::PI;
use std::path::PathBuf;

mod consts {
    // Number of sectors in the filter
    pub static NUM_SECTORS: i32 = 8;
    // Standard deviation for Gaussian partial derivatives
    pub static PARTIAL_DERIVATIVE_STD: f64 = 1.0;
    // Gaussian spatial derivatives kernel size
    pub static PARTIAL_DERIVATIVE_KERNEL_SIZE: usize = 5;
    // Standard deviation for smoothing structure tensors
    pub static TENSOR_SMOOTHING_STD: f64 = 2.0;
    // Standard deviation of filter sector smoothing
    pub static FILTER_SMOOTHING_STD: f64 = 1.0;
    // Standard deviation for filter decay
    pub static FILTER_DECAY_STD: f64 = 3.0;
}

#[derive(Parser)]
#[command(about = "The anisotropic Kuwahara filter")]
#[command(long_about = "The anisotropic kuwahara filter tries to incorporate the
    directionality of the underlying image into the filter.")]
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

// Turns an image into an ndarray with dimensions (3, height, width)
fn image_to_ndarray(img: RgbImage) -> Array3<f64> {
    let (width, height) = img.dimensions();
    let dimensions = (height as usize, width as usize, 3);
    let data = img
        .into_raw()
        .into_iter()
        .map(|x| x as f64 / 255.0)
        .collect();
    // Will not panic because the dimensions are correct
    let image = Array3::from_shape_vec(dimensions, data).unwrap();
    image.permuted_axes([2, 0, 1])
}

fn ndarray_to_rgbimage(arr: Array3<f64>) -> RgbImage {
    let arr = arr.permuted_axes([1, 2, 0]);
    let (height, width, _) = arr.dim();
    let data = arr.iter().map(|x| (x * 255.0) as u8).collect::<Vec<u8>>();
    // Will not return none because the dimensions are correct
    RgbImage::from_raw(width as u32, height as u32, data).unwrap()
}

/// Assumes the input array is a grayscale image with values in the range [0, 1]
fn ndarray_to_grayimage(arr: Array2<f64>) -> GrayImage {
    let (height, width) = arr.dim();
    let data = arr.iter().map(|x| (x * 255.0) as u8).collect::<Vec<u8>>();
    // Will not return none because the dimensions are correct
    GrayImage::from_raw(width as u32, height as u32, data).unwrap()
}

fn load_image(input_file: &PathBuf) -> ImageResult<Array3<f64>> {
    let img = ImageReader::open(input_file)?.decode()?.to_rgb8();
    let img = image_to_ndarray(img);
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

/// Creates a 2D odd-side-length kernel defined by a function f(x,y), where (x,y)
/// is (0,0) at the centre of the kernel.
/// Panics if the kernel length is not odd
fn array2_from_fn(kernel_size: usize, f: impl Fn(f64, f64) -> f64) -> Array2<f64> {
    assert!(kernel_size % 2 == 1, "Kernel length must be odd");

    let half_kernel_size = (kernel_size / 2) as i32;
    Array2::from_shape_fn((kernel_size, kernel_size), |(y, x)| {
        let x = x as i32 - half_kernel_size;
        let y = y as i32 - half_kernel_size;
        f(x as f64, y as f64)
    })
}

fn gaussian_kernel(size: usize, std: f64) -> Array2<f64> {
    let f = |x, y| gaussian(x, y, std);
    array2_from_fn(size, f)
}

/// Represents the structure tensor with values:
/// |e f|
/// |f g|
#[derive(Copy, Clone, Default, Debug, PartialEq)]
struct StructureTensor {
    e: f64,
    f: f64,
    g: f64,
}

impl StructureTensor {
    fn new(e: f64, f: f64, g: f64) -> Self {
        StructureTensor { e, f, g }
    }

    fn approx_zero(&self) -> bool {
        const THRESHOLD: f64 = 1e-9;
        self.e.abs() < THRESHOLD && self.f.abs() < THRESHOLD && self.g.abs() < THRESHOLD
    }
}
struct Gradients {
    x: Array3<f64>,
    y: Array3<f64>,
}

fn compute_gradients(img: &Array3<f64>, kernel_size: usize) -> Gradients {
    /// Normalise kernels so that the sum of absolute values is equal to 2.
    /// The sum is arbritary, but we simply want to avoid floating point error
    /// by using kernels that have values that are too small.
    fn normalise_kernel(arr: Array2<f64>) -> Array2<f64> {
        let sum = arr.map(|x| x.abs()).sum();
        arr / sum
    }

    let f_x = |x, y| gaussian_derivative_x(x, y, consts::PARTIAL_DERIVATIVE_STD);
    let f_y = |x, y| gaussian_derivative_y(x, y, consts::PARTIAL_DERIVATIVE_STD);

    let x_kernel = normalise_kernel(array2_from_fn(kernel_size, f_x));
    let y_kernel = normalise_kernel(array2_from_fn(kernel_size, f_y));

    /// Applies the kernel convolution to each of the three RGB channels
    /// independently
    fn apply_convolution_on_channels(img: &Array3<f64>, kernel: &Array2<f64>) -> Array3<f64> {
        let gradients = img
            .clone()
            .axis_iter(Axis(0))
            .map(|channel| {
                channel
                    .conv(kernel, ConvMode::Same, PaddingMode::Replicate)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let gradient_views = gradients.iter().map(|x| x.view()).collect::<Vec<_>>();
        stack(Axis(0), &gradient_views).unwrap()
    }

    let x_gradients = apply_convolution_on_channels(img, &x_kernel);
    let y_gradients = apply_convolution_on_channels(img, &y_kernel);

    Gradients {
        x: x_gradients,
        y: y_gradients,
    }
}

/// For each pixel, computes the corresponding structure tensor
fn compute_structure_tensors(gradients: Gradients) -> Array2<StructureTensor> {
    // Compute a structure tensor for each pixel

    #[inline]
    fn dot_product(a1: &Array3<f64>, a2: &Array3<f64>) -> Array2<f64> {
        (a1 * a2).sum_axis(Axis(0))
    }

    let fxx = dot_product(&gradients.x, &gradients.x);
    let fxy = dot_product(&gradients.x, &gradients.y);
    let fyy = dot_product(&gradients.y, &gradients.y);

    Zip::from(&fxx)
        .and(&fxy)
        .and(&fyy)
        .map_collect(|&e, &f, &g| StructureTensor::new(e, f, g))
}

fn smooth_structure_tensors(
    structure_tensors: &Array2<StructureTensor>,
) -> Array2<StructureTensor> {
    // We smooth the structure tensors by piecewise smoothing each of it's three
    // fields with the structure tensors around it, using a Gaussian kernel.

    let es = structure_tensors.mapv(|tensor| tensor.e);
    let fs = structure_tensors.mapv(|tensor| tensor.f);
    let gs = structure_tensors.mapv(|tensor| tensor.g);

    let kernel_size = consts::TENSOR_SMOOTHING_STD as usize * 3 * 2 + 1;
    let smoothing_kernel = gaussian_kernel(kernel_size, consts::TENSOR_SMOOTHING_STD);

    let es = es
        .conv(&smoothing_kernel, ConvMode::Same, PaddingMode::Replicate)
        .unwrap();
    let fs = fs
        .conv(&smoothing_kernel, ConvMode::Same, PaddingMode::Replicate)
        .unwrap();
    let gs = gs
        .conv(&smoothing_kernel, ConvMode::Same, PaddingMode::Replicate)
        .unwrap();

    let structure_tensors = Zip::from(&es)
        .and(&fs)
        .and(&gs)
        .map_collect(|&e, &f, &g| StructureTensor::new(e, f, g));
    structure_tensors
}

#[derive(Clone)]
struct Anisotropy {
    anisotropy: f64,
    angle: f64, // bounded by [-PI, PI]
    transform: Array2<f64>,
    rotation: Array2<f64>,
}

/// Finds the eigenvalue of a structure tensor. The largest eigenvalue is given first.
/// TODO: Move this as a method of StructureTensor
fn get_eigenvalues(tensor: &StructureTensor) -> (f64, f64) {
    let determinant = (tensor.e - tensor.g).powi(2) + 4.0 * tensor.f.powi(2);
    let eigenvalue1 = (tensor.e + tensor.g + determinant.sqrt()) / 2.0;
    let eigenvalue2 = (tensor.e + tensor.g - determinant.sqrt()) / 2.0;
    (eigenvalue1, eigenvalue2)
}

fn calc_anisotropy(tensor: &StructureTensor) -> Anisotropy {
    if tensor.approx_zero() {
        // This area is isotropic
        let identity = array![[1.0, 0.0], [0.0, 1.0]];
        return Anisotropy {
            anisotropy: 0.0,
            angle: 0.0,
            transform: identity.clone(),
            rotation: identity.clone(),
        };
    } else {
        let (eigenvalue1, eigenvalue2) = get_eigenvalues(&tensor);

        let t = array![eigenvalue1 - tensor.e, -tensor.f];
        let angle = t[1].atan2(t[0]);

        let anisotropy = (eigenvalue1 - eigenvalue2) / (eigenvalue1 + eigenvalue2);

        const ALPHA: f64 = 1.0;
        let transform = array![
            [ALPHA / (ALPHA + anisotropy), 0.],
            [0., (ALPHA + anisotropy) / ALPHA]
        ];

        let rotation = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];

        Anisotropy {
            anisotropy,
            angle,
            transform,
            rotation,
        }
    }
}

// /// Weighting function for the disc.
// fn disc_space_weighting(i: usize) -> Array2<f64> {
//     // Calculate the charateristic function
//     const SIZE: usize = 27;
//     let mut charateristic = Array2::<f64>::zeros((SIZE, SIZE));

//     let half_size: isize = (SIZE as f64 / 2.0).floor() as isize;
//     for x in 0..SIZE {
//         for y in 0..SIZE {
//             let dx = (x as isize - half_size) as f64;
//             let dy = (y as isize - half_size) as f64;

//             let inside_circle =
//                 dx.powi(2) + dy.powi(2) <= (half_size as f64).powi(2);

//             let angle = dy.atan2(dx);
//             let lower =
//                 ((2.0 * i as f64 - 1.0) * PI) / consts::NUM_SECTORS as f64;
//             let upper =
//                 ((2.0 * i as f64 + 1.0) * PI) / consts::NUM_SECTORS as f64;
//             let inside_segment = lower < angle && angle <= upper;

//             charateristic[(x, y)] = if inside_circle && inside_segment {
//                 1.0
//             } else {
//                 0.0
//             };
//         }
//     }

//     // Smooth out the charateristic function
//     let smoothing_diameter =
//         (consts::FILTER_SMOOTHING_STD * 3.0 * 2.0).ceil() as usize;
//     let smoothing_kernel =
//         gaussian_kernel(smoothing_diameter, consts::FILTER_SMOOTHING_STD);
//     let weights = charateristic
//         .conv_fft(&smoothing_kernel, ConvMode::Same, PaddingMode::Zeros)
//         .unwrap();

//     // Ensure the charateristic function decays the further from (0,0) we go
//     let decay_kernel = gaussian_kernel(SIZE, consts::FILTER_DECAY_STD);
//     let weights = weights * decay_kernel;

//     weights
// }

// /// Takes a given (x,y) offset from a pixel and returns it's weight.
// fn weight(i: usize, x: isize, y: isize, anisotropy: &Anisotropy) -> f64 {
//     let omega_weights = disc_space_weighting(i);
//     let offset: Array1<f64> = array![x as f64, y as f64];
//     let omega_offset = anisotropy.transform * anisotropy.rotation * offset;
//     // Continue implementing this.
//     todo!()
// }

pub fn run(args: &Args) -> ImageResult<()> {
    let img = load_image(&args.input)?;

    let gradients = compute_gradients(&img, consts::PARTIAL_DERIVATIVE_KERNEL_SIZE);

    let structure_tensors = compute_structure_tensors(gradients);
    let structure_tensors = smooth_structure_tensors(&structure_tensors);

    let anisotropy = structure_tensors.map(calc_anisotropy);

    // compute the kernels at different orientations and save them.
    // Function for w_i

    // for each pixel within a certain area, for each sector, compute
    // m_i and s_i^2

    // find the weighted sum of them and use that for the final output.
    println!("Done!");
    todo!()
}

#[cfg(test)]
mod tests {

    use ndarray::{Array, Dimension};
    use std::path::Path;

    use super::*;

    const TEST_FILE: &str = "test.png";

    fn get_tests_root() -> String {
        let root = env!("CARGO_MANIFEST_DIR");
        format!("{}/tests", root)
    }

    fn get_test_file_name() -> String {
        let path = Path::new(TEST_FILE);
        path.file_stem().unwrap().to_string_lossy().into_owned()
    }

    #[test]
    fn image_conversion() {
        let root = env!("CARGO_MANIFEST_DIR");
        let filepath = PathBuf::from(format!("{}/{}", root, TEST_FILE));

        let arr1 = load_image(&filepath).unwrap();

        let img1 = ndarray_to_rgbimage(arr1.clone());

        let arr2 = image_to_ndarray(img1);

        assert_eq!(arr1, arr2);
    }

    /// Normalise any array to be between the values 0 and 1.
    fn normalise<D: Dimension>(arr: Array<f64, D>) -> Array<f64, D> {
        let maximum = arr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let minimum = arr.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        arr.map(|x| (x - minimum) / (maximum - minimum))
    }

    #[test]
    fn gradient_calculations() {
        let filepath = PathBuf::from(format!("{}/{}", get_tests_root(), TEST_FILE));

        let img = load_image(&filepath).unwrap();

        let gradients = compute_gradients(&img, consts::PARTIAL_DERIVATIVE_KERNEL_SIZE);

        fn remove_first_dimension(dim: (usize, usize, usize)) -> (usize, usize) {
            assert_eq!(dim.0, 3);
            (dim.1, dim.2)
        }

        assert_eq!(gradients.x.dim(), img.dim());
        assert_eq!(gradients.y.dim(), img.dim());

        let img_x_grad = ndarray_to_rgbimage(normalise(gradients.x));
        let img_y_grad = ndarray_to_rgbimage(normalise(gradients.y));

        let filepath_x_grad = PathBuf::from(format!(
            "{}/{}_x_grad.png",
            get_tests_root(),
            get_test_file_name()
        ));
        let filepath_y_grad = PathBuf::from(format!(
            "{}/{}_y_grad.png",
            get_tests_root(),
            get_test_file_name()
        ));

        img_x_grad.save(filepath_x_grad).unwrap();
        img_y_grad.save(filepath_y_grad).unwrap();
    }

    /// Hue is in the range [0, 360), saturation is in the range [0, 1], and value is in the range [0, 1].
    fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f64; 3] {
        let c = v * s;
        let h = h / 60.;
        let x = c * (1. - (h % 2. - 1.).abs());
        let m = v - c;

        let (r, g, b) = if h < 1. {
            (c, x, 0.)
        } else if h < 2. {
            (x, c, 0.)
        } else if h < 3. {
            (0., c, x)
        } else if h < 4. {
            (0., x, c)
        } else if h < 5. {
            (x, 0., c)
        } else {
            (c, 0., x)
        };

        [r + m, g + m, b + m]
    }

    /// Given an ndarray containing elements of the range [0, 1], map this to hue
    /// values (red is left, green is up, teal is right, purple is down)
    fn angle_ndarray_to_rgbimage(arr: Array2<f64>) -> RgbImage {
        let get_rgb = |angle: f64| {
            let hue = angle * 360.0;
            let saturation = 0.5;
            let value = 0.8;
            hsv_to_rgb(hue, saturation, value)
        };
        let rs = arr.map(|&angle| get_rgb(angle)[0]);
        let gs = arr.map(|&angle| get_rgb(angle)[1]);
        let bs = arr.map(|&angle| get_rgb(angle)[2]);
        let arr = stack![Axis(0), rs, gs, bs];

        ndarray_to_rgbimage(arr)
    }

    /// Maps angles to hues, and strength to saturation
    fn angle_and_strength_to_rgbimage(
        angle_img: Array2<f64>,
        strength_img: Array2<f64>,
    ) -> RgbImage {
        fn get_rgb(angle: f64, strength: f64) -> [f64; 3] {
            let hue = angle * 360.0;
            let saturation = strength;
            let value = 0.8;
            hsv_to_rgb(hue, saturation, value)
        }

        let arr = Zip::from(&angle_img)
            .and(&strength_img)
            .map_collect(|&angle, &strength| get_rgb(angle, strength));

        // Convert Array2<[f64; 3]> into Array3 of dimension (3, H, W)
        let (height, width) = arr.dim();

        let arr = Array::from_shape_vec(
            (height, width, 3),
            arr.iter()
                .flat_map(|rgb| rgb.into_iter().copied())
                .collect(),
        )
        .unwrap();

        let arr = arr.permuted_axes([2, 0, 1]);

        ndarray_to_rgbimage(arr)
    }

    #[test]
    fn anisotropy_calculations() {
        let filepath = PathBuf::from(format!("{}/{}", get_tests_root(), TEST_FILE));

        let img = load_image(&filepath).unwrap();
        let gradients = compute_gradients(&img, consts::PARTIAL_DERIVATIVE_KERNEL_SIZE);

        let structure_tensors = compute_structure_tensors(gradients);
        let structure_tensors = smooth_structure_tensors(&structure_tensors);
        let anisotropy = structure_tensors.map(calc_anisotropy);

        let strength = anisotropy.mapv(|anisotropy| anisotropy.anisotropy);
        let angle = anisotropy.mapv(|anisotropy| (anisotropy.angle + (PI / 2.0)) / PI);
        // The elements of angle is normalised from [-PI/2, PI/2] to [0,1]

        let img_anisotropy = angle_and_strength_to_rgbimage(angle, strength);

        let filepath_anisotropy = PathBuf::from(format!(
            "{}/{}_anisotropy.png",
            get_tests_root(),
            get_test_file_name()
        ));

        img_anisotropy.save(filepath_anisotropy).unwrap();
    }
}
