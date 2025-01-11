#![allow(dead_code)]

use clap::Parser;
use image::{GrayImage, ImageReader, ImageResult, RgbImage};
use indicatif::ProgressStyle;
use ndarray::{array, stack, Array1, Array2, Array3, Axis, Zip};
use ndarray_conv::{ConvExt, ConvMode, PaddingMode};
use std::f64::consts::PI;
use std::path::PathBuf;
use std::time::Instant;

mod consts {
    // Number of sectors in the filter
    pub static NUM_SECTORS: usize = 8;
    // Standard deviation for Gaussian partial derivatives
    pub static PARTIAL_DERIVATIVE_STD: f64 = 1.0;
    // Gaussian spatial derivatives kernel size
    pub static PARTIAL_DERIVATIVE_KERNEL_SIZE: usize = 5;
    // Standard deviation for smoothing structure tensors
    pub static TENSOR_SMOOTHING_STD: f64 = 1.0;
    // Standard deviation of filter sector smoothing
    pub static FILTER_SMOOTHING_STD: f64 = 1.0;
    // Standard deviation for filter decay
    pub static FILTER_DECAY_STD: f64 = 3.0;
    // TODO: Something
    pub static SHARPNESS_COEFFICIENT: u64 = 8;
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
    assert_eq!(
        kernel_size % 2,
        1,
        "Kernel length {} must be odd",
        kernel_size
    );

    let half_kernel_size = (kernel_size / 2) as i32;
    Array2::from_shape_fn((kernel_size, kernel_size), |(y, x)| {
        let x = x as i32 - half_kernel_size;
        let y = y as i32 - half_kernel_size;
        f(x as f64, y as f64)
    })
}

fn gaussian_kernel(size: usize, std: f64) -> Array2<f64> {
    let f = |x, y| gaussian(x, y, std);
    let kernel = array2_from_fn(size, f);
    kernel
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

/// The i-th weighting function for the disc, returned as a 2D array centered at
/// (0,0). This is the weight function that should be applied to pixels after
/// the transformation and rotation have been applied. The different values of
/// `i`` reprsent which segment of the disc we are considering.
fn disc_space_weighting(i: usize) -> Array2<f64> {
    // Calculate the charateristic function
    const SIZE: usize = 27;
    let characteristic = array2_from_fn(SIZE, |x, y| {
        let half_size = (SIZE / 2) as f64;
        let inside_circle = x.powi(2) + y.powi(2) <= (half_size).powi(2);

        let angle = y.atan2(x);
        let lower = ((2.0 * i as f64 - 1.0) * PI) / consts::NUM_SECTORS as f64;
        let upper = ((2.0 * i as f64 + 1.0) * PI) / consts::NUM_SECTORS as f64;

        let inside_segment1 = lower < angle && angle <= upper;
        let inside_segment2 = lower < angle + 2.0 * PI && angle + 2.0 * PI <= upper;
        let inside_segment = inside_segment1 || inside_segment2;

        if inside_circle && inside_segment {
            1.0
        } else {
            0.0
        }
    });

    // Smooth out the characteristic function
    let smoothing_diameter = (consts::FILTER_SMOOTHING_STD).ceil() as usize * 3 * 2 + 1;
    let smoothing_kernel = gaussian_kernel(smoothing_diameter, consts::FILTER_SMOOTHING_STD);
    let weights = characteristic
        .conv(&smoothing_kernel, ConvMode::Same, PaddingMode::Zeros)
        .unwrap();

    // Ensure the characteristic function decays the further from (0,0) we go
    let decay_kernel = gaussian_kernel(SIZE, consts::FILTER_DECAY_STD);
    let weights = weights * decay_kernel;

    weights
}

fn query_point_in_array2(point: &Array1<f64>, arr: &Array2<f64>) -> f64 {
    let (height, width) = arr.dim();
    let half_height = height / 2;
    let half_width = width / 2;

    let x = point[0] as usize + half_width;
    let y = point[1] as usize + half_height;

    if !(0..height).contains(&y) || !(0..width).contains(&x) {
        0.0
    } else {
        arr[[y, x]]
    }
}

/// Takes a given (x,y) offset from a pixel and returns it's weight.
/// TODO: Update this comment
fn sector_weight(i: usize, x: isize, y: isize, anisotropy: &Anisotropy) -> f64 {
    let omega_weights = disc_space_weighting(i);
    let offset: Array1<f64> = array![x as f64, y as f64];
    let omega_offset = anisotropy.transform.dot(&anisotropy.rotation.dot(&offset));
    query_point_in_array2(&omega_offset, &omega_weights)
}

/// TODO: Define a type for Array3

/// Mean and var are 2D arrays of size [consts::NUM_SECTORS, 3]. The ith value
/// represents the mean/variance of the RGB values in the ith sector.
struct PixelStatistics {
    mean: Array2<f64>,
    var: Array2<f64>,
}

fn calc_weighted_local_statistics(
    x0: usize,
    y0: usize,
    img: &Array3<f64>, // [3, H, W]
    anisotropy: &Anisotropy,
) -> PixelStatistics {
    const WINDOW_SIZE: usize = 27 * 2;
    const HALF_WINDOW_SIZE: usize = WINDOW_SIZE / 2;

    let mut mean: Array2<f64> = Array2::zeros((consts::NUM_SECTORS, 3));
    let mut var: Array2<f64> = Array2::zeros((consts::NUM_SECTORS, 3));

    let mut divisor: Array1<f64> = Array1::zeros(consts::NUM_SECTORS);

    let (_, height, width) = img.dim();
    for y in y0 as isize - HALF_WINDOW_SIZE as isize..=y0 as isize + HALF_WINDOW_SIZE as isize {
        for x in x0 as isize - HALF_WINDOW_SIZE as isize..=x0 as isize + HALF_WINDOW_SIZE as isize {
            // (y1, x1) is the actual position of the pixel
            let y1 = y + y0 as isize;
            let x1 = x + x0 as isize;
            if !(0..height as isize).contains(&y1) || !(0..width as isize).contains(&x1) {
                continue;
            }

            // (y1, x1) are definitely inside the image, it's safe to type-cast
            // them to usizes
            let y1 = y1 as usize;
            let x1 = x1 as usize;

            for i in 0..consts::NUM_SECTORS {
                let weight = sector_weight(i, x as isize, y as isize, &anisotropy);
                for c in 0..3 {
                    mean[[i, c]] += weight * img[[c, y1, x1]];
                    var[[i, c]] += weight * img[[c, y1, x1]] * img[[c, y1, x1]];
                }
                divisor[[i]] += weight;
            }
        }
    }

    // Normalise the mean and variance by the sum of weights.
    for i in 0..consts::NUM_SECTORS {
        for c in 0..3 {
            mean[[i, c]] /= divisor[[i]];
        }
    }
    for i in 0..consts::NUM_SECTORS {
        for c in 0..3 {
            var[[i, c]] /= divisor[[i]];
            var[[i, c]] -= mean[[i, c]]
        }
    }

    PixelStatistics { mean, var }
}

/// Calculates the final pixel value as a size 3 Array1
fn calculate_pixel_value(
    x: usize,
    y: usize,
    img: &Array3<f64>,
    anisotropy: &Array2<Anisotropy>,
) -> Array1<f64> {
    let PixelStatistics { mean, var } =
        calc_weighted_local_statistics(x, y, img, &anisotropy[[y, x]]);

    let mut output: Array1<f64> = Array1::zeros(3);
    let mut divisor: Array1<f64> = Array1::zeros(3);

    let std = var.sqrt();
    for i in 0..consts::NUM_SECTORS {
        let norm = (std[[i, 0]].powi(2) + std[[i, 1]].powi(2) + std[[i, 2]].powi(2)).sqrt();
        let weighting_factor = 1.0 / (1.0 + norm.powi(consts::SHARPNESS_COEFFICIENT as i32));

        for c in 0..3 {
            output[c] += weighting_factor * mean[[i, c]];
            divisor[c] += weighting_factor;
        }
    }

    output / divisor
}

fn generate_output(img: &Array3<f64>, anisotropy: &Array2<Anisotropy>) -> Array3<f64> {
    let (_, height, width) = img.dim();

    let mut output = Array3::zeros((3, height, width));

    let start = Instant::now();

    let pb = indicatif::ProgressBar::new((height * width) as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%) ETA: {eta}")
        .unwrap()
        .progress_chars("#>-"));

    for y in 0..height {
        for x in 0..width {
            let rgb = calculate_pixel_value(x, y, img, anisotropy);
            for c in 0..3 {
                output[[c, y, x]] = rgb[c];
            }
            pb.inc(1);
        }
    }

    pb.finish_with_message("done!");

    let duration = start.elapsed();
    println!("Took {:.2?} seconds", duration);

    output
}

// TODO: Put debug switches into here.
pub fn run(args: &Args) -> ImageResult<()> {
    let img = load_image(&args.input)?;

    let gradients = compute_gradients(&img, consts::PARTIAL_DERIVATIVE_KERNEL_SIZE);

    let structure_tensors = compute_structure_tensors(gradients);
    let structure_tensors = smooth_structure_tensors(&structure_tensors);

    let anisotropy = structure_tensors.map(calc_anisotropy);

    let output = generate_output(&img, &anisotropy);

    let output_img = ndarray_to_rgbimage(output);

    output_img.save(&args.get_output())?;

    Ok(())
}

#[cfg(test)]
mod tests {

    use image::{EncodableLayout, ImageBuffer, Pixel, PixelWithColorType};
    use ndarray::{Array, Dimension};
    use std::{ops::Deref, path::Path};

    use super::*;

    // const TEST_FILE: &str = "test2.png";
    const TEST_FILE: &str = "field.webp";

    fn get_tests_root() -> String {
        let root = env!("CARGO_MANIFEST_DIR");
        format!("{}/tests", root)
    }

    fn get_test_file_name() -> String {
        let path = Path::new(TEST_FILE);
        path.file_stem().unwrap().to_string_lossy().into_owned()
    }

    fn save_with_suffix<P, Container>(img: ImageBuffer<P, Container>, suffix: &str)
    where
        P: Pixel + PixelWithColorType,
        [P::Subpixel]: EncodableLayout,
        Container: Deref<Target = [P::Subpixel]>,
    {
        let filepath = PathBuf::from(format!(
            "{}/{}{}.png",
            get_tests_root(),
            get_test_file_name(),
            suffix
        ));

        img.save(filepath).unwrap();
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

        save_with_suffix(img_x_grad, "_x_grad");
        save_with_suffix(img_y_grad, "_y_grad");
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
            let saturation = 0.5;
            let value = strength;
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

        save_with_suffix(img_anisotropy, "_anisotropy");
    }

    #[test]
    fn check_weighting_function() {
        for i in 0..consts::NUM_SECTORS {
            let img = ndarray_to_grayimage(normalise(disc_space_weighting(i)));
            save_with_suffix(img, &format!("_weight_{i}"));
        }
    }
}
