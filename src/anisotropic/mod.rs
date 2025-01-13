#![allow(dead_code)]

use image::{EncodableLayout, ImageBuffer, ImageReader, ImageResult, Pixel, PixelWithColorType};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{array, stack, Array1, Array2, Array3, Axis, Zip};
use ndarray_conv::{ConvExt, ConvMode, PaddingMode};
use std::f64::consts::PI;
use std::ops::Deref;
use std::path::PathBuf;

pub mod args;
mod converters;

use args::Args;

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

fn load_image(input_file: &PathBuf) -> ImageResult<Array3<f64>> {
    let img = ImageReader::open(input_file)?.decode()?.to_rgb8();
    let img = converters::image_to_ndarray(img);
    Ok(img)
}

fn save_with_suffix<P, Container>(args: &Args, img: ImageBuffer<P, Container>, suffix: &str)
where
    P: Pixel + PixelWithColorType,
    [P::Subpixel]: EncodableLayout,
    Container: Deref<Target = [P::Subpixel]>,
{
    let filepath = PathBuf::from(format!(
        "{}/{}{}.png",
        args.get_output_dir(),
        args.get_file_stem(),
        suffix
    ));

    img.save(filepath).unwrap();
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

fn gaussian_kernel(size: usize, std: f64) -> Array2<f64> {
    let f = |x, y| gaussian(x, y, std);
    let kernel = converters::array2_from_fn(size, f);
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

    /// Finds the eigenvalue of a structure tensor. The largest eigenvalue is given first.
    fn get_eigenvalues(&self) -> (f64, f64) {
        let determinant = (self.e - self.g).powi(2) + 4.0 * self.f.powi(2);
        let eigenvalue1 = (self.e + self.g + determinant.sqrt()) / 2.0;
        let eigenvalue2 = (self.e + self.g - determinant.sqrt()) / 2.0;
        (eigenvalue1, eigenvalue2)
    }

    fn into_anisotropy(self) -> Anisotropy {
        if self.approx_zero() {
            // This area is isotropic
            let identity = array![[1.0, 0.0], [0.0, 1.0]];
            return Anisotropy {
                anisotropy: 0.0,
                angle: 0.0,
                transform: identity.clone(),
            };
        } else {
            let (eigenvalue1, eigenvalue2) = self.get_eigenvalues();

            let t = array![eigenvalue1 - self.e, -self.f];
            let angle = t[1].atan2(t[0]);

            let anisotropy = (eigenvalue1 - eigenvalue2) / (eigenvalue1 + eigenvalue2);

            const ALPHA: f64 = 1.0;
            let scale = array![
                [ALPHA / (ALPHA + anisotropy), 0.],
                [0., (ALPHA + anisotropy) / ALPHA]
            ];

            let rotation = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];

            let transform = scale.dot(&rotation);

            Anisotropy {
                anisotropy,
                angle,
                transform,
            }
        }
    }
}

/// Represents the partial derivatives of each of the three channels with
/// respect to x and y.  Dimensions are [channels, height, width]
struct Gradients {
    x: Array3<f64>,
    y: Array3<f64>,
}

impl Gradients {
    fn new(img: &Array3<f64>, kernel_size: usize) -> Self {
        /// Normalise kernels so that the sum of absolute values is equal to 2.
        /// The sum is arbritary, but we simply want to avoid floating point error
        /// by using kernels that have values that are too small.
        fn normalise_kernel(arr: Array2<f64>) -> Array2<f64> {
            let sum = arr.map(|x| x.abs()).sum();
            arr / sum
        }

        let f_x = |x, y| gaussian_derivative_x(x, y, consts::PARTIAL_DERIVATIVE_STD);
        let f_y = |x, y| gaussian_derivative_y(x, y, consts::PARTIAL_DERIVATIVE_STD);

        let x_kernel = normalise_kernel(converters::array2_from_fn(kernel_size, f_x));
        let y_kernel = normalise_kernel(converters::array2_from_fn(kernel_size, f_y));

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
    fn into_structure_tensors(self) -> Array2<StructureTensor> {
        #[inline]
        fn dot_product(a1: &Array3<f64>, a2: &Array3<f64>) -> Array2<f64> {
            (a1 * a2).sum_axis(Axis(0))
        }

        let fxx = dot_product(&self.x, &self.x);
        let fxy = dot_product(&self.x, &self.y);
        let fyy = dot_product(&self.y, &self.y);

        Zip::from(&fxx)
            .and(&fxy)
            .and(&fyy)
            .map_collect(|&e, &f, &g| StructureTensor::new(e, f, g))
    }
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

#[derive(Clone, Debug)]
struct Anisotropy {
    anisotropy: f64,
    angle: f64, // bounded by [-PI, PI]
    transform: Array2<f64>,
}

/// The i-th weighting function for the disc, returned as a 2D array centered at
/// (0,0). This is the weight function that should be applied to pixels after
/// the transformation and rotation have been applied. The different values of
/// `i`` reprsent which segment of the disc we are considering.
fn get_disc_space_weighting(i: usize) -> Array2<f64> {
    assert!(i < consts::NUM_SECTORS);

    // Calculate the charateristic function
    const SIZE: usize = 27;
    let characteristic = converters::array2_from_fn(SIZE, |x, y| {
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

/// A quick and dirty function that just truncates the floating points of the
/// coordinates in the point to get the array value.
fn query_point_in_array2(point: &Array1<f64>, arr: &Array2<f64>) -> f64 {
    let (height, width) = arr.dim();
    let half_height = height / 2;
    let half_width = width / 2;

    let x = point[0] as isize + half_width as isize;
    let y = point[1] as isize + half_height as isize;

    if !(0..height as isize).contains(&y) || !(0..width as isize).contains(&x) {
        0.0
    } else {
        // (y, x) are within the image, and can be safely converted to usize
        arr[[y as usize, x as usize]]
    }
}

type DiscWeights = [Array2<f64>; consts::NUM_SECTORS];

/// Mean and var are 2D arrays of size [consts::NUM_SECTORS, 3]. The ith value
/// represents the mean/variance of the RGB values in the ith sector.
struct PixelStatistics {
    mean: Array2<f64>,
    var: Array2<f64>,
}

impl PixelStatistics {
    fn new(
        x0: usize,
        y0: usize,
        img: &Array3<f64>,
        anisotropy: &Anisotropy,
        disc_weights: &DiscWeights,
    ) -> Self {
        const WINDOW_SIZE: isize = 27 * 2;
        const HALF_WINDOW_SIZE: isize = WINDOW_SIZE / 2;

        let (_, height, width) = img.dim();

        let mut mean: Array2<f64> = Array2::zeros((consts::NUM_SECTORS, 3));
        let mut var: Array2<f64> = Array2::zeros((consts::NUM_SECTORS, 3));
        let mut divisor: Array1<f64> = Array1::zeros(consts::NUM_SECTORS);

        for y in -HALF_WINDOW_SIZE..HALF_WINDOW_SIZE {
            for x in -HALF_WINDOW_SIZE..HALF_WINDOW_SIZE {
                // (y1, x1) is the actual position of the pixel
                let y1 = y + y0 as isize;
                let x1 = x + x0 as isize;
                if !(0..height as isize).contains(&y1) || !(0..width as isize).contains(&x1) {
                    continue;
                }
                // (y1, x1) are definitely inside the image, so  it's safe to
                // type-cast them to usizes
                let y1 = y1 as usize;
                let x1 = x1 as usize;

                let offset = array![x as f64, y as f64];
                let disc_offset = anisotropy.transform.dot(&offset);

                // Optimisation: We don't bother to calculate the weight for this
                // pixel if we know that it is outside of the effective radius
                // of the sector kernels.
                static EFFECTIVE_RADIUS_SQUARED: f64 =
                    (consts::FILTER_DECAY_STD * 3.0) * (consts::FILTER_DECAY_STD * 3.0);
                if disc_offset[0].powi(2) + disc_offset[1].powi(2) > EFFECTIVE_RADIUS_SQUARED {
                    continue;
                }

                for i in 0..consts::NUM_SECTORS {
                    let weight = query_point_in_array2(&disc_offset, &disc_weights[i]);
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
                var[[i, c]] -= mean[[i, c]] * mean[[i, c]];
            }
        }

        PixelStatistics { mean, var }
    }
}

/// Calculates the final pixel value of (x,y) in the image as a Array1
/// containing an RGB value
fn calculate_pixel_value(
    x: usize,
    y: usize,
    img: &Array3<f64>,
    anisotropy: &Anisotropy,
    disc_weights: &DiscWeights,
) -> Array1<f64> {
    // Calculate the mean and variance of each of the sectors in our image.
    let PixelStatistics { mean, var } = PixelStatistics::new(x, y, img, &anisotropy, disc_weights);

    let mut output: Array1<f64> = Array1::zeros(3);
    let mut divisor: Array1<f64> = Array1::zeros(3);

    let std = var.sqrt();
    for i in 0..consts::NUM_SECTORS {
        // We multiply the norm by 255, since the paper expects pixel values
        // between [0, 255], whereas we currently use [0, 1]
        let norm = (std[[i, 0]].powi(2) + std[[i, 1]].powi(2) + std[[i, 2]].powi(2)).sqrt() * 255.0;
        let weighting_factor = 1.0 / (1.0 + norm.powi(consts::SHARPNESS_COEFFICIENT as i32));

        for c in 0..3 {
            output[c] += weighting_factor * mean[[i, c]];
            divisor[c] += weighting_factor;
        }
    }

    output / divisor
}

fn apply_kuwahara_filter(
    img: &Array3<f64>,
    anisotropy: &Array2<Anisotropy>,
    disc_weights: &DiscWeights,
) -> Array3<f64> {
    let (_, height, width) = img.dim();

    let mut output = Array3::zeros((3, height, width));

    let pb = ProgressBar::new((height * width) as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:50.cyan/blue}] {pos}/{len} ({percent}%) ETA: {eta}")
        .unwrap()
        .progress_chars("#>-"));

    for y in 0..height {
        for x in 0..width {
            let rgb = calculate_pixel_value(x, y, img, &anisotropy[[y, x]], disc_weights);
            for c in 0..3 {
                output[[c, y, x]] = rgb[c];
            }
            pb.inc(1);
        }
    }

    pb.finish_with_message("done!");

    output
}

pub fn run(args: &Args) -> ImageResult<()> {
    let img = load_image(&args.input)?;

    let gradients = Gradients::new(&img, consts::PARTIAL_DERIVATIVE_KERNEL_SIZE);

    if args.intermediate_results {
        let img_x_grad = converters::ndarray_to_rgbimage(converters::normalise(&gradients.x));
        let img_y_grad = converters::ndarray_to_rgbimage(converters::normalise(&gradients.y));

        save_with_suffix(args, img_x_grad, "_x_grad");
        save_with_suffix(args, img_y_grad, "_y_grad");
    }

    let structure_tensors = gradients.into_structure_tensors();

    if args.intermediate_results {
        let anisotropy = structure_tensors.map(|tensor| tensor.into_anisotropy());

        let strength = anisotropy.mapv(|anisotropy| anisotropy.anisotropy);
        let angle = anisotropy.mapv(|anisotropy| (anisotropy.angle + (PI / 2.0)) / PI);
        // The elements of angle are normalised from [-PI/2, PI/2] to [0,1]

        let img_anisotropy = converters::angle_and_strength_to_rgbimage(angle, strength);

        save_with_suffix(args, img_anisotropy, "_unsmoothed_anisotropy");
    }

    let structure_tensors = smooth_structure_tensors(&structure_tensors);

    let anisotropy = structure_tensors.map(|tensor| tensor.into_anisotropy());

    if args.intermediate_results {
        let strength = anisotropy.mapv(|anisotropy| anisotropy.anisotropy);
        let angle = anisotropy.mapv(|anisotropy| (anisotropy.angle + (PI / 2.0)) / PI);
        // The elements of angle are normalised from [-PI/2, PI/2] to [0,1]

        let img_anisotropy = converters::angle_and_strength_to_rgbimage(angle, strength);

        save_with_suffix(args, img_anisotropy, "_anisotropy");
    }

    let disc_weights: DiscWeights = std::array::from_fn(|i| get_disc_space_weighting(i));

    let output = apply_kuwahara_filter(&img, &anisotropy, &disc_weights);

    let output_img = converters::ndarray_to_rgbimage(output);

    let output_file = format!("{}/{}", args.get_output_dir(), args.get_output_filename());
    output_img.save(&output_file)?;

    Ok(())
}
