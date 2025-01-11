//! Different tools for
//! - Converting between ndarray and image
//! - Converting rgb into hsv ndarrays
//! - Converting function definitions into ndarrays

use image::{GrayImage, RgbImage};
use ndarray::{stack, Array, Array2, Array3, Axis, Dimension, Zip};

/// Hue is in the range [0, 360), saturation is in the range [0, 1], and value
/// is in the range [0, 1].
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
pub fn angle_and_strength_to_rgbimage(
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

// Turns an image into an ndarray with dimensions (3, height, width)
pub fn image_to_ndarray(img: RgbImage) -> Array3<f64> {
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

pub fn ndarray_to_rgbimage(arr: Array3<f64>) -> RgbImage {
    let arr = arr.permuted_axes([1, 2, 0]);
    let (height, width, _) = arr.dim();
    let data = arr.iter().map(|x| (x * 255.0) as u8).collect::<Vec<u8>>();
    // Will not return none because the dimensions are correct
    RgbImage::from_raw(width as u32, height as u32, data).unwrap()
}

/// Assumes the input array is a grayscale image with values in the range [0, 1]
pub fn ndarray_to_grayimage(arr: Array2<f64>) -> GrayImage {
    let (height, width) = arr.dim();
    let data = arr.iter().map(|x| (x * 255.0) as u8).collect::<Vec<u8>>();
    // Will not return none because the dimensions are correct
    GrayImage::from_raw(width as u32, height as u32, data).unwrap()
}

/// Normalise any array to be between the values 0 and 1.
pub fn normalise<D: Dimension>(arr: &Array<f64, D>) -> Array<f64, D> {
    let maximum = arr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let minimum = arr.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    arr.map(|x| (x - minimum) / (maximum - minimum))
}

/// Creates a 2D odd-side-length kernel defined by a function f(x,y), where (x,y)
/// is (0,0) at the centre of the kernel.
/// Panics if the kernel length is not odd
pub fn array2_from_fn(kernel_size: usize, f: impl Fn(f64, f64) -> f64) -> Array2<f64> {
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
