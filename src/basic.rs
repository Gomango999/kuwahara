#![allow(dead_code)]

use clap::Parser;
use image::{ImageReader, ImageResult, Rgb, RgbImage};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    cmp::{max, min},
    path::PathBuf,
};

#[derive(Parser)]
#[command(about = "The basic Kuwahara filter")]
#[command(
    long_about = "The Kuwahara filter computes the mean and variance of the
4 quadrants surrounding pixel (x,y) and sets that pixel
to the mean value of the quadrant that has the smallest
variance."
)]
pub struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long, default_value_t = 5)]
    kernel_size: u32,
}
impl Args {
    /// Generates a default output filename based on the input filename
    fn get_output(&self) -> PathBuf {
        let basename = self.input.parent().unwrap().to_str().unwrap();
        let basename = PathBuf::from(basename);

        let filename = format!(
            "{}_b_k{}.png",
            self.input.file_stem().unwrap().to_str().unwrap(),
            self.kernel_size
        );
        let filename = PathBuf::from(filename);

        let filepath = basename.join(filename);
        filepath
    }
}

pub fn run(args: &Args) -> ImageResult<()> {
    // Load images
    let img = ImageReader::open(&args.input)?.decode()?.to_rgb8();
    let (width, height) = img.dimensions();

    let num_quadrant_cols = (width + (args.kernel_size - 1) * 2) as usize;
    let num_quadrant_rows = (height + (args.kernel_size - 1) * 2) as usize;
    let num_quadrants = (num_quadrant_rows * num_quadrant_cols) as u64;
    let num_pixels = (width * height) as u64;

    // Create a progress bar
    let pb = ProgressBar::new(num_quadrants + num_pixels);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40}] {percent}%")
            .expect("Progress bar template incorrect")
            .progress_chars("=>-"),
    );

    // `quadrant_cache[y][x]` represents the statistics of the (kernel_size * kernel_size)
    // window with top left corner (x,y). The window may fall off the image,
    // in which case only the intersecting pixels are used for calculation.
    let quadrant_cache: Vec<Vec<_>> = (-(args.kernel_size as i32)..(height as i32))
        .map(|y| {
            (-(args.kernel_size as i32)..(width as i32))
                .map(|x| {
                    let quadrant_result = QuadrantResult::new(&img, (x, y), args.kernel_size);
                    pb.inc(1);
                    quadrant_result
                })
                .collect()
        })
        .collect();

    // For each pixel, compute the quadrant with the least variance in brightness
    // and use it's mean color.
    let result = RgbImage::from_fn(width, height, |x, y| {
        let x = x as usize;
        let y = y as usize;
        let a = args.kernel_size as usize;
        let quadrants = vec![
            &quadrant_cache[y][x],
            &quadrant_cache[y + a - 1][x],
            &quadrant_cache[y][x + a - 1],
            &quadrant_cache[y + a - 1][x + a - 1],
        ];

        let min_quadrant = quadrants
            .iter()
            .min_by(|q1, q2| q1.brightness_var.partial_cmp(&q2.brightness_var).unwrap())
            .unwrap();

        pb.inc(1);

        min_quadrant.mean_color
    });

    let output = args.get_output();
    result.save(output)?;

    pb.finish_with_message("Done!");

    return Ok(());
}

#[derive(Debug, Clone, Default)]
struct Window {
    x_start: u32,
    y_start: u32,
    x_end: u32,
    y_end: u32,
}
impl Window {
    fn new((x, y): (i32, i32), (width, height): (u32, u32), kernel_size: u32) -> Self {
        let x_start = max(0, x) as u32;
        let y_start = max(0, y) as u32;
        let x_end = min(x_start + kernel_size, width);
        let y_end = min(y_start + kernel_size, height);
        return Window {
            x_start,
            y_start,
            x_end,
            y_end,
        };
    }

    fn num_pixels(&self) -> u32 {
        (self.y_end - self.y_start) * (self.x_end - self.x_start)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct QuadrantResult {
    mean_brightness: f32,
    brightness_var: f32,
    mean_color: Rgb<u8>,
}
impl QuadrantResult {
    /// Computes stats for a quadrant with top left corner (x,y)
    fn new(img: &RgbImage, (x, y): (i32, i32), kernel_size: u32) -> Self {
        let window = Window::new((x, y), img.dimensions(), kernel_size);

        let QuadrantMeans {
            brightness: mean_brightness,
            color: mean_color,
        } = QuadrantMeans::new(img, &window);
        let brightness_var = compute_variance(img, &window, mean_brightness);

        return QuadrantResult {
            mean_brightness,
            brightness_var,
            mean_color,
        };
    }
}
impl Default for QuadrantResult {
    fn default() -> Self {
        QuadrantResult {
            mean_brightness: 0.0,
            brightness_var: 0.0,
            mean_color: Rgb([0, 0, 0]),
        }
    }
}

#[derive(Debug, Clone)]
struct QuadrantMeans {
    brightness: f32,
    color: Rgb<u8>,
}
impl QuadrantMeans {
    fn new(img: &RgbImage, window: &Window) -> Self {
        let mut color_sum = [0, 0, 0];
        let mut brightness_sum = 0;
        for y in window.y_start..window.y_end {
            for x in window.x_start..window.x_end {
                if let Some(rgb) = img.get_pixel_checked(x, y) {
                    brightness_sum += rgb_to_brightness(rgb) as u32;
                    let Rgb([r, g, b]) = rgb;
                    color_sum = [
                        color_sum[0] + (*r as u32),
                        color_sum[1] + (*g as u32),
                        color_sum[2] + (*b as u32),
                    ];
                }
            }
        }

        let num_pixels = window.num_pixels();
        let mean_color = Rgb::<u8>([
            (color_sum[0] as f32 / num_pixels as f32) as u8,
            (color_sum[1] as f32 / num_pixels as f32) as u8,
            (color_sum[2] as f32 / num_pixels as f32) as u8,
        ]);
        let mean_brightness = brightness_sum as f32 / num_pixels as f32;

        QuadrantMeans {
            color: mean_color,
            brightness: mean_brightness,
        }
    }
}

/// computes the variance in brightness for a window in an image
fn compute_variance(img: &RgbImage, window: &Window, mean_brightness: f32) -> f32 {
    let mut squared_diff_sum = 0.0;
    for y in window.y_start..window.y_end {
        for x in window.x_start..window.x_end {
            if let Some(rgb) = img.get_pixel_checked(x, y) {
                let diff = (rgb_to_brightness(rgb) as f32) - mean_brightness;
                squared_diff_sum += diff * diff;
            }
        }
    }

    let num_pixels = window.num_pixels();
    let brightness_var = if num_pixels == 1 {
        0.
    } else {
        squared_diff_sum / (num_pixels - 1) as f32
    };
    brightness_var
}

fn rgb_to_brightness(rgb: &Rgb<u8>) -> u8 {
    let Rgb(rgb) = rgb;
    let brightness = rgb.iter().copied().max().unwrap();
    brightness
}
