use clap::{arg, value_parser, Args};
use image::{ImageReader, ImageResult, Rgb, RgbImage};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    cmp::{max, min},
    path::{Path, PathBuf},
};

// Command information
pub const NAME : &str = "basic";
pub fn command() -> clap::Command {
    clap::Command::new(NAME)
    .about("Basic kuwahara filter, as described on Wikipedia")
    .arg(
        arg!(-i --input "The input image file")
            .value_parser(value_parser!(PathBuf)),
    )
    .arg(arg!(-k --kernel_size "The side length of each quadrant"))
}

// The Kuwahara filter computes the mean and variance of the
// 4 quadrants surrounding pixel (x,y) and sets that pixel
// to the mean value of the quadrant that has the smallest
// variance.
pub fn run(args: &dyn Args) -> ImageResult<RgbImage> {
    let input = args.get_one::<PathBuf>("input").

    // load images
    let filepath = filepath.to_str().unwrap();
    let img = ImageReader::open(filepath)?.decode()?.to_rgb8();
    let (width, height) = img.dimensions();

    // create a progress bar
    let num_pixels = (width * height) as u64;
    let pb = ProgressBar::new(num_pixels);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40}] {percent}%")
            .expect("Progress bar template incorrect")
            .progress_chars("=>-"),
    );

    // process pixels
    let mut result = RgbImage::new(width, height);
    const WINDOW_SIZE: u32 = 5;
    for y in 0..height {
        for x in 0..width {
            let x_i32 = x as i32;
            let y_i32 = y as i32;
            let a_i32 = WINDOW_SIZE as i32;
            let quadrants = vec![
                compute_quadrant(
                    &img,
                    x_i32 - a_i32 + 1,
                    y_i32 - a_i32 + 1,
                    WINDOW_SIZE,
                ),
                compute_quadrant(&img, x_i32, y_i32 - a_i32 + 1, WINDOW_SIZE),
                compute_quadrant(&img, x_i32 - a_i32 + 1, y_i32, WINDOW_SIZE),
                compute_quadrant(&img, x_i32, y_i32, WINDOW_SIZE),
            ];

            let min_quadrant = quadrants
                .iter()
                .min_by(|q1, q2| q1.var.partial_cmp(&q2.var).unwrap())
                .unwrap();

            result.put_pixel(x, y, min_quadrant.mean_color);

            pb.inc(1);
        }
    }

    if let Err(e) = result.save(cli.get_output()) {
        eprintln!("Error saving image: {e}");
        std::process::exit(1)
    }

    pb.finish_with_message("Done!");

    return Ok(result);
};

fn rgb_to_brightness(rgb: Rgb<u8>) -> u8 {
    let Rgb(rgb) = rgb;
    // array is guaranteed to be non-empty
    rgb.iter().copied().max().unwrap()
}

fn clamp_window_coordinates(
    x_start: i32,
    y_start: i32,
    side_length: u32,
    width: u32,
    height: u32,
) -> ((u32, u32), (u32, u32)) {
    let x_start = max(0, x_start) as u32;
    let y_start = max(0, y_start) as u32;
    let x_end = min(x_start + side_length, width);
    let y_end = min(y_start + side_length, height);
    return ((x_start, x_end), (y_start, y_end));
}

// Computes the mean pixel color in the a*a window with top left corner (x_start, y_start)
// This function lets you query squares that are partially outside
fn compute_window_mean_color(
    img: &RgbImage,
    left: i32,
    top: i32,
    side_length: u32,
) -> Rgb<u8> {
    let (width, height) = img.dimensions();
    let ((x_start, x_end), (y_start, y_end)) =
        clamp_window_coordinates(left, top, side_length, width, height);

    let mut total = Rgb::<u32>([0, 0, 0]);
    let mut count = 0;
    for y in y_start..y_end {
        for x in x_start..x_end {
            if let Some(Rgb([r, g, b])) = img.get_pixel_checked(x, y) {
                total = Rgb::<u32>([
                    total.0[0] + (*r as u32),
                    total.0[1] + (*g as u32),
                    total.0[2] + (*b as u32),
                ]);
                count += 1;
            }
        }
    }
    return Rgb::<u8>([
        (total.0[0] as f32 / count as f32) as u8,
        (total.0[1] as f32 / count as f32) as u8,
        (total.0[2] as f32 / count as f32) as u8,
    ]);
}

// Computes the mean pixel brightness in the a*a window with top left corner (x_start, y_start)
// This function lets you query squares that are partially outside
fn compute_window_mean_brightness(
    img: &RgbImage,
    left: i32,
    top: i32,
    side_length: u32,
) -> u8 {
    let (width, height) = img.dimensions();
    let ((x_start, x_end), (y_start, y_end)) =
        clamp_window_coordinates(left, top, side_length, width, height);

    let mut total: u32 = 0;
    let mut count = 0;
    for y in y_start..y_end {
        for x in x_start..x_end {
            if let Some(rgb) = img.get_pixel_checked(x, y) {
                total += rgb_to_brightness(*rgb) as u32;
                count += 1;
            }
        }
    }
    return (total / count as u32) as u8;
}

// Computes the pixel brightness variance in the a*a window with top left corner (x_start, y_start)
// This function lets you query squares that are partially outside
fn compute_window_var(
    img: &RgbImage,
    left: i32,
    top: i32,
    side_length: u32,
    mean: u8,
) -> f32 {
    let mean = mean as i32;
    let (width, height) = img.dimensions();
    let ((x_start, x_end), (y_start, y_end)) =
        clamp_window_coordinates(left, top, side_length, width, height);
    let mut total = 0;
    let mut count: u32 = 0;
    for y in y_start..y_end {
        for x in x_start..x_end {
            if let Some(rgb) = img.get_pixel_checked(x, y) {
                let diff = (rgb_to_brightness(*rgb) as i32) - mean;
                total += diff * diff;
                count += 1;
            }
        }
    }
    return if count == 1 {
        0.
    } else {
        total as f32 / (count - 1) as f32
    };
}

#[derive(Debug)]
struct QuadrantResult {
    mean_color: Rgb<u8>,
    var: f32,
}

fn compute_quadrant(
    img: &RgbImage,
    x: i32,
    y: i32,
    side_length: u32,
) -> QuadrantResult {
    let mean = compute_window_mean_brightness(img, x, y, side_length);
    let var = compute_window_var(img, x, y, side_length, mean);
    let mean_color = compute_window_mean_color(img, x, y, side_length);
    return QuadrantResult { mean_color, var };
}

fn generate_output_filepath(input: &Path, output: Option<PathBuf>) -> PathBuf {
    match &self.output {
        Some(filepath) => filepath.clone(),
        None => {
            let basename = self.input.parent().unwrap().to_str().unwrap();
            let basename = PathBuf::from(basename);

            let filename = format!(
                "{}_kuwahara.png",
                self.input.file_stem().unwrap().to_str().unwrap()
            );
            let filename = PathBuf::from(filename);

            let filepath = basename.join(filename);
            filepath
        }
    }
}

