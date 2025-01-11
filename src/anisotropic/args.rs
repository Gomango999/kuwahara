use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "The anisotropic Kuwahara filter")]
#[command(long_about = "The anisotropic kuwahara filter tries to incorporate the
    directionality of the underlying image into the filter.")]
pub struct Args {
    #[arg(short, long)]
    pub input: PathBuf,

    #[arg(short, long, default_value_t = false)]
    pub intermediate_results: bool,

    #[arg(short, long)]
    pub output: Option<PathBuf>,

    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,
}

impl Args {
    pub fn get_file_stem(&self) -> &str {
        self.input.file_stem().unwrap().to_str().unwrap()
    }

    pub fn get_filename_with_suffix(&self, suffix: &str) -> String {
        format!("{}_{}.png", self.get_file_stem(), suffix)
    }

    /// Generates a default output filename based on the input filename, unless
    /// an output filename was specified by the user.
    pub fn get_output_filename(&self) -> String {
        if let Some(output) = &self.output {
            return output.clone().to_str().unwrap().to_string();
        }

        format!("{}_kuwahara.png", self.get_file_stem())
    }

    /// Uses the output dir if specified, otherwise uses the parent of the input
    /// file as default.
    pub fn get_output_dir(&self) -> &str {
        if let Some(output_dir) = &self.output_dir {
            return output_dir.to_str().unwrap();
        }

        self.input.parent().unwrap().to_str().unwrap()
    }
}
