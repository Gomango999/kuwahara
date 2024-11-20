use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

mod basic;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Basic(basic::Args),
}

fn main() {
    let start_time = Instant::now();

    {
        let cli = Cli::parse();
        match &cli.command {
            Commands::Basic(args) => {
                if let Err(e) = basic::run(args) {
                    eprintln!("Error generating image: {e}");
                    std::process::exit(1)
                }
            }
        }
    }

    let duration = start_time.elapsed();
    println!("Completed in {:.3?}", duration);
}
