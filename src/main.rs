use clap::{Parser, Subcommand};
use std::time::Instant;

mod basic;
mod basic_slow;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    BasicSlow(basic_slow::Args),
    Basic(basic::Args),
}

fn main() {
    let start_time = Instant::now();

    let cli = Cli::parse();
    let result = match &cli.command {
        Commands::BasicSlow(args) => basic_slow::run(args),
        Commands::Basic(args) => basic::run(args),
    };
    if let Err(e) = result {
        eprintln!("Error generating image: {e}");
        std::process::exit(1)
    }

    let duration = start_time.elapsed();
    println!("Completed in {:.3?}", duration);
}
