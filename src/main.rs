use clap::command;
use std::time::Instant;

mod basic;

// derive pattern: can't store args in basic.rs and pass it over
// builder pattern: can't pass args to the other side

fn main() {
    let start_time = Instant::now();

    {
        let matches = command!()
            .subcommand_required(true)
            .subcommand(basic::command())
            .arg_required_else_help(true)
            .get_matches();

        if let Err(e) = match &matches.subcommand() {
            Some((basic::NAME, args)) => basic::run(args),
            _ => unreachable!("Exhausted list of subcommands"),
        } {
            eprintln!("Error generating image: {e}");
            std::process::exit(1)
        }
    }

    let duration = start_time.elapsed();
    println!("Completed in {:.3?}", duration);
}
