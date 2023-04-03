use std::error::Error;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(target_os = "macos") {
        build_mac_dependencies()?
    }

    tonic_build::configure()
        .build_server(false)
        .compile(&["src/proto/vector_service.proto"], &["src/proto/"])?;

    Ok(())
}

fn build_mac_dependencies() -> Result<(), Box<dyn Error>> {
    // Install dependencies with homebrew
    let res = Command::new("brew").args(["install", "protobuf"]).output();

    match res {
        Ok(output) => {
            if output.status.success() {
                println!("Installed dependencies with Homebrew");
            } else {
                println!(
                    "Failed to install dependencies with Homebrew: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
        Err(e) => return Err(format!("Failed to install dependencies with Homebrew: {e}").into()),
    }

    Ok(())
}
