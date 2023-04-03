use std::error::Error;
use std::path::Path;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    let path = Command::new("cargo").args(["locate-project", "--workspace", "--message-format", "plain"]).output()?;
    let path = String::from_utf8( path.stdout)?;
    let path = Path::new(&path).parent().unwrap();
    let output = Command::new("make").current_dir(path).args(["generate-index-service"]).output()?;

    //If a previous auto-generated output already exists, don't error out.
    //This is needed for CI build on macOS and windows where `docker` is expected to fail
    if !output.status.success() {
        if path.join("index_service").join("src").join("apis").join("index_operations_api.rs").exists() {
            eprintln!("Warning - failed to generate OpenAPI: {output:?}. Found existing index_operations_api.rs. Continuing anyway...");
        } else {
            eprintln!("Failed to generate OpenAPI: {output:?}");
            return Err("process failed".into());
        }
    }
    Ok(())
}