// Todo: Move this build script to admin.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    vergen::EmitBuilder::builder()
        .all_build()
        .all_git()
        .emit()?;
    Ok(())
}
