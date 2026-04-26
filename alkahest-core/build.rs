fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // On macOS with Homebrew (especially Apple Silicon / M1), libraries live
    // under /opt/homebrew rather than /usr/local.  Ask brew for the FLINT
    // prefix and emit the search path so rust-lld can find libflint.dylib.
    if cfg!(target_os = "macos") {
        if let Ok(out) = std::process::Command::new("brew")
            .args(["--prefix", "flint"])
            .output()
        {
            if out.status.success() {
                let prefix = String::from_utf8_lossy(&out.stdout).trim().to_string();
                println!("cargo:rustc-link-search=native={prefix}/lib");
            }
        }
    }

    println!("cargo:rustc-link-lib=flint");
}
