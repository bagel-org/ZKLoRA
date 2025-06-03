fn main() {
    // Ensure we're using nightly for certain features
    println!("cargo:rustc-env=RUSTFLAGS=--cfg=nightly");
    
    // Link against system libraries if needed
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dylib=stdc++");
    
    // Rebuild if any of these files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/quantization.rs");
    println!("cargo:rerun-if-changed=build.rs");
} 