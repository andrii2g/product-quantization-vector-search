fn main() {
    if let Err(error) = product_quantization_vector_search::cli::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
