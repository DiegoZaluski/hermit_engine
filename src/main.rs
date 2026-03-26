mod gguf;
use gguf::read_header::read_header;
use std::fs::File;

fn main() {
    let mut file = File::open("src/models/qwen2.5-0.5b-instruct-q4_k_m.gguf").unwrap();
    match read_header(&mut file) {
        Ok(header) => {
            println!("{:?}", header);
        }
        Err(e) => {
            eprintln!("Error reading GGUF header: {}", e);
        }
    }
}
