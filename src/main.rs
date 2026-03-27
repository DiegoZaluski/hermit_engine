mod gguf;

use gguf::extract_gguf;
use gguf::extract_gguf::GgufValue;
use std::fs::File;

fn main() {
    let mut file = File::open("src/models/qwen2.5-0.5b-instruct-q4_k_m.gguf").unwrap();

    if !extract_gguf::magic(&mut file) {
        println!("not a GGUF file");
        return;
    }

    println!("is GGUF");

    let (tensor_count, kv_count) = extract_gguf::header(&mut file).unwrap();

    let metadata = extract_gguf::read_metadata(&mut file, kv_count).unwrap();

    for (key, value) in &metadata {
        match value {
            GgufValue::Array(arr) => println!("{}: [array with {} elements]", key, arr.len()),
            _                     => println!("{}: {:?}", key, value),
        }
    }

    let alignment = metadata.iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| if let GgufValue::U32(n) = v { Some(*n as u64) } else { None })
        .unwrap_or(32);

    let tensor_info = extract_gguf::tensor_info(&mut file, tensor_count).unwrap();
    let tensor_data = extract_gguf::tensor_data(&mut file, &tensor_info, alignment).unwrap();

    println!("{:?}", tensor_info);
    println!("{:?}", tensor_data);
}
