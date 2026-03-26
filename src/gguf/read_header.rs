use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub fn read_header(file: &mut File) -> Result<[u32; 5], std::io::Error> {
    let magic = 0x46554747u32;
    let supported_versions = [3u32, 2u32, 1u32, 0u32];
    
    file.seek(SeekFrom::Start(0))?;
    
    let mut buffer = [0u8; 24];
    file.read_exact(&mut buffer)?;
    
    let file_magic = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
    if file_magic != magic {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, 
            format!("Invalid GGUF magic number: expected 0x{:08x}, got 0x{:08x}", magic, file_magic)
        ));
    }
    
    let version = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
    if !supported_versions.contains(&version) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported GGUF version: {}", version)
        ));
    }
    
    let tensor_count = u32::from_le_bytes(buffer[8..12].try_into().unwrap());
    let kv_count = u32::from_le_bytes(buffer[16..20].try_into().unwrap());
    let header_size = 24u32;
    
    Ok([file_magic, version, tensor_count, kv_count, header_size])
}