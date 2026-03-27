use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const GGUF_MAGIC: u32    = 0x46554747;
const SUPPORTED_VERSIONS: [u32; 4] = [3, 2, 1, 0];

// ── Types ────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
#[derive(Debug)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct TensorInfo {
    pub name:   String,
    pub dims:   Vec<u64>,
    pub kind:   u32,
    pub offset: u64,
}

// ── Primitives ───────────────────────────────────────────────────────────────

fn read_u32(file: &mut File) -> Option<u32> {
    let mut b = [0u8; 4];
    file.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

fn read_u64(file: &mut File) -> Option<u64> {
    let mut b = [0u8; 8];
    file.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

fn read_string(file: &mut File) -> Option<String> {
    let len = read_u64(file)? as usize;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn read_value(file: &mut File, value_type: u32) -> Option<GgufValue> {
    match value_type {
        0  => Some(GgufValue::U8(  { let mut b = [0u8; 1]; file.read_exact(&mut b).ok()?; b[0] })),
        1  => Some(GgufValue::I8(  { let mut b = [0u8; 1]; file.read_exact(&mut b).ok()?; b[0] as i8 })),
        2  => Some(GgufValue::U16( { let mut b = [0u8; 2]; file.read_exact(&mut b).ok()?; u16::from_le_bytes(b) })),
        3  => Some(GgufValue::I16( { let mut b = [0u8; 2]; file.read_exact(&mut b).ok()?; i16::from_le_bytes(b) })),
        4  => Some(GgufValue::U32( { let mut b = [0u8; 4]; file.read_exact(&mut b).ok()?; u32::from_le_bytes(b) })),
        5  => Some(GgufValue::I32( { let mut b = [0u8; 4]; file.read_exact(&mut b).ok()?; i32::from_le_bytes(b) })),
        6  => Some(GgufValue::F32( { let mut b = [0u8; 4]; file.read_exact(&mut b).ok()?; f32::from_le_bytes(b) })),
        7  => Some(GgufValue::Bool({ let mut b = [0u8; 1]; file.read_exact(&mut b).ok()?; b[0] != 0 })),
        8  => Some(GgufValue::String(read_string(file)?)),
        9  => {
            let elem_type = read_u32(file)?;
            let count     = read_u64(file)? as usize;
            let mut array = Vec::with_capacity(count);
            for _ in 0..count {
                array.push(read_value(file, elem_type)?);
            }
            Some(GgufValue::Array(array))
        }
        10 => Some(GgufValue::U64( { let mut b = [0u8; 8]; file.read_exact(&mut b).ok()?; u64::from_le_bytes(b) })),
        11 => Some(GgufValue::I64( { let mut b = [0u8; 8]; file.read_exact(&mut b).ok()?; i64::from_le_bytes(b) })),
        12 => Some(GgufValue::F64( { let mut b = [0u8; 8]; file.read_exact(&mut b).ok()?; f64::from_le_bytes(b) })),
        _  => None,
    }
}

fn ggml_type_size(kind: u32, n_elements: u64) -> Option<usize> {
    match kind {
        0  => Some((n_elements * 4) as usize),
        1  => Some((n_elements * 2) as usize),
        2  => Some((n_elements / 32  * 18)  as usize), // Q4_0
        6  => Some((n_elements / 32  * 34)  as usize), // Q8_0
        12 => Some((n_elements / 256 * 144) as usize), // Q4_K
        15 => Some((n_elements / 256 * 176) as usize), // Q6_K
        _  => None,
    }
}

// ── Sections (in file order) ──────────────────────────────────────────────────

pub fn magic(file: &mut File) -> bool {
    let mut buf = [0u8; 8];
    if file.read_exact(&mut buf).is_err() {
        return false;
    }
    let magic   = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    magic == GGUF_MAGIC && SUPPORTED_VERSIONS.contains(&version)
}

pub fn header(file: &mut File) -> Option<(u64, u64)> {
    let tensor_count = read_u64(file)?;
    let kv_count     = read_u64(file)?;
    Some((tensor_count, kv_count))
}

pub fn read_metadata(file: &mut File, kv_count: u64) -> Option<Vec<(String, GgufValue)>> {
    let mut pairs = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let key        = read_string(file)?;
        let value_type = read_u32(file)?;
        let value      = read_value(file, value_type)?;
        pairs.push((key, value));
    }
    Some(pairs)
}

pub fn tensor_info(file: &mut File, tensor_count: u64) -> Option<Vec<TensorInfo>> {
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name   = read_string(file)?;
        let n_dims = read_u32(file)?;
        let dims   = (0..n_dims).map(|_| read_u64(file)).collect::<Option<Vec<_>>>()?;
        let kind   = read_u32(file)?;
        let offset = read_u64(file)?;
        tensors.push(TensorInfo { name, dims, kind, offset });
    }
    Some(tensors)
}

pub fn tensor_data(file: &mut File, tensors: &[TensorInfo], alignment: u64) -> Option<Vec<Vec<u8>>> {
    let pos       = file.seek(SeekFrom::Current(0)).ok()?;
    let remainder = pos % alignment;
    if remainder != 0 {
        file.seek(SeekFrom::Current((alignment - remainder) as i64)).ok()?;
    }

    let data_start = file.seek(SeekFrom::Current(0)).ok()?;
    let mut result = Vec::with_capacity(tensors.len());

    for tensor in tensors {
        let n_elements: u64 = tensor.dims.iter().product();
        let n_bytes         = ggml_type_size(tensor.kind, n_elements)?;
        file.seek(SeekFrom::Start(data_start + tensor.offset)).ok()?;
        let mut bytes = vec![0u8; n_bytes];
        file.read_exact(&mut bytes).ok()?;
        result.push(bytes);
    }

    Some(result)
}