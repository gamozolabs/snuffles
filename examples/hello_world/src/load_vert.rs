//! Vertex data parser

use std::fs::File;
use std::io::{Read, BufReader};
use std::f32::consts::PI;
use std::path::Path;
use xz2::read::XzDecoder;
use snuffles::Vertex;
use snuffles::cgmath::InnerSpace;

/// Load vertex data from disk
pub fn load_vert<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Vertex>> {
    // Open the file
    let mut fd = XzDecoder::new(BufReader::new(File::open(path)?));

    // Get the number of entries
    let entries = { let mut x = [0; 8]; fd.read_exact(&mut x)?; u64::from_le_bytes(x) };

    // Load vertex data
    let mut vert = Vec::new();
    for _ in 0..entries {
        // Parse the data
        let x = { let mut x = [0; 4]; fd.read_exact(&mut x)?; f32::from_le_bytes(x) };
        let y = { let mut x = [0; 4]; fd.read_exact(&mut x)?; f32::from_le_bytes(x) };
        let z = { let mut x = [0; 4]; fd.read_exact(&mut x)?; f32::from_le_bytes(x) };

        // Save the vertex
        vert.push(Vertex::new(x, y, z, 0, 0, 128));
    }

    Ok(vert)
}

/// Load time data from disk
pub fn load_time<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<u64>> {
    // Open the file
    let mut fd = XzDecoder::new(BufReader::new(File::open(path)?));

    // Get the number of entries
    let entries = { let mut x = [0; 8]; fd.read_exact(&mut x)?; u64::from_le_bytes(x) };

    // Load time data
    let mut times = Vec::new();
    for _ in 0..entries {
        // Parse the data
        let time = { let mut x = [0; 8]; fd.read_exact(&mut x)?; u64::from_le_bytes(x) };

        // Save the time
        times.push(time);
    }

    Ok(times)
}

/// Recolor a triangle by its normal
pub fn color_by_normal(vertex_data: &mut [Vertex]) {
    // Update color based on slope of triangle (normal)
    for [v0, v1, v2] in vertex_data.array_chunks_mut::<3>() {
        // Compute normal WRT `y`
        let u = v0.pos - v1.pos;
        let v = v2.pos - v1.pos;
        let ny = u.cross(v).normalize().y.acos();

        // Update color for this triangle
        v0.r = (ny * 255. / PI) as u8;
        v1.r = (ny * 255. / PI) as u8;
        v2.r = (ny * 255. / PI) as u8;

        v0.g = 0; v0.b = 0;
        v1.g = 0; v1.b = 0;
        v2.g = 0; v2.b = 0;
    }
}

