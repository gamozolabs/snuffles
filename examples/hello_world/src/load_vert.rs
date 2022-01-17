//! Vertex data parser

use std::fs::File;
use std::io::{Read, BufReader};
use std::f32::consts::PI;
use std::path::Path;
use flate2::read::GzDecoder;
use snuffles::Vertex;
use snuffles::cgmath::InnerSpace;

/// Load vertex data from disk
pub fn load_vert<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Vertex>> {
    // Open the file
    let mut fd = BufReader::new(File::open(path)?);

    // Create the vertex and triangle buffers
    let mut vertices = Vec::new();

    // Get the number of vertices
    let mut num_vertices = [0u8; 8];
    fd.read_exact(&mut num_vertices)?;
    let num_vertices = u64::from_le_bytes(num_vertices);

    // Load the vertex data
    for _ in 0..num_vertices {
        let mut x = [0u8; 4];
        fd.read_exact(&mut x)?;
        let x = f32::from_le_bytes(x);

        let mut y = [0u8; 4];
        fd.read_exact(&mut y)?;
        let y = f32::from_le_bytes(y);

        let mut z = [0u8; 4];
        fd.read_exact(&mut z)?;
        let z = f32::from_le_bytes(z);

        // Save the vertex data
        vertices.push((x, y, z));
    }

    // Get the number of triangles
    let mut num_triangles = [0u8; 8];
    fd.read_exact(&mut num_triangles)?;
    let num_triangles = u64::from_le_bytes(num_triangles);

    // Load the triangle data
    let mut vert = Vec::new();
    for _ in 0..num_triangles {
        let mut a = [0u8; 4];
        fd.read_exact(&mut a)?;
        let a = u32::from_le_bytes(a);

        let mut b = [0u8; 4];
        fd.read_exact(&mut b)?;
        let b = u32::from_le_bytes(b);

        let mut c = [0u8; 4];
        fd.read_exact(&mut c)?;
        let c = u32::from_le_bytes(c);

        // Save the triangle data
        for x in [a, b, c] {
            vert.push(
                Vertex::new(
                    vertices[x as usize].0,
                    vertices[x as usize].1,
                    vertices[x as usize].2,
                    0, 0, 0,
                )
            );
        }
    }

    Ok(vert)
}

/// Load colored vertex data from disk
pub fn load_vert_color<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Vertex>> {
    // Open the file
    let mut fd = GzDecoder::new(BufReader::new(File::open(path)?));

    // Get the number of entries
    let entries = { let mut x = [0; 8]; fd.read_exact(&mut x)?; u64::from_le_bytes(x) };

    // Load vertex data
    let mut vert = Vec::new();
    for _ in 0..entries {
        // Parse the data
        let x = { let mut x = [0; 4]; fd.read_exact(&mut x)?; f32::from_le_bytes(x) };
        let mut y = { let mut x = [0; 4]; fd.read_exact(&mut x)?; f32::from_le_bytes(x) };
        let z = { let mut x = [0; 4]; fd.read_exact(&mut x)?; f32::from_le_bytes(x) };
        let mut rgb = [0; 3];
        fd.read_exact(&mut rgb)?;

        // Move players a bit up from the ground
        y += 5.0;

        // Save the vertex
        vert.push(Vertex::new(x, y, z, rgb[0], rgb[1], rgb[2]));
    }

    Ok(vert)
}

/// Load time data from disk
pub fn load_time<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<(u32, u32, f32, u64)>> {
    // Open the file
    let mut fd = GzDecoder::new(BufReader::new(File::open(path)?));

    // Get the number of entries
    let entries = { let mut x = [0; 8]; fd.read_exact(&mut x)?; u64::from_le_bytes(x) };

    // Load time data
    let mut times = Vec::new();
    for _ in 0..entries {
        // Parse the data
        let uid    = { let mut x = [0; 4]; fd.read_exact(&mut x)?; u32::from_le_bytes(x) };
        let npc_id = { let mut x = [0; 4]; fd.read_exact(&mut x)?; u32::from_le_bytes(x) };
        let facing = { let mut x = [0; 4]; fd.read_exact(&mut x)?; f32::from_le_bytes(x) };
        let time   = { let mut x = [0; 8]; fd.read_exact(&mut x)?; u64::from_le_bytes(x) };

        // Save the time
        times.push((uid, npc_id, facing, time));
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

        v0.g = 0; v0.b = 0; v0.r = 0;
        v1.g = 0; v1.b = 0; v1.r = 0;
        v2.g = 0; v2.b = 0; v2.r = 0;

        // Update color for this triangle
        v0.g = (ny * 255. / PI) as u8;
        v1.g = (ny * 255. / PI) as u8;
        v2.g = (ny * 255. / PI) as u8;
    }
}

