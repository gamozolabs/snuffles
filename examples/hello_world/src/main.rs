#![feature(array_chunks)]

mod load_vert;

use snuffles::{Window, EventHandler, CameraMode, DrawCommand};

struct Handler {
    /// List of active draw commands
    draw_command: Vec<DrawCommand>,
}

impl EventHandler for Handler {
    fn create(window: &mut Window<Self>) -> Self {
        // Get the raw vertex data
        let map = load_vert::load_vert("MoltenCore.vert.xz")
            .expect("Failed to load Molten Core map");

        // Send the vertex data to the GPU
        let map_buffer = window.create_vertex_buffer(&map);

        // Create the renderer
        Self {
            draw_command: vec![
                DrawCommand::Triangles(map_buffer, 0..map.len() as u32)
            ],
        }
    }

    // Dispatch list of draw commands
    fn render(&mut self) -> &[DrawCommand] {
        // Just return the draw command slice!
        self.draw_command.as_slice()
    }
}

fn main() {
    Window::<Handler>::new("Hello world", 800, 600)
        .expect("Failed to create window")
        .camera_mode(CameraMode::Flight3d)
        .msaa(4)
        .run();
}

