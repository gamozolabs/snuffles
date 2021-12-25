#![feature(array_chunks)]

mod load_vert;

use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use std::sync::atomic::{AtomicU32, Ordering};
use snuffles::{Persist, Buffer};
use snuffles::{Window, EventHandler, CameraMode, DrawCommand, RedrawTrigger};
use snuffles::cgmath::{Deg, point3};

#[derive(Default)]
struct Timeline {
    /// Line head index (what has been drawn up to)
    head: AtomicU32,

    /// Line tail index (what should be drawn to)
    tail: AtomicU32,
}

/// Worker thread for blocking incremental updates to the line
fn player_worker(timeline: Arc<Timeline>, redraw_trigger: RedrawTrigger) {
    let times = load_vert::load_time("movement.time.xz").unwrap();

    // Starting time
    let start_time = times[0];
    let it = Instant::now();

    let mut old = 0;
    let mut new = 0;

    // Speedup to replay
    const TIME_SPEEDUP: f64 = 1.;

    // This is really gross, but basically we find the lines which should be
    // drawn for a given time range. We stop rendering old lines, and add
    // rendering of new lines
    loop {
        // Compute the target time
        let target_time = start_time +
            (it.elapsed().as_nanos() as f64 * TIME_SPEEDUP) as u64;

        // Seek as far as we can
        loop {
            // If our target time is past this time display it
            if target_time > times[new] {
                new += 1;
                continue;
            }

            // If our the old entry has expired, remove it (5 seconds max)
            if (target_time - 5_000_000_000) > times[old] {
                old += 1;
                continue;
            }

            break;
        }

        // Produce more data
        timeline.head.store(old as u32 * 2, Ordering::Relaxed);
        timeline.tail.store(new as u32 * 2, Ordering::Relaxed);

        // Redraw
        redraw_trigger.request_redraw(false).unwrap();

        // Sleepy time
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

struct Handler {
    /// Command to issue to redraw the map
    map_command: DrawCommand,

    /// Line buffer GPU storage
    line_buffer: Rc<Buffer>,

    /// State of overlay lines to draw
    timeline: Arc<Timeline>,
}

impl EventHandler for Handler {
    fn create(window: &mut Window<Self>) -> Self {
        // Get the raw vertex data
        let mut map = load_vert::load_vert("MoltenCore.vert.xz")
            .expect("Failed to load Molten Core map");
        load_vert::color_by_normal(&mut map);

        // Get the movement vertex data
        let movement = load_vert::load_vert("movement.vert.xz")
            .expect("Failed to load Molten Core movement");

        // Create the timeline
        let timeline = Arc::new(Timeline::default());

        // Create a thread for handling the player movements and give it a
        // redraw trigger
        {
            let tm = timeline.clone();
            let rt = window.redraw_trigger();
            std::thread::spawn(move || player_worker(tm, rt));
        }

        // Send the vertex data to the GPU
        let map_buffer = window.create_vertex_buffer(&map);

        // Send line data to the GPU
        let line_buffer = window.create_vertex_buffer(&movement);

        // Set a reasonable camera to start
        window.update_camera(
            point3(-776.2651, -173.64767, 581.6539),
            Deg(-20.6), Deg(-25.6));

        Self {
            map_command: DrawCommand::Triangles(
                Persist::Yes, map_buffer, 0..map.len() as u32),
            line_buffer,
            timeline,
        }
    }

    // Fill a list of draw commands
    fn render(&mut self, window: &mut Window<Self>, incremental: bool) {
        // Render the whole scene if we need to
        if !incremental {
            window.push_command(self.map_command.clone());
        }

        // Issue a line draw command
        let head = self.timeline.head.load(Ordering::Relaxed);
        let tail = self.timeline.tail.load(Ordering::Relaxed);
        window.push_command(
            DrawCommand::Lines(Persist::No,
                self.line_buffer.clone(), head..tail));
    }
}

fn main() {
    Window::<Handler>::new("Hello world", 1440, 900, 4)
        .expect("Failed to create window")
        .camera_mode(CameraMode::Flight3d)
        .run();
}

