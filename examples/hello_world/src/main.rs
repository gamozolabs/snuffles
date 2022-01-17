#![feature(array_chunks)]

mod load_vert;

use std::rc::Rc;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicU32, Ordering};
use std::collections::HashMap;
use snuffles::{Persist, Buffer, Msaa, Vsync, Vertex};
use snuffles::{Window, EventHandler, CameraMode, DrawCommand, RedrawTrigger};
use snuffles::cgmath::{Deg, point3};

/// If benchmarking is enabled, vsync is disabled and frames are redrawn
/// without waiting for a redraw request
const BENCHMARK_MODE: bool = false;

#[derive(Default)]
struct Timeline {
    /// Line head index (what has been drawn up to)
    head: AtomicU32,

    /// Line tail index (what should be drawn to)
    tail: AtomicU32,

    most_recent: Mutex<Vec<Vertex>>,
}

/// Worker thread for blocking incremental updates to the line
fn player_worker(timeline: Arc<Timeline>, redraw_trigger: RedrawTrigger) {
    let times = load_vert::load_time("movement.time.gz").unwrap();
    let lines = load_vert::load_vert_color("movement.vert.gz").unwrap();

    // Starting time
    let start_time = times[0].3;
    let it = Instant::now();

    let mut old = 0;
    let mut new = 0;

    // Speedup to replay
    const TIME_SPEEDUP: f64 = 50.;

    let mut most_recent = HashMap::new();

    // This is really gross, but basically we find the lines which should be
    // drawn for a given time range. We stop rendering old lines, and add
    // rendering of new lines
    loop {
        // Compute the target time
        let target_time = start_time +
            (it.elapsed().as_nanos() as f64 * TIME_SPEEDUP) as u64;

        // Seek as far as we can
        loop {
            let uid = times[new].0;
            most_recent.insert(uid, (lines[new * 2 + 1], times[new].2, times[new].1, times[new].3));

            // If our target time is past this time display it
            if target_time > times[new].3 {
                new += 1;
                continue;
            }

            // If our the old entry has expired, remove it (5 seconds max)
            if (target_time - 1500_000_000_000) > times[old].3 {
                old += 1;
                continue;
            }

            break;
        }

        // Produce more data
        timeline.head.store(old as u32 * 2, Ordering::Relaxed);
        timeline.tail.store(new as u32 * 2, Ordering::Relaxed);

        // Update most recent data
        {
            let mut mr = timeline.most_recent.lock().unwrap();
            mr.clear();
            for (_uid, &(vertex, facing, npc_id, age)) in most_recent.iter() {
                // Create a directional arrow triangle around vertex

                let facing = facing;

                // Skip aged mobs
                if (target_time - 1500_000_000_000) > age {
                    continue;
                }

                let scale = 10.0;

                let xo = facing.sin() * scale;
                let yo = facing.cos() * scale;
                let mut vpt = vertex; vpt.pos.x += xo; vpt.pos.z += yo;
                mr.push(vpt);

                let xo = (facing + 5. * PI / 6.).sin() * scale;
                let yo = (facing + 5. * PI / 6.).cos() * scale;
                let mut vpt = vertex; vpt.pos.x += xo; vpt.pos.z += yo;
                mr.push(vpt);

                let xo = (facing - 5. * PI / 6.).sin() * scale;
                let yo = (facing - 5. * PI / 6.).cos() * scale;
                let mut vpt = vertex; vpt.pos.x += xo; vpt.pos.z += yo;
                mr.push(vpt);
            }
        }

        // Redraw
        redraw_trigger.request_redraw(true).unwrap();

        // Sleepy time
        std::thread::sleep(Duration::from_millis(10));
    }
}

struct Handler {
    /// Start time
    start_time: Instant,

    /// Last report time of FPS
    last_report: Instant,

    /// Number of frames
    frames: u64,

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
        let mut map = load_vert::load_vert("Kalimdor.falkvbo")
            .expect("Failed to load Molten Core map");
        load_vert::color_by_normal(&mut map);

        // Get the movement vertex data
        let movement = load_vert::load_vert_color("movement.vert.gz")
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
            point3(
                -6140.2725,
                1109.4958,
                2518.2297,
            ),
            Deg(-89.), Deg(0.));

        Self {
            map_command: DrawCommand::Triangles(
                Persist::Yes, map_buffer, 0..map.len() as u32),
            line_buffer,
            timeline,
            frames: 0,
            start_time: Instant::now(),
            last_report: Instant::now(),
        }
    }

    fn should_redraw(&mut self, window: &mut Window<Self>) {
        // Always request an incremental redraw for FPS benchmark
        if BENCHMARK_MODE {
            window.request_redraw(true);
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

        {
            // Create triangle data and send it to the GPU
            let mr = self.timeline.most_recent.lock().unwrap();
            let point_buffer = window.create_vertex_buffer(mr.as_slice());
            window.push_command(
                DrawCommand::Triangles(Persist::No,
                    point_buffer, 0..mr.len() as u32));
        }

        self.frames += 1;
        if self.last_report.elapsed() >= Duration::from_millis(2000) {
            println!("FPS: {:10.4}",
                self.frames as f64 / self.start_time.elapsed().as_secs_f64());
            self.last_report = Instant::now();
        }
    }
}

fn main() {
    Window::<Handler>::new("Hello world", 1920, 1080, Msaa::X4,
        if BENCHMARK_MODE { Vsync::Off } else { Vsync::On })
        .expect("Failed to create window")
        .camera_mode(CameraMode::Flight3d)
        .run();
}

