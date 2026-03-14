use cgmath::{perspective, point3, Angle, Deg, Matrix4, Point3, Vector3};
use cgmath::{Vector2, Vector4};
use std::borrow::Cow;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window as WinitWindow;

pub use cgmath;
pub use wgpu::*;
pub use winit;
pub use winit::keyboard::KeyCode as VirtualKeyCode;

const FONT: Font = Font::Font9x16;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Font {
    Font4x6,
    Font6x8,
    Font9x16,
    Font24x36,
    Font48x72,
}

impl Font {
    const fn width(self) -> usize {
        match self {
            Font::Font4x6 => 4,
            Font::Font6x8 => 6,
            Font::Font9x16 => 9,
            Font::Font24x36 => 24,
            Font::Font48x72 => 48,
        }
    }
    const fn height(self) -> usize {
        match self {
            Font::Font4x6 => 6,
            Font::Font6x8 => 8,
            Font::Font9x16 => 16,
            Font::Font24x36 => 36,
            Font::Font48x72 => 72,
        }
    }
}

type UserEvent = bool;
type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    CreateWindow(winit::error::OsError),
    GetAdapter,
    CreateDevice(RequestDeviceError),
    PreferredFormat,
    GetSurfaceTexture(SurfaceError),
    SendEvent(winit::event_loop::EventLoopClosed<UserEvent>),
}

pub enum CameraMode {
    None,
    Pannable2d,
    Flight3d,
}

#[derive(Debug)]
enum CameraState {
    None,
    Pannable2d {
        panning: bool,
        x: f32,
        y: f32,
        zoom: f32,
    },
    Flight3d {
        eye: Point3<f32>,
        pitch: Deg<f32>,
        yaw: Deg<f32>,
        speed: f32,
        panning: bool,
        key_w: bool,
        key_a: bool,
        key_s: bool,
        key_d: bool,
    },
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex {
    pub pos: Vector3<f32>,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    _padding: u8,
}

impl Vertex {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, r: u8, g: u8, b: u8) -> Self {
        Self {
            pos: Vector3::new(x, y, z),
            r,
            g,
            b,
            _padding: 0,
        }
    }
    #[inline]
    pub const fn color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.r = r;
        self.g = g;
        self.b = b;
        self
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TextureVertex {
    pub pos: Vector3<f32>,
    pub tex_coords: Vector2<f32>,
    pub color: Vector4<f32>,
}

impl TextureVertex {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, u: f32, v: f32, r: f32, g: f32, b: f32) -> Self {
        Self {
            pos: Vector3::new(x, y, z),
            tex_coords: Vector2::new(u, v),
            color: Vector4::new(r, g, b, 1.),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Persist {
    Yes,
    No,
}

#[derive(Clone)]
pub enum DrawCommand {
    Triangles(Persist, Rc<Buffer>, Range<u32>),
    Points(Persist, Rc<Buffer>, Range<u32>),
    Lines(Persist, Rc<Buffer>, Range<u32>),
}

impl DrawCommand {
    pub fn len(&self) -> usize {
        match self {
            DrawCommand::Triangles(_, _, r)
            | DrawCommand::Points(_, _, r)
            | DrawCommand::Lines(_, _, r) => (r.end - r.start) as usize,
        }
    }
}

pub trait EventHandler: Sized {
    fn create(window: &mut Window<Self>) -> Self;
    fn key_down(&mut self, _window: &mut Window<Self>, _key: KeyCode) {}
    fn key_up(&mut self, _window: &mut Window<Self>, _key: KeyCode) {}
    fn mouse_move(&mut self, _window: &mut Window<Self>, _dx: f64, _dy: f64) {}
    fn mouse_scroll(&mut self, _window: &mut Window<Self>, _delta_y: f32) {}
    fn mouse_down(&mut self, _window: &mut Window<Self>, _button: MouseButton) {}
    fn mouse_up(&mut self, _window: &mut Window<Self>, _button: MouseButton) {}
    fn should_redraw(&mut self, _window: &mut Window<Self>) {}
    fn render(&mut self, _window: &mut Window<Self>, _incremental: bool) {}
    fn render_ui(&mut self, _window: &mut Window<Self>) {}
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Msaa {
    None = 1,
    X2 = 2,
    X4 = 4,
}

#[derive(Clone, Copy)]
pub enum Vsync {
    Off,
    On,
}

pub struct RedrawTrigger(EventLoopProxy<UserEvent>);
impl RedrawTrigger {
    pub fn request_redraw(&self, incremental: bool) -> Result<()> {
        self.0.send_event(incremental).map_err(Error::SendEvent)
    }
}

pub struct Window<EH: EventHandler> {
    pub start: Instant,
    handler: Option<EH>,
    event_loop: Option<EventLoop<UserEvent>>,
    winit_window: Option<Arc<WinitWindow>>,
    pub width: u32,
    pub height: u32,
    device: Option<Device>,
    surface: Option<Surface<'static>>,
    queue: Option<Queue>,
    camera_state: CameraState,
    camera_bind_group: Option<BindGroup>,
    camera_buffer: Option<Buffer>,
    texture_bind_group: Option<BindGroup>,
    incremental: bool,
    triangle_pipeline: Option<RenderPipeline>,
    line_pipeline: Option<RenderPipeline>,
    point_pipeline: Option<RenderPipeline>,
    texture_pipeline: Option<RenderPipeline>,
    msaa_level: Msaa,
    swapchain_format: Option<TextureFormat>,
    vsync: Vsync,
    output_texture: Option<Texture>,
    output_view: Option<TextureView>,
    output_depth: Option<Texture>,
    output_depth_view: Option<TextureView>,
    scene_texture: Option<Texture>,
    scene_depth: Option<Texture>,
    pub znear: f32,
    camera_uniform: Matrix4<f32>,
    proxy: EventLoopProxy<UserEvent>,
    persist_tri_commands: Vec<(Rc<Buffer>, Range<u32>)>,
    tri_commands: Vec<(Rc<Buffer>, Range<u32>)>,
    persist_line_commands: Vec<(Rc<Buffer>, Range<u32>)>,
    line_commands: Vec<(Rc<Buffer>, Range<u32>)>,
    persist_point_commands: Vec<(Rc<Buffer>, Range<u32>)>,
    point_commands: Vec<(Rc<Buffer>, Range<u32>)>,
    text_commands: Vec<(Rc<Buffer>, Range<u32>)>,
    text_temp: Vec<TextureVertex>,
    title: String,
    init_camera_mode: CameraMode,
    last_mouse: Option<(f64, f64)>,
}

impl<EH: 'static + EventHandler> Window<EH> {
    pub fn new(
        title: impl AsRef<str>,
        width: u32,
        height: u32,
        msaa_level: Msaa,
        vsync: Vsync,
    ) -> Result<Self> {
        let event_loop: EventLoop<UserEvent> = EventLoop::with_user_event().build().unwrap();
        let proxy = event_loop.create_proxy();
        Ok(Self {
            start: Instant::now(),
            handler: None,
            event_loop: Some(event_loop),
            winit_window: None,
            width,
            height,
            device: None,
            surface: None,
            queue: None,
            camera_state: CameraState::None,
            camera_bind_group: None,
            camera_buffer: None,
            texture_bind_group: None,
            incremental: false,
            triangle_pipeline: None,
            line_pipeline: None,
            point_pipeline: None,
            texture_pipeline: None,
            msaa_level,
            swapchain_format: None,
            vsync,
            output_texture: None,
            output_view: None,
            output_depth: None,
            output_depth_view: None,
            scene_texture: None,
            scene_depth: None,
            znear: 1.,
            camera_uniform: [[0f32; 4]; 4].into(),
            proxy,
            persist_tri_commands: Vec::new(),
            tri_commands: Vec::new(),
            persist_line_commands: Vec::new(),
            line_commands: Vec::new(),
            persist_point_commands: Vec::new(),
            point_commands: Vec::new(),
            text_commands: Vec::new(),
            text_temp: Vec::new(),
            title: title.as_ref().to_string(),
            init_camera_mode: CameraMode::None,
            last_mouse: None,
        })
    }

    pub fn set_title(&self, title: impl AsRef<str>) {
        if let Some(w) = &self.winit_window {
            w.set_title(title.as_ref());
        }
    }

    pub fn camera_mode(mut self, mode: CameraMode) -> Self {
        self.init_camera_mode = mode;
        self
    }

    pub fn redraw_trigger(&self) -> RedrawTrigger {
        RedrawTrigger(self.proxy.clone())
    }

    pub fn camera_speed(&self) -> Option<f32> {
        match self.camera_state {
            CameraState::Flight3d { speed, .. } => Some(speed),
            _ => None,
        }
    }

    #[inline]
    pub fn push_command(&mut self, command: DrawCommand) {
        match command {
            DrawCommand::Triangles(Persist::Yes, b, r) => self.persist_tri_commands.push((b, r)),
            DrawCommand::Triangles(Persist::No, b, r) => self.tri_commands.push((b, r)),
            DrawCommand::Lines(Persist::Yes, b, r) => self.persist_line_commands.push((b, r)),
            DrawCommand::Lines(Persist::No, b, r) => self.line_commands.push((b, r)),
            DrawCommand::Points(Persist::Yes, b, r) => self.persist_point_commands.push((b, r)),
            DrawCommand::Points(Persist::No, b, r) => self.point_commands.push((b, r)),
        }
    }

    pub fn push_text(
        &mut self,
        mut x: f32,
        y: f32,
        r: f32,
        g: f32,
        b: f32,
        text: impl AsRef<[u8]>,
    ) {
        const GL_W: f32 = 1. / 16.;
        const GL_H: f32 = 1. / 16.;
        for &ch in text.as_ref() {
            let (x1, x2) = (x, x + FONT.width() as f32);
            let (y1, y2) = (y, y + FONT.height() as f32);
            let u1 = (ch % 16) as f32 * GL_W;
            let v1 = (ch / 16) as f32 * GL_H;
            let (u2, v2) = (u1 + GL_W, v1 + GL_H);
            self.text_temp
                .push(TextureVertex::new(x2, y2, 0., u2, v1, r, g, b));
            self.text_temp
                .push(TextureVertex::new(x1, y2, 0., u1, v1, r, g, b));
            self.text_temp
                .push(TextureVertex::new(x2, y1, 0., u2, v2, r, g, b));
            self.text_temp
                .push(TextureVertex::new(x2, y1, 0., u2, v2, r, g, b));
            self.text_temp
                .push(TextureVertex::new(x1, y2, 0., u1, v1, r, g, b));
            self.text_temp
                .push(TextureVertex::new(x1, y1, 0., u1, v2, r, g, b));
            x += FONT.width() as f32;
        }
    }

    pub fn set_ui_camera(&mut self) {
        let m = cgmath::ortho(0., self.width as f32, 0., self.height as f32, -1., 1.);
        let b = unsafe { std::slice::from_raw_parts(std::ptr::addr_of!(m) as *const u8, 64) };
        if let (Some(q), Some(buf)) = (&self.queue, &self.camera_buffer) {
            q.write_buffer(buf, 0, b);
        }
    }

    pub fn screen_position(&self, x: f32, y: f32, z: f32) -> Option<(f32, f32)> {
        let t = self.camera_uniform * Vector4::new(x, y, z, 1.);
        let n = t / t.w;
        if t.w >= 0. {
            Some((
                (n.x + 1.) / 2. * self.width as f32,
                (n.y + 1.) / 2. * self.height as f32,
            ))
        } else {
            None
        }
    }

    pub fn set_camera_matrix(&mut self, matrix: &[f32; 16]) {
        let b = unsafe { std::slice::from_raw_parts(matrix.as_ptr() as *const u8, 64) };
        if let (Some(q), Some(buf)) = (&self.queue, &self.camera_buffer) {
            q.write_buffer(buf, 0, b);
        }
        let cols: &[[f32; 4]; 4] = unsafe { &*(matrix.as_ptr() as *const _) };
        self.camera_uniform = (*cols).into();
    }

    fn update_camera_int(&mut self) {
        let cu = match &self.camera_state {
            CameraState::None => return,
            CameraState::Pannable2d { x, y, zoom, .. } => cgmath::ortho(
                (*x - *zoom) * (self.width as f32 / self.height as f32),
                (*x + *zoom) * (self.width as f32 / self.height as f32),
                *y - *zoom,
                *y + *zoom,
                -1.,
                1.,
            ),
            CameraState::Flight3d {
                eye, pitch, yaw, ..
            } => {
                let d = Vector3::new(
                    pitch.cos() * yaw.sin(),
                    pitch.sin(),
                    pitch.cos() * yaw.cos(),
                );
                let v = Matrix4::look_to_rh(*eye, d, Vector3::unit_y());
                let p = perspective(
                    Deg(45.),
                    self.width as f32 / self.height as f32,
                    self.znear,
                    100000.,
                );
                self.camera_uniform = p * v;
                self.camera_uniform
            }
        };
        let b = unsafe { std::slice::from_raw_parts(std::ptr::addr_of!(cu) as *const u8, 64) };
        if let (Some(q), Some(buf)) = (&self.queue, &self.camera_buffer) {
            q.write_buffer(buf, 0, b);
        }
    }

    pub fn set_camera(&mut self, e: Point3<f32>, p: Deg<f32>, y: Deg<f32>) {
        if let CameraState::Flight3d {
            eye, pitch, yaw, ..
        } = &mut self.camera_state
        {
            *eye = e;
            *pitch = p;
            *yaw = y;
        }
        self.update_camera_int();
        self.request_redraw(false);
    }

    pub fn update_camera(&mut self) {
        self.update_camera_int();
        self.request_redraw(false);
    }

    pub fn request_redraw(&mut self, incremental: bool) {
        self.incremental &= incremental;
        if let Some(w) = &self.winit_window {
            w.request_redraw();
        }
    }

    pub fn client_area(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn create_vertex_buffer(&mut self, data: impl AsRef<[Vertex]>) -> Rc<Buffer> {
        let d = self.device.as_ref().unwrap();
        Rc::new(d.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: unsafe {
                std::slice::from_raw_parts(
                    data.as_ref().as_ptr() as *const u8,
                    std::mem::size_of_val(data.as_ref()),
                )
            },
            usage: BufferUsages::VERTEX,
        }))
    }

    fn create_texture_vertex_buffer(&mut self) -> Rc<Buffer> {
        let d = self.device.as_ref().unwrap();
        Rc::new(d.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: unsafe {
                std::slice::from_raw_parts(
                    self.text_temp.as_ptr() as *const u8,
                    std::mem::size_of_val(self.text_temp.as_slice()),
                )
            },
            usage: BufferUsages::VERTEX,
        }))
    }

    fn mk_pipeline(
        device: &Device,
        bgl: &BindGroupLayout,
        topology: PrimitiveTopology,
        shader: &ShaderModule,
        fmt: TextureFormat,
        msaa: u32,
    ) -> RenderPipeline {
        let pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bgl],
            immediate_size: 0,
        });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pl),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[
                        VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: VertexFormat::Float32x3,
                        },
                        VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: VertexFormat::Unorm8x4,
                        },
                    ],
                }],
            },
            primitive: PrimitiveState {
                topology,
                cull_mode: None,
                ..Default::default()
            },
            fragment: Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: fmt,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: msaa,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
            multiview_mask: None,
        })
    }

    fn mk_tex_pipeline(
        device: &Device,
        bgl: &BindGroupLayout,
        topology: PrimitiveTopology,
        shader: &ShaderModule,
        fmt: TextureFormat,
        msaa: u32,
    ) -> RenderPipeline {
        let pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bgl],
            immediate_size: 0,
        });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pl),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[VertexBufferLayout {
                    array_stride: std::mem::size_of::<TextureVertex>() as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[
                        VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: VertexFormat::Float32x3,
                        },
                        VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: VertexFormat::Float32x2,
                        },
                        VertexAttribute {
                            offset: 20,
                            shader_location: 2,
                            format: VertexFormat::Float32x4,
                        },
                    ],
                }],
            },
            primitive: PrimitiveState {
                topology,
                cull_mode: None,
                ..Default::default()
            },
            fragment: Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: fmt,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::default(),
                })],
            }),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: msaa,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
            multiview_mask: None,
        })
    }

    fn mk_tex(
        device: &Device,
        w: u32,
        h: u32,
        fmt: TextureFormat,
        usage: TextureUsages,
        msaa: u32,
    ) -> (Texture, TextureView) {
        let t = device.create_texture(&TextureDescriptor {
            label: None,
            mip_level_count: 1,
            sample_count: msaa,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            format: fmt,
            usage,
            view_formats: &[],
        });
        let v = t.create_view(&TextureViewDescriptor::default());
        (t, v)
    }

    fn mk_tex_pair(
        device: &Device,
        w: u32,
        h: u32,
        msaa: u32,
        fmt: TextureFormat,
    ) -> (Texture, TextureView, Texture, TextureView) {
        let u = TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::COPY_SRC;
        let (t, v) = Self::mk_tex(device, w, h, fmt, u, msaa);
        let (d, dv) = Self::mk_tex(device, w, h, TextureFormat::Depth32Float, u, msaa);
        (t, v, d, dv)
    }

    fn init_gpu(&mut self, window: Arc<WinitWindow>) {
        let sz = window.inner_size();
        self.width = sz.width;
        self.height = sz.height;
        eprintln!(
            "[{:14.6}] Window {}x{} scale={}",
            self.start.elapsed().as_secs_f64(),
            sz.width,
            sz.height,
            window.scale_factor()
        );

        let inst = Instance::new(&InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });
        let surface = inst.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(inst.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .expect("No adapter");
        eprintln!(
            "[{:14.6}] Renderer: {} ({:?})",
            self.start.elapsed().as_secs_f64(),
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: None,
            required_features: Features::empty(),
            required_limits: Limits::default(),
            memory_hints: Default::default(),
            trace: Default::default(),
            experimental_features: Default::default(),
        }))
        .expect("No device");

        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats[0];
        self.swapchain_format = Some(fmt);
        let pm = match self.vsync {
            Vsync::Off => {
                if caps.present_modes.contains(&PresentMode::Immediate) {
                    PresentMode::Immediate
                } else {
                    PresentMode::AutoNoVsync
                }
            }
            Vsync::On => PresentMode::Fifo,
        };
        surface.configure(
            &device,
            &SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST,
                format: fmt,
                width: self.width,
                height: self.height,
                present_mode: pm,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });
        let tex_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("texture_shader.wgsl"))),
        });

        let cam_buf = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: core::num::NonZeroU64::new(64),
                },
                count: None,
            }],
        });
        let cam_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &cam_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            }],
        });

        // Font
        let fb = match FONT {
            Font::Font4x6 => include_bytes!("VGA4x6.png").as_slice(),
            Font::Font6x8 => include_bytes!("VGA6x8.png").as_slice(),
            Font::Font9x16 => include_bytes!("VGA9x16.png").as_slice(),
            Font::Font24x36 => include_bytes!("VGA24x36.png").as_slice(),
            Font::Font48x72 => include_bytes!("VGA48x72.png").as_slice(),
        };
        let fi = image::load_from_memory(fb).unwrap();
        let fr = fi.to_rgba8().into_vec();
        use image::GenericImageView;
        let fd = fi.dimensions();
        let ft = device.create_texture(&TextureDescriptor {
            label: None,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: fd.0,
                height: fd.1,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let fv = ft.create_view(&TextureViewDescriptor::default());
        let fs = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let tex_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
                        sample_type: TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let tex_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &tex_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: cam_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&fv),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&fs),
                },
            ],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ft,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &fr,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * fd.0),
                rows_per_image: Some(fd.1),
            },
            Extent3d {
                width: fd.0,
                height: fd.1,
                depth_or_array_layers: 1,
            },
        );

        let ml = self.msaa_level as u32;
        let (ot, ov, od, odv) = Self::mk_tex_pair(&device, self.width, self.height, ml, fmt);
        let (st, _, sd, _) = Self::mk_tex_pair(&device, self.width, self.height, ml, fmt);
        let tp = Self::mk_pipeline(
            &device,
            &cam_bgl,
            PrimitiveTopology::TriangleList,
            &shader,
            fmt,
            ml,
        );
        let lp = Self::mk_pipeline(
            &device,
            &cam_bgl,
            PrimitiveTopology::LineList,
            &shader,
            fmt,
            ml,
        );
        let pp = Self::mk_pipeline(
            &device,
            &cam_bgl,
            PrimitiveTopology::PointList,
            &shader,
            fmt,
            ml,
        );
        let txp = Self::mk_tex_pipeline(
            &device,
            &tex_bgl,
            PrimitiveTopology::TriangleList,
            &tex_shader,
            fmt,
            ml,
        );

        self.winit_window = Some(window);
        self.device = Some(device);
        self.surface = Some(surface);
        self.queue = Some(queue);
        self.camera_buffer = Some(cam_buf);
        self.camera_bind_group = Some(cam_bg);
        self.texture_bind_group = Some(tex_bg);
        self.output_texture = Some(ot);
        self.output_view = Some(ov);
        self.output_depth = Some(od);
        self.output_depth_view = Some(odv);
        self.scene_texture = Some(st);
        self.scene_depth = Some(sd);
        self.triangle_pipeline = Some(tp);
        self.line_pipeline = Some(lp);
        self.point_pipeline = Some(pp);
        self.texture_pipeline = Some(txp);

        self.camera_state = match self.init_camera_mode {
            CameraMode::None => CameraState::None,
            CameraMode::Pannable2d => CameraState::Pannable2d {
                panning: false,
                x: 0.,
                y: 0.,
                zoom: 1.,
            },
            CameraMode::Flight3d => CameraState::Flight3d {
                eye: point3(0., 0., 0.),
                pitch: Deg(0.),
                yaw: Deg(0.),
                speed: 1.,
                panning: false,
                key_w: false,
                key_a: false,
                key_s: false,
                key_d: false,
            },
        };
        self.init_camera_mode = CameraMode::None;
        self.update_camera_int();
    }

    fn render_internal(&mut self, handler: &mut EH, ui: bool, frame: &mut SurfaceTexture) {
        self.text_temp.clear();
        if ui {
            handler.render_ui(self);
        } else {
            handler.render(self, self.incremental);
        }

        let tb = self.create_texture_vertex_buffer();
        self.text_commands
            .push((tb, 0..self.text_temp.len() as u32));

        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let cam_bg = self.camera_bind_group.as_ref().unwrap();
        let tex_bg = self.texture_bind_group.as_ref().unwrap();
        let msaa = self.msaa_level != Msaa::None;
        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let sz = Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        if !ui && self.incremental {
            let dst = if msaa {
                self.output_texture.as_ref().unwrap().as_image_copy()
            } else {
                frame.texture.as_image_copy()
            };
            enc.copy_texture_to_texture(
                self.scene_texture.as_ref().unwrap().as_image_copy(),
                dst,
                sz,
            );
            enc.copy_texture_to_texture(
                self.scene_depth.as_ref().unwrap().as_image_copy(),
                self.output_depth.as_ref().unwrap().as_image_copy(),
                sz,
            );
        }
        if !ui && !self.incremental {
            let (cv, rv) = if msaa {
                (self.output_view.as_ref().unwrap(), Some(&view))
            } else {
                (&view, None)
            };
            {
                let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: cv,
                        resolve_target: rv,
                        ops: Operations {
                            load: LoadOp::Clear(Color::BLACK),
                            store: StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: self.output_depth_view.as_ref().unwrap(),
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(1.0),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_bind_group(0, cam_bg, &[]);
                rp.set_pipeline(self.triangle_pipeline.as_ref().unwrap());
                for (b, r) in &self.persist_tri_commands {
                    rp.set_vertex_buffer(0, b.slice(..));
                    rp.draw(r.clone(), 0..1);
                }
                rp.set_pipeline(self.line_pipeline.as_ref().unwrap());
                for (b, r) in &self.persist_line_commands {
                    rp.set_vertex_buffer(0, b.slice(..));
                    rp.draw(r.clone(), 0..1);
                }
                rp.set_pipeline(self.point_pipeline.as_ref().unwrap());
                for (b, r) in &self.persist_point_commands {
                    rp.set_vertex_buffer(0, b.slice(..));
                    rp.draw(r.clone(), 0..1);
                }
            }
            let src = if msaa {
                self.output_texture.as_ref().unwrap().as_image_copy()
            } else {
                frame.texture.as_image_copy()
            };
            enc.copy_texture_to_texture(
                src,
                self.scene_texture.as_ref().unwrap().as_image_copy(),
                sz,
            );
            enc.copy_texture_to_texture(
                self.output_depth.as_ref().unwrap().as_image_copy(),
                self.scene_depth.as_ref().unwrap().as_image_copy(),
                sz,
            );
        }
        {
            let (cv, rv) = if msaa {
                (self.output_view.as_ref().unwrap(), Some(&view))
            } else {
                (&view, None)
            };
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: cv,
                    resolve_target: rv,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: self.output_depth_view.as_ref().unwrap(),
                    depth_ops: Some(Operations {
                        load: if ui { LoadOp::Clear(1.0) } else { LoadOp::Load },
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_bind_group(0, cam_bg, &[]);
            rp.set_pipeline(self.triangle_pipeline.as_ref().unwrap());
            for (b, r) in &self.tri_commands {
                rp.set_vertex_buffer(0, b.slice(..));
                rp.draw(r.clone(), 0..1);
            }
            rp.set_pipeline(self.line_pipeline.as_ref().unwrap());
            for (b, r) in &self.line_commands {
                rp.set_vertex_buffer(0, b.slice(..));
                rp.draw(r.clone(), 0..1);
            }
            rp.set_pipeline(self.point_pipeline.as_ref().unwrap());
            for (b, r) in &self.point_commands {
                rp.set_vertex_buffer(0, b.slice(..));
                rp.draw(r.clone(), 0..1);
            }
            rp.set_bind_group(0, tex_bg, &[]);
            rp.set_pipeline(self.texture_pipeline.as_ref().unwrap());
            for (b, r) in &self.text_commands {
                rp.set_vertex_buffer(0, b.slice(..));
                rp.draw(r.clone(), 0..1);
            }
        }
        queue.submit(Some(enc.finish()));
        self.persist_tri_commands.clear();
        self.tri_commands.clear();
        self.persist_line_commands.clear();
        self.line_commands.clear();
        self.persist_point_commands.clear();
        self.point_commands.clear();
        self.text_commands.clear();
        self.incremental = true;
    }

    pub fn run(mut self) -> ! {
        let el = self.event_loop.take().expect("EventLoop consumed");
        let mut app = AppHandler { window: self };
        el.run_app(&mut app).unwrap();
        std::process::exit(0);
    }
}

struct AppHandler<EH: EventHandler> {
    window: Window<EH>,
}

impl<EH: 'static + EventHandler> ApplicationHandler<UserEvent> for AppHandler<EH> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.winit_window.is_some() {
            return;
        }
        let attrs = WinitWindow::default_attributes()
            .with_inner_size(LogicalSize::new(self.window.width, self.window.height))
            .with_title(&self.window.title);
        let w = Arc::new(event_loop.create_window(attrs).expect("window"));
        self.window.init_gpu(w);
        self.window.handler = Some(EH::create(&mut self.window));
        eprintln!(
            "[{:14.6}] Window ready",
            self.window.start.elapsed().as_secs_f64()
        );
    }

    fn user_event(&mut self, _: &ActiveEventLoop, inc: bool) {
        self.window.request_redraw(inc);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let CameraState::Flight3d {
                    key_w,
                    key_a,
                    key_s,
                    key_d,
                    ref mut eye,
                    speed,
                    pitch,
                    yaw,
                    ..
                } = self.window.camera_state
                {
                    if key_w || key_a || key_s || key_d {
                        let fwd = match (key_w, key_s) {
                            (true, true) => 0.,
                            (true, false) => speed,
                            (false, true) => -speed,
                            _ => 0.,
                        };
                        let stf = match (key_a, key_d) {
                            (true, true) => 0.,
                            (true, false) => speed,
                            (false, true) => -speed,
                            _ => 0.,
                        };
                        let d = Vector3::new(
                            pitch.cos() * yaw.sin(),
                            pitch.sin(),
                            pitch.cos() * yaw.cos(),
                        );
                        let sy = yaw + Deg(90.);
                        let sd = Vector3::new(
                            Deg(0f32).cos() * sy.sin(),
                            0.,
                            Deg(0f32).cos() * sy.cos(),
                        );
                        *eye += d * fwd;
                        *eye += sd * stf;
                        self.window.update_camera_int();
                    }
                }
                {
                    let mut h = self.window.handler.take();
                    if let Some(ref mut hh) = h {
                        hh.should_redraw(&mut self.window);
                    }
                    self.window.handler = h;
                }
                let surface = self.window.surface.as_ref().unwrap();
                let mut frame = match surface.get_current_texture() {
                    Ok(f) => f,
                    Err(_) => return,
                };
                self.window.update_camera_int();
                let mut handler = self.window.handler.take().unwrap();
                self.window.render_internal(&mut handler, false, &mut frame);
                self.window.set_ui_camera();
                self.window.render_internal(&mut handler, true, &mut frame);
                self.window.handler = Some(handler);
                frame.present();
                if let Some(w) = &self.window.winit_window {
                    w.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let key = match event.physical_key {
                    PhysicalKey::Code(c) => c,
                    _ => return,
                };
                match (event.state, key, &mut self.window.camera_state) {
                    (ElementState::Pressed, KeyCode::KeyW, CameraState::Flight3d { key_w, .. }) => {
                        *key_w = true
                    }
                    (
                        ElementState::Released,
                        KeyCode::KeyW,
                        CameraState::Flight3d { key_w, .. },
                    ) => *key_w = false,
                    (ElementState::Pressed, KeyCode::KeyA, CameraState::Flight3d { key_a, .. }) => {
                        *key_a = true
                    }
                    (
                        ElementState::Released,
                        KeyCode::KeyA,
                        CameraState::Flight3d { key_a, .. },
                    ) => *key_a = false,
                    (ElementState::Pressed, KeyCode::KeyS, CameraState::Flight3d { key_s, .. }) => {
                        *key_s = true
                    }
                    (
                        ElementState::Released,
                        KeyCode::KeyS,
                        CameraState::Flight3d { key_s, .. },
                    ) => *key_s = false,
                    (ElementState::Pressed, KeyCode::KeyD, CameraState::Flight3d { key_d, .. }) => {
                        *key_d = true
                    }
                    (
                        ElementState::Released,
                        KeyCode::KeyD,
                        CameraState::Flight3d { key_d, .. },
                    ) => *key_d = false,
                    (ElementState::Pressed, _, _) => {
                        let mut h = self.window.handler.take().unwrap();
                        h.key_down(&mut self.window, key);
                        self.window.handler = Some(h);
                    }
                    (ElementState::Released, _, _) => {
                        let mut h = self.window.handler.take().unwrap();
                        h.key_up(&mut self.window, key);
                        self.window.handler = Some(h);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some((lx, ly)) = self.window.last_mouse {
                    let (dx, dy) = (position.x - lx, position.y - ly);
                    match &mut self.window.camera_state {
                        CameraState::Pannable2d {
                            x,
                            y,
                            zoom,
                            panning: true,
                            ..
                        } => {
                            *x -= dx as f32 / self.window.width as f32 * 2. * *zoom;
                            *y += dy as f32 / self.window.height as f32 * 2. * *zoom;
                            self.window.update_camera();
                        }
                        CameraState::Flight3d {
                            pitch,
                            yaw,
                            panning: true,
                            ..
                        } => {
                            *pitch = Deg((pitch.0 + (-dy as f32 / 5.)).clamp(-89., 89.));
                            *yaw += Deg(-dx as f32 / 5.);
                            self.window.update_camera();
                        }
                        _ => {
                            let mut h = self.window.handler.take().unwrap();
                            h.mouse_move(&mut self.window, dx, dy);
                            self.window.handler = Some(h);
                        }
                    }
                }
                self.window.last_mouse = Some((position.x, position.y));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                match (state, button, &mut self.window.camera_state) {
                    (
                        ElementState::Pressed,
                        MouseButton::Left,
                        CameraState::Pannable2d { panning, .. }
                        | CameraState::Flight3d { panning, .. },
                    ) => *panning = true,
                    (
                        ElementState::Released,
                        MouseButton::Left,
                        CameraState::Pannable2d { panning, .. }
                        | CameraState::Flight3d { panning, .. },
                    ) => *panning = false,
                    (ElementState::Pressed, _, _) => {
                        let mut h = self.window.handler.take().unwrap();
                        h.mouse_down(&mut self.window, button);
                        self.window.handler = Some(h);
                    }
                    (ElementState::Released, _, _) => {
                        let mut h = self.window.handler.take().unwrap();
                        h.mouse_up(&mut self.window, button);
                        self.window.handler = Some(h);
                    }
                }
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, y),
                ..
            } => match &mut self.window.camera_state {
                CameraState::Pannable2d { zoom, .. } => {
                    if y > 0. {
                        *zoom /= 1.25;
                    } else {
                        *zoom *= 1.25;
                    }
                    self.window.update_camera();
                }
                CameraState::Flight3d { speed, .. } => {
                    if y > 0. {
                        *speed *= 2.;
                    } else {
                        *speed /= 2.;
                    }
                }
                _ => {
                    let mut h = self.window.handler.take().unwrap();
                    h.mouse_scroll(&mut self.window, y);
                    self.window.handler = Some(h);
                }
            },
            _ => {}
        }
    }
}
