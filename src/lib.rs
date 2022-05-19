use std::rc::Rc;
use std::ops::Range;
use std::num::NonZeroU32;
use std::time::Instant;
use std::borrow::Cow;
use wgpu::util::{DeviceExt, BufferInitDescriptor};
use cgmath::{Deg, Angle, Point3, point3, Vector3, perspective, Matrix4};
use cgmath::{Vector2, Vector4};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState};
use winit::event::MouseButton;
use winit::event::MouseScrollDelta;
use winit::event::DeviceEvent::MouseMotion;
use winit::window::WindowBuilder;
use winit::event_loop::{EventLoop, ControlFlow, EventLoopProxy};

pub use wgpu::*;
pub use winit::event::VirtualKeyCode;
pub use winit;
pub use cgmath;

const FONT_WIDTH: usize = 9;
const FONT_HEIGHT: usize = 16;

/// Type to use user events to the event loop
type UserEvent = bool;

/// Wrapper type around [`Error`]
type Result<T> = std::result::Result<T, Error>;

/// Error statuses
#[derive(Debug)]
pub enum Error {
    /// Creating the window failed
    CreateWindow(winit::error::OsError),

    /// Failed to get a compatible adapter
    GetAdapter,

    /// Failed to create to the logical device and queue
    CreateDevice(RequestDeviceError),

    /// Failed to get the preferred format for the output device's surface
    PreferredFormat,

    /// Failed to get the texture associated with a surface we are rendering
    /// to
    GetSurfaceTexture(SurfaceError),

    /// Grabbing the cursor failed
    CursorGrab(winit::error::ExternalError),

    /// Failed to send an event with an event loop proxy
    SendEvent(winit::event_loop::EventLoopClosed<UserEvent>),
}

/// Camera modes for the built-in cameras
pub enum CameraMode {
    /// A 2d canvas which is designed to be panned with the mouse and zoomed
    /// with the scroll wheel
    Pannable2d,

    /// A "flying" 3d camera where 'W' and 'S' move your forwards or backwards
    /// relative to the cameras facing, 'A' and 'S' strafe you left and right
    /// and the scroll wheel increases or decreases this movement speed.
    ///
    /// The camera pitch and yaw can be panned by left clicking and moving the
    /// mouse
    Flight3d,
}

/// Camera states for the built-in cameras
#[derive(Debug)]
enum CameraState {
    /// A 2d canvas which is designed to be panned with the mouse and zoomed
    /// with the scroll wheel
    Pannable2d {
        /// Tracks if we're currently panning the camera (eg. left mouse held)
        panning: bool,

        /// Camera center X
        x: f32,

        /// Camera center Y
        y: f32,

        /// Camera zoom level, larger number means more zoomed out
        zoom: f32,
    },

    /// A "flying" 3d camera where 'W' and 'S' move your forwards or backwards
    /// relative to the cameras facing, 'A' and 'S' strafe you left and right
    /// and the scroll wheel increases or decreases this movement speed.
    ///
    /// The camera pitch and yaw can be panned by left clicking and moving the
    /// mouse
    Flight3d {
        /// Eye location
        eye: Point3<f32>,

        /// Camera pitch
        pitch: Deg<f32>,

        /// Camera yaw
        yaw: Deg<f32>,

        /// Speed to move the camera at when holding a movement key
        speed: f32,

        /// Tracks if the mouse is currently held (thus panning)
        panning: bool,

        /// Is the W key currently pressed?
        key_w: bool,

        /// Is the A key currently pressed?
        key_a: bool,

        /// Is the S key currently pressed?
        key_s: bool,

        /// Is the D key currently pressed?
        key_d: bool,
    }
}

/// The format for vertices for this program
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex {
    /// Position of the vertex
    pub pos: Vector3<f32>,

    /// Red component of the vertex
    pub r: u8,

    /// Green component of the vertex
    pub g: u8,

    /// Blue component of the vertex
    pub b: u8,

    /// Padding
    _padding: u8,
}

impl Vertex {
    /// Create a new vertex with color
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, r: u8, g: u8, b: u8) -> Self {
        Self { pos: Vector3::new(x, y, z), r, g, b, _padding: 0 }
    }

    /// Set the color and return a new point
    #[inline]
    pub const fn color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.r = r;
        self.g = g;
        self.b = b;
        self
    }
}

/// Format for verticies for textures
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TextureVertex {
    /// Position of the vertex
    pub pos: Vector3<f32>,

    /// Texture coordinates
    pub tex_coords: Vector2<f32>,
    
    /// Color
    pub color: Vector4<f32>,
}

impl TextureVertex {
    /// Create a new vertex with color
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, u: f32, v: f32,
            r: f32, g: f32, b: f32) -> Self {
        Self {
            pos:        Vector3::new(x, y, z),
            tex_coords: Vector2::new(u, v),
            color:      Vector4::new(r, g, b, 1.),
        }
    }
}

/// Should the drawing persist in the scene buffer such that it can be
/// incrementally built on
#[derive(Clone, Copy)]
pub enum Persist {
    Yes,
    No,
}

/// Different commands we can use to draw data
#[derive(Clone)]
pub enum DrawCommand {
    Triangles(Persist, Rc<Buffer>, std::ops::Range<u32>),
    Points(Persist, Rc<Buffer>, std::ops::Range<u32>),
    Lines(Persist, Rc<Buffer>, std::ops::Range<u32>),
}

impl DrawCommand {
    /// Get the number of vertices to draw
    pub fn len(&self) -> usize {
        match self {
            DrawCommand::Triangles(_, _, x) | DrawCommand::Points(_, _, x) |
            DrawCommand::Lines(_, _, x) => {
                (x.end - x.start) as usize
            }
        }
    }
}

/// Required trait to handle rendering events
pub trait EventHandler: Sized {
    /// Called when the window is created to create this handler
    fn create(window: &mut Window<Self>) -> Self;

    /// A key was pressed down
    fn key_down(&mut self, _window: &mut Window<Self>, _key: VirtualKeyCode) {}

    /// A key was released
    fn key_up(&mut self, _window: &mut Window<Self>, _key: VirtualKeyCode) {}

    /// The mouse moved
    fn mouse_move(&mut self, _window: &mut Window<Self>,
        _delta_x: f64, _delta_y: f64) {}

    /// The mouse vertically scrolled
    fn mouse_scroll(&mut self, _window: &mut Window<Self>, _delta_y: f32) {}

    /// A mouse button was pressed
    fn mouse_down(&mut self, _window: &mut Window<Self>,
        _button: winit::event::MouseButton) {}

    /// A mouse button was released
    fn mouse_up(&mut self, _window: &mut Window<Self>,
        _button: winit::event::MouseButton) {}

    /// We're deciding if we want to schedule a new frame for drawing. You
    /// should use this if you need another frame to be drawn without an
    /// event firing (eg. moving while a key is still held down)
    fn should_redraw(&mut self, _window: &mut Window<Self>) {}

    /// Called before rendering each frame to get draw commands
    /// A user should invoke `window.push_command()` to enqueue draw commands
    fn render(&mut self, _window: &mut Window<Self>, _incremental: bool) {}

    /// Called before rendering each frame to get draw commands
    /// A user should invoke `window.push_command()` to enqueue draw commands
    ///
    /// Called when rendering the UI, the view will be an ortho projection
    /// which is pixel-identical to the dimensions of the inner window with
    /// the bottom left corner being (0, 0)
    fn render_ui(&mut self, _window: &mut Window<Self>) {}
}

/// MSAA values
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Msaa {
    /// No subsampling
    None = 1,

    /// 2x MSAA
    X2 = 2,

    /// 4x MSAA
    X4 = 4,
}

/// Vsync
#[derive(Clone, Copy)]
pub enum Vsync {
    /// Disabled
    Off,

    /// Enabled
    On,
}

/// Proxy to request a redraw from another thread
pub struct RedrawTrigger(EventLoopProxy<UserEvent>);

impl RedrawTrigger {
    /// Request a redraw
    ///
    /// If `incremental` is set, initiates a redraw using the most recently
    /// rendered scene
    pub fn request_redraw(&self, incremental: bool) -> Result<()> {
        self.0.send_event(incremental).map_err(Error::SendEvent)
    }
}

/// A 3d accelerated window
pub struct Window<EH: EventHandler> {
    /// The time this `Window` was created
    pub start: Instant,

    /// User-supplied event handler
    handler: Option<EH>,

    /// Event loop for the window
    event_loop: Option<EventLoop<UserEvent>>,

    /// Window that we created
    window: winit::window::Window,

    /// Logical width of the inner part of the window
    width: u32,

    /// Logical height of the inner part of the window
    height: u32,

    /// Device we're using to render with
    device: Device,

    /// Surface to draw on when rendering
    surface: Surface,

    /// Queue to use for sending commands to the device
    queue: Queue,

    /// Internal camera mode
    camera_state: CameraState,

    /// Bind group for the camera
    camera_bind_group: BindGroup,

    /// Camera uniform buffer for the shader
    camera_buffer: Buffer,

    /// Bind group for the texture render pipeline
    texture_bind_group: BindGroup,

    /// Tracks if the next redraw is incremental
    incremental: bool,

    /// Render pipeline for triangles
    triangle_pipeline: RenderPipeline,

    /// Render pipeline for line segments
    line_pipeline: RenderPipeline,

    /// Render pipeline for points
    point_pipeline: RenderPipeline,

    /// Render pipeline for textures
    texture_pipeline: RenderPipeline,

    /// Configured MSAA level
    msaa_level: Msaa,

    /// Output texture
    output_texture: Texture,

    /// Output texture view
    output_view: TextureView,

    /// Output depth
    output_depth: Texture,

    /// Output depth view
    output_depth_view: TextureView,

    /// Scene texture (saved from output on a non-incremental render)
    scene_texture: Texture,

    /// Scene depth (saved from output on a non-incremental render)
    scene_depth: Texture,
    
    /// UI texture, copied to the background of the screen, allowing for pixel
    /// drawing
    background_texture: Texture,

    /// UI texture, copied to the background of the screen, allowing for pixel
    /// drawing. This is the depth buffer.
    background_depth: Texture,

    /// Preferred swapchain format
    swapchain_format: TextureFormat,

    /// Vsync state
    vsync: Vsync,

    /// View projection for the 3d camera
    camera_uniform: Matrix4<f32>,

    /// Has the window recently been resized (and thus needs the texture
    /// buffers to be updated on the next rendering)
    resized: bool,

    /// A proxy for the event loop so we can send commands from a thread
    proxy: EventLoopProxy<UserEvent>,

    /// Queued draw commands for drawing triangles to the scene
    persist_tri_commands: Vec<(Rc<Buffer>, Range<u32>)>,

    /// Queued draw commands for temporary triangles
    tri_commands: Vec<(Rc<Buffer>, Range<u32>)>,

    /// Queued draw commands for drawing lines to the scene
    persist_line_commands: Vec<(Rc<Buffer>, Range<u32>)>,

    /// Queued draw commands for temporary lines
    line_commands: Vec<(Rc<Buffer>, Range<u32>)>,

    /// Queued draw commands for drawing points to the scene
    persist_point_commands: Vec<(Rc<Buffer>, Range<u32>)>,

    /// Queued draw commands for temporary points
    point_commands: Vec<(Rc<Buffer>, Range<u32>)>,

    /// Queued draw commands for text
    text_commands: Vec<(Rc<Buffer>, Range<u32>)>,

    /// Temporary storage which is re-purposed in `push_text`
    text_temp: Vec<TextureVertex>,
}

impl<EH: 'static + EventHandler> Window<EH> {
    /// Create a new window with a given `title`, `width` and `height` using
    /// MSAA level `msaa_level`
    ///
    /// The window will not be displayed until `window.run()` is invoked
    pub fn new(title: impl AsRef<str>,
            width: u32, height: u32, msaa_level: Msaa, vsync: Vsync)
                -> Result<Self> {
        // Get the start time
        let start = Instant::now();

        // Create an event loop for window events
        let event_loop = EventLoop::with_user_event();

        // Create a window
        let window = WindowBuilder::new()
            .with_inner_size(LogicalSize::new(width, height))
            .with_title(title.as_ref())
            .with_visible(false)
            .build(&event_loop).map_err(Error::CreateWindow)?;

        // Get the inner physical size of the window, as we originally used a
        // logical size to set the window size
        let width  = window.inner_size().width;
        let height = window.inner_size().height;

        // Create new instance of WGPU using a first-tier supported backend
        // Eg: Vulkan + Metal + DX12 + Browser WebGPU
        let instance = Instance::new(Backends::PRIMARY);

        // Create a surface for our Window. Unsafe since this uses raw window
        // handles and the window must remain valid for the lifetime of the
        // created surface
        //
        // A Surface represents a platform-specific surface (e.g. a window)
        // onto which rendered images may be presented.
        let surface = unsafe { instance.create_surface(&window) };

        // Get a handle to a physical graphics and/or compute device
        let adapter = pollster::block_on(async {
            instance.request_adapter(&RequestAdapterOptions {
                // Request the high performance graphics adapter, eg. pick the
                // discrete GPU over the integrated GPU
                power_preference: PowerPreference::HighPerformance,

                // Don't force fallback, we don't want software rendering :D
                force_fallback_adapter: false,

                // Make sure the adapter we request can render on `surface`
                compatible_surface: Some(&surface),
            }).await
        }).ok_or(Error::GetAdapter)?;

        // Create the logical device and command queue
        let (mut device, queue) = pollster::block_on(async {
            adapter.request_device(&DeviceDescriptor {
                // Debug label for the device
                label: None,

                // Features that the device should support
                features: Features::empty(),

                // Limits that the device should support. If any limit is
                // "better" than the limit exposed by the adapter, creating a
                // device will panic.
                limits: Limits::default(),
            }, None).await
        }).map_err(Error::CreateDevice)?;

        // Get the preferred texture format for the swapchain with the surface
        // and adapter we are using
        let swapchain_format = surface.get_preferred_format(&adapter)
            .ok_or(Error::PreferredFormat)?;

        // Load and compile the shaders
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            // Debugging label
            label:  None,

            // Shader source code
            source: ShaderSource::Wgsl(
                Cow::Borrowed(include_str!("shader.wgsl"))),
        });
        let texture_shader =
                device.create_shader_module(&ShaderModuleDescriptor {
            // Debugging label
            label:  None,

            // Shader source code
            source: ShaderSource::Wgsl(
                Cow::Borrowed(include_str!("texture_shader.wgsl"))),
        });

        // Configure the swap buffers
        surface.configure(&device, &SurfaceConfiguration {
            // Usage for the swap chain. In this case, this is currently the
            // only supported option.
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC |
                TextureUsages::COPY_DST,

            // Set the preferred texture format for the swap chain to be what
            // the surface and adapter want.
            format: swapchain_format,

            // Set the width of the swap chain
            width,

            // Set the height of the swap chain
            height,

            // The way data is presented to the screen
            // `Immediate` (no vsync)
            // `Mailbox`   (no vsync for rendering, but frames synced on vsync)
            // `Fifo`      (full vsync)
            present_mode: match vsync {
                Vsync::Off => PresentMode::Immediate,
                Vsync::On  => PresentMode::Fifo,
            },
        });

        // Create the camera buffer for supplying the camera uniform to the
        // shader
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            // Debug label
            label: None,

            // Buffer size `mat4x4<f32>`
            size: 4 * 4 * 4,

            // Usage for the buffer
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,

            // Buffer not mapped at creation
            mapped_at_creation: false,
        });

        // Create the bind group layout for the camera projection matrix
        let camera_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                // Debug label
                label: None,

                entries: &[
                    BindGroupLayoutEntry {
                        // Bind as zero
                        binding: 0,

                        // We only need this for the vertex shader
                        visibility: ShaderStages::VERTEX,

                        // Set the typing as a uniform
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   core::num::NonZeroU64::new(4 * 4 * 4),
                        },

                        // Not an array
                        count: None,
                    },
                ],
            });

        // Create the bind group for the camera projection matrix
        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            // Debug label
            label: None,

            // Layout to use for camera bind group
            layout: &camera_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    // Program binding zero
                    binding: 0,

                    // Set the binding to the camera buffer
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });

        // Load the font
        let diffuse_bytes = include_bytes!("VGA9x16.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let font_rgba = diffuse_image.to_rgba8().into_vec();

        use image::GenericImageView;
        let font_dimensions = diffuse_image.dimensions();

        // Create the texture for holding font data
        let font_texture = device.create_texture(&TextureDescriptor {
            label:           None,
            mip_level_count: 1,
            sample_count:    msaa_level as u32,
            dimension:       TextureDimension::D2,

            size: Extent3d {
                width:  font_dimensions.0,
                height: font_dimensions.1,
                depth_or_array_layers: 1
            },

            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        });

        // Create texture view and sampler for the font
        let font_texture_view =
            font_texture.create_view(&TextureViewDescriptor::default());
        let font_sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter:     FilterMode::Nearest,
            min_filter:     FilterMode::Nearest,
            mipmap_filter:  FilterMode::Nearest,
            ..Default::default()
        });

        // Create the bind group layout for textures
        let texture_bind_group_layout = device.create_bind_group_layout(
            &BindGroupLayoutDescriptor {
                // Debug label
                label: None,

                entries: &[
                    BindGroupLayoutEntry {
                        // Bind as zero
                        binding: 0,

                        // We only need this for the vertex shader
                        visibility: ShaderStages::VERTEX,

                        // Set the typing as a uniform
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   None,
                        },

                        // Not an array
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding:    1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled:   false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type:
                                TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding:    2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(
                            SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            });

        // Create the bind group for the font
        let texture_bind_group =
                device.create_bind_group(&BindGroupDescriptor {
            // Debug label
            label: None,

            // Layout to use for font bind group
            layout: &texture_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    // Program binding zero
                    binding: 0,

                    // Set the binding to the camera buffer
                    resource: camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&font_texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&font_sampler),
                }
            ],
        });

        // Create output
        let (output_texture, output_view, output_depth, output_depth_view) =
                Self::create_texture_pair(
            &mut device, width, height, msaa_level as u32, swapchain_format
        );

        // Create scene save buffer
        let (scene_texture, _scene_view, scene_depth, _scene_depth_view) =
                Self::create_texture_pair(
            &mut device, width, height, msaa_level as u32, swapchain_format
        );

        // Create pixel background for pixel drawing
        let (background_texture, _scene_view, background_depth,
                _scene_depth_view) = Self::create_texture_pair(
            &mut device, width, height, msaa_level as u32, swapchain_format
        );

        // Create line pipeline
        let line_pipeline = Self::create_render_pipeline(
            &mut device,
            &camera_bind_group_layout,
            PrimitiveTopology::LineList,
            &shader,
            swapchain_format,
            msaa_level as u32);

        // Create triangle pipeline
        let triangle_pipeline = Self::create_render_pipeline(
            &mut device,
            &camera_bind_group_layout,
            PrimitiveTopology::TriangleList,
            &shader,
            swapchain_format,
            msaa_level as u32);

        // Create texture pipeline
        let texture_pipeline = Self::create_texture_render_pipeline(
            &mut device,
            &texture_bind_group_layout,
            PrimitiveTopology::TriangleList,
            &texture_shader,
            swapchain_format,
            msaa_level as u32);

        // Create point pipeline
        let point_pipeline = Self::create_render_pipeline(
            &mut device,
            &camera_bind_group_layout,
            PrimitiveTopology::PointList,
            &shader,
            swapchain_format,
            msaa_level as u32);

        // Create the window
        let mut ret = Self {
            handler: None,
            start,
            window,
            width,
            height,
            proxy: event_loop.create_proxy(),
            event_loop: Some(event_loop),
            device,
            surface,
            queue,
            resized: false,
            incremental: false,
            camera_uniform: [[0f32; 4]; 4].into(),
            camera_state: CameraState::Pannable2d {
                panning: false,
                x:       0.,
                y:       0.,
                zoom:    1.,
            },
            camera_bind_group,
            camera_buffer,
            texture_pipeline,
            texture_bind_group,
            background_texture,
            background_depth,
            scene_texture,
            scene_depth,
            output_texture,
            output_view,
            output_depth,
            output_depth_view,
            point_pipeline,
            line_pipeline,
            triangle_pipeline,
            swapchain_format,
            vsync,
            msaa_level,
            persist_tri_commands: Vec::new(),
            tri_commands: Vec::new(),
            persist_line_commands: Vec::new(),
            line_commands: Vec::new(),
            persist_point_commands: Vec::new(),
            point_commands: Vec::new(),
            text_commands: Vec::new(),
            text_temp: Vec::new(),
        };

        // Write the texture data for the fonts
        ret.queue.write_texture(
            ImageCopyTexture {
                texture:   &font_texture,
                mip_level: 0,
                origin:    Origin3d::ZERO,
                aspect:    TextureAspect::All,
            },
            &font_rgba,
            ImageDataLayout {
                offset: 0,
                bytes_per_row:  NonZeroU32::new(4 * font_dimensions.0),
                rows_per_image: NonZeroU32::new(font_dimensions.1),
            },
            Extent3d {
                width:  font_dimensions.0,
                height: font_dimensions.1,
                depth_or_array_layers: 1
            },
        );

        // Set an initial camera state
        ret.update_camera();

        Ok(ret)
    }

    /// Set the window title
    pub fn set_title(&self, title: impl AsRef<str>) {
        self.window.set_title(title.as_ref());
    }

    /// Get the camera speed
    pub fn camera_speed(&self) -> Option<f32> {
        match self.camera_state {
            CameraState::Flight3d { speed, .. } => {
                Some(speed)
            }
            _ => None,
        }
    }

    /// Set the internal camera mode
    pub fn camera_mode(mut self, camera_mode: CameraMode) -> Self {
        // Set the initial camera state
        self.camera_state = match camera_mode {
            CameraMode::Pannable2d => {
                CameraState::Pannable2d {
                    panning: false,
                    x:       0.,
                    y:       0.,
                    zoom:    1.,
                }
            }
            CameraMode::Flight3d => {
                CameraState::Flight3d {
                    eye:     point3(0., 0., 0.),
                    pitch:   Deg(0.),
                    yaw:     Deg(0.),
                    speed:   1.,
                    panning: false,
                    key_w:   false,
                    key_a:   false,
                    key_s:   false,
                    key_d:   false,
                }
            }
        };

        self
    }

    /// Request a redraw trigger. This can be moved to other threads to trigger
    /// a redraw remotely
    pub fn redraw_trigger(&mut self) -> RedrawTrigger {
        RedrawTrigger(self.proxy.clone())
    }

    /// Push a draw command to the draw command list
    #[inline]
    pub fn push_command(&mut self, command: DrawCommand) {
        match command {
            DrawCommand::Triangles(Persist::Yes, buf, range) => {
                self.persist_tri_commands.push((buf, range));
            }
            DrawCommand::Triangles(Persist::No, buf, range) => {
                self.tri_commands.push((buf, range));
            }
            DrawCommand::Lines(Persist::Yes, buf, range) => {
                self.persist_line_commands.push((buf, range));
            }
            DrawCommand::Lines(Persist::No, buf, range) => {
                self.line_commands.push((buf, range));
            }
            DrawCommand::Points(Persist::Yes, buf, range) => {
                self.persist_point_commands.push((buf, range));
            }
            DrawCommand::Points(Persist::No, buf, range) => {
                self.point_commands.push((buf, range));
            }
        }
    }

    /// Push a string to be drawn on the screen
    pub fn push_text(&mut self, mut x: f32, y: f32, r: f32, g: f32, b: f32,
            text: impl AsRef<[u8]>) {
        const GL_W: f32 = 1. / 16.;
        const GL_H: f32 = 1. / 16.;

        for &letter in text.as_ref() {
            // Determine bounds of the quad
            let x1 = x;
            let x2 = x + FONT_WIDTH as f32;
            let y1 = y;
            let y2 = y + FONT_HEIGHT as f32;

            // Determine texture coordinates from the bitmap
            let u1 = (letter % 16) as f32 * GL_W;
            let v1 = (letter / 16) as f32 * GL_H;
            let u2 = u1 + GL_W;
            let v2 = v1 + GL_H;

            // Push the quad to hold this letter
            self.text_temp.push(TextureVertex::new(x2, y2, 0., u2, v1, r,g,b));
            self.text_temp.push(TextureVertex::new(x1, y2, 0., u1, v1, r,g,b));
            self.text_temp.push(TextureVertex::new(x2, y1, 0., u2, v2, r,g,b));

            self.text_temp.push(TextureVertex::new(x2, y1, 0., u2, v2, r,g,b));
            self.text_temp.push(TextureVertex::new(x1, y2, 0., u1, v1, r,g,b));
            self.text_temp.push(TextureVertex::new(x1, y1, 0., u1, v2, r,g,b));

            // Advance to the next screen position
            x += FONT_WIDTH as f32 + 1.;
        }
    }

    /// Set the camera uniform to the UI camera
    pub fn set_ui_camera(&mut self) {
        let camera_uniform = cgmath::ortho(
            0.,
            self.width as f32,
            0.,
            self.height as f32,
            -1.,
            1.
        );

        // Update the camera uniform buffer
        let camera_uniform = unsafe {
            std::slice::from_raw_parts(
                std::ptr::addr_of!(camera_uniform) as *const u8,
                std::mem::size_of_val(&camera_uniform))
        };
        self.queue.write_buffer(&self.camera_buffer, 0, camera_uniform);
    }

    /// Get the position of the camera-rendered coordinate in
    /// UI x, y coordinates
    /// Returns `None` if the `(x, y, z)` is not currently on screen (eg.
    /// behind the camera)
    pub fn screen_position(&self, x: f32, y: f32, z: f32) -> Option<(f32, f32)> {
        // Create the target location
        let target = Vector4::new(x, y, z, 1.);

        // Compute the normalized ([-1, 1] being edges of screen) position
        let target_un = self.camera_uniform * target;
        let target    = target_un / target_un.w; // Normalize it

        // Convert to UI x, y coords and return them!
        let x = (target.x + 1.) / 2. * self.width  as f32;
        let y = (target.y + 1.) / 2. * self.height as f32;

        if target_un.w >= 0. {
            Some((x, y))
        } else {
            None
        }
    }

    /// Update the camera uniform
    pub fn update_camera_int(&mut self) {
        let camera_uniform = match &mut self.camera_state {
            CameraState::Pannable2d { x, y, zoom, .. }
                    => {
                cgmath::ortho(
                    (*x - *zoom) * (self.width as f32 / self.height as f32),
                    (*x + *zoom) * (self.width as f32 / self.height as f32),
                    *y - *zoom,
                    *y + *zoom,
                    -1.,
                    1.
                )
            }
            CameraState::Flight3d { eye, pitch, yaw, .. } => {
                // Compute the vector which is the direction the camera is
                // facing
                let direction = Vector3::new(
                    pitch.cos() * yaw.sin(), pitch.sin(),
                    pitch.cos() * yaw.cos());

                // Create a look_to view matrix based on the pitch and yaw
                let view = Matrix4::look_to_rh(*eye, direction,
                    Vector3::unit_y());

                // Create a perspective with 45 degree FoV and a znear and zfar
                // of
                // 1 and 10000
                let proj = perspective(Deg(45.),
                    self.width as f32 / self.height as f32, 1., 100000.);

                // Update camera uniform
                self.camera_uniform = proj * view;
                self.camera_uniform
            }
        };

        // Update the camera uniform buffer
        let camera_uniform = unsafe {
            std::slice::from_raw_parts(
                std::ptr::addr_of!(camera_uniform) as *const u8,
                std::mem::size_of_val(&camera_uniform))
        };
        self.queue.write_buffer(&self.camera_buffer, 0, camera_uniform);
    }
    
    /// Update the camera position
    pub fn set_camera(&mut self, new_eye: Point3<f32>,
            new_pitch: Deg<f32>, new_yaw: Deg<f32>) {
        if let CameraState::Flight3d { eye, pitch, yaw, .. } =
                &mut self.camera_state {
            *eye = new_eye;
            *pitch = new_pitch;
            *yaw = new_yaw;
        }

        self.update_camera_int();

        // Request a full redraw
        self.request_redraw(false);
    }

    /// Update the camera uniform and request a redraw
    pub fn update_camera(&mut self) {
        self.update_camera_int();

        // Request a full redraw
        self.request_redraw(false);
    }

    /// Request a redraw
    ///
    /// If `incremental` is set, initiates a redraw using the most recently
    /// rendered scene
    pub fn request_redraw(&mut self, incremental: bool) {
        // If anyone requests non-incremental, make sure we stay
        // non-incremental
        self.incremental &= incremental;

        // Request the redraw!
        self.window.request_redraw();
    }

    /// Draw directly to the background texture
    pub fn write_background_texture(&mut self, data: impl AsRef<[u8]>) {
        // Write the texture data for the fonts
        self.queue.write_texture(
            ImageCopyTexture {
                texture:   &self.background_texture,
                mip_level: 0,
                origin:    Origin3d::ZERO,
                aspect:    TextureAspect::All,
            },
            data.as_ref(),
            ImageDataLayout {
                offset: 0,
                bytes_per_row:  NonZeroU32::new(4 * self.width),
                rows_per_image: NonZeroU32::new(self.height),
            },
            Extent3d {
                width:  self.width,
                height: self.height,
                depth_or_array_layers: 1
            },
        );
    }

    /// Create a new vertex buffer with given `data`
    pub fn create_vertex_buffer(&mut self,
            data: impl AsRef<[Vertex]>) -> Rc<Buffer> {
        // Make a new buffer with the desired shape and contents
        Rc::new(self.device.create_buffer_init(&BufferInitDescriptor {
            // Debug label
            label: None,

            // Contents for the buffer
            contents: unsafe {
                std::slice::from_raw_parts(
                    data.as_ref().as_ptr() as *const u8,
                    std::mem::size_of_val(data.as_ref()))
            },

            // Usage of the buffer is vertex data
            usage: BufferUsages::VERTEX,
        }))
    }

    /// Create a new texture vertex buffer from internal text data
    fn create_texture_vertex_buffer(&mut self) -> Rc<Buffer> {
        // Make a new buffer with the desired shape and contents
        Rc::new(self.device.create_buffer_init(&BufferInitDescriptor {
            // Debug label
            label: None,

            // Contents for the buffer
            contents: unsafe {
                std::slice::from_raw_parts(
                    self.text_temp.as_ptr() as *const u8,
                    std::mem::size_of_val(self.text_temp.as_slice()))
            },

            // Usage of the buffer is vertex data
            usage: BufferUsages::VERTEX,
        }))
    }

    /// Create arendering pipeline for a given topology
    fn create_render_pipeline(
            device:                   &mut Device,
            camera_bind_group_layout: &BindGroupLayout,
            topology:                 PrimitiveTopology,
            shader:                   &ShaderModule,
            swapchain_format:         TextureFormat,
            msaa_level:               u32)
                -> RenderPipeline {
        // Create a render pipeline, mainly we have to do this to set the bind
        // groups
        let render_pipeline_layout = device.create_pipeline_layout(
            &PipelineLayoutDescriptor {
                // Label for debugging
                label: None,

                // Bind groups
                bind_group_layouts: &[
                    camera_bind_group_layout,
                ],

                // Constant ranges
                push_constant_ranges: &[],
            }
        );

        // Create a pipeline which applies the passes needed for rendering
        device.create_render_pipeline(&RenderPipelineDescriptor {
            // Debug label of the pipeline. This will show up in graphics
            // debuggers for easy identification.
            label: None,

            // The layout of bind groups for this pipeline.
            layout: Some(&render_pipeline_layout),

            // The compiled vertex stage, its entry point, and the input
            // buffers layout.
            vertex: VertexState {
                // Compiled shader
                module: shader,

                // Name of the function for the entry point
                entry_point: "vs_main",

                // Buffers to pass in
                buffers: &[VertexBufferLayout {
                    // Stride of elements in the buffer (in bytes)
                    array_stride: std::mem::size_of::<Vertex>() as u64,

                    // Step mode
                    step_mode: VertexStepMode::Vertex,

                    // Attributes to define the layout of the buffer
                    attributes: &[
                        // A `vec3<f32>` which contains X, Y, Z floats
                        VertexAttribute {
                            // Offset in the structure
                            offset: 0,

                            // Location to bind to in the shader
                            shader_location: 0,

                            // Format
                            format: VertexFormat::Float32x3,
                        },

                        // The R, G, B values as a `uvec4<u8>`
                        VertexAttribute {
                            // Offset in the structure
                            offset: 12,

                            // Location to bind to in the shader
                            shader_location: 1,

                            // Format
                            // Four unsigned bytes (u8). [0, 255] converted to
                            // float [0, 1] vec4 in shaders.
                            format: VertexFormat::Unorm8x4,
                        },
                    ],
                }],
            },

            // The properties of the pipeline at the primitive assembly and
            // rasterization level.
            primitive: PrimitiveState {
                topology,
                cull_mode: Some(Face::Back),
                ..Default::default()
            },

            // The compiled fragment stage, its entry point, and the color
            // targets.
            fragment: Some(FragmentState {
                // Compiled shader
                module: shader,

                // Name of the function for the entry point
                entry_point: "fs_main",

                // Type of output for the fragment shader (the correct texture
                // format that our GPU wants)
                targets: &[swapchain_format.into()],
            }),

            // The effect of draw calls on the depth and stencil aspects of the
            // output target, if any.
            depth_stencil: Some(DepthStencilState {
                // 32-bit floats for depth
                format: TextureFormat::Depth32Float,

                // Enable depth updates
                depth_write_enabled: true,

                // Depth comparison function
                depth_compare: CompareFunction::Less,

                // Stencil
                stencil: StencilState::default(),

                // Bias
                bias: DepthBiasState::default(),
            }),

            // The multi-sampling properties of the pipeline.
            multisample: MultisampleState {
                count: msaa_level,
                mask:  !0,
                alpha_to_coverage_enabled: false,
            },

            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
        })
    }

    /// Create a texture rendering pipeline
    fn create_texture_render_pipeline(
            device:                   &mut Device,
            camera_bind_group_layout: &BindGroupLayout,
            topology:                 PrimitiveTopology,
            shader:                   &ShaderModule,
            swapchain_format:         TextureFormat,
            msaa_level:               u32)
                -> RenderPipeline {
        // Create a render pipeline, mainly we have to do this to set the bind
        // groups
        let render_pipeline_layout = device.create_pipeline_layout(
            &PipelineLayoutDescriptor {
                // Label for debugging
                label: None,

                // Bind groups
                bind_group_layouts: &[
                    camera_bind_group_layout,
                ],

                // Constant ranges
                push_constant_ranges: &[],
            }
        );

        // Create a pipeline which applies the passes needed for rendering
        device.create_render_pipeline(&RenderPipelineDescriptor {
            // Debug label of the pipeline. This will show up in graphics
            // debuggers for easy identification.
            label: None,

            // The layout of bind groups for this pipeline.
            layout: Some(&render_pipeline_layout),

            // The compiled vertex stage, its entry point, and the input
            // buffers layout.
            vertex: VertexState {
                // Compiled shader
                module: shader,

                // Name of the function for the entry point
                entry_point: "vs_main",

                // Buffers to pass in
                buffers: &[VertexBufferLayout {
                    // Stride of elements in the buffer (in bytes)
                    array_stride: std::mem::size_of::<TextureVertex>() as u64,

                    // Step mode
                    step_mode: VertexStepMode::Vertex,

                    // Attributes to define the layout of the buffer
                    attributes: &[
                        // A `vec3<f32>` which contains X, Y, Z floats
                        VertexAttribute {
                            // Offset in the structure
                            offset: 0,

                            // Location to bind to in the shader
                            shader_location: 0,

                            // Format
                            format: VertexFormat::Float32x3,
                        },

                        // A `vec2<f32>` which contains texture coordinates
                        VertexAttribute {
                            // Offset in the structure
                            offset: 12,

                            // Location to bind to in the shader
                            shader_location: 1,

                            // Format
                            format: VertexFormat::Float32x2,
                        },
                        
                        // A `vec4<f32>` which contains color
                        VertexAttribute {
                            // Offset in the structure
                            offset: 20,

                            // Location to bind to in the shader
                            shader_location: 2,

                            // Format
                            format: VertexFormat::Float32x4,
                        },
                    ],
                }],
            },

            // The properties of the pipeline at the primitive assembly and
            // rasterization level.
            primitive: PrimitiveState {
                topology,
                cull_mode: Some(Face::Back),
                ..Default::default()
            },

            // The compiled fragment stage, its entry point, and the color
            // targets.
            fragment: Some(FragmentState {
                // Compiled shader
                module: shader,

                // Name of the function for the entry point
                entry_point: "fs_main",

                // Type of output for the fragment shader (the correct texture
                // format that our GPU wants)
                targets: &[swapchain_format.into()],
            }),

            // The effect of draw calls on the depth and stencil aspects of the
            // output target, if any.
            depth_stencil: Some(DepthStencilState {
                // 32-bit floats for depth
                format: TextureFormat::Depth32Float,

                // Enable depth updates
                depth_write_enabled: true,

                // Depth comparison function
                depth_compare: CompareFunction::Less,

                // Stencil
                stencil: StencilState::default(),

                // Bias
                bias: DepthBiasState::default(),
            }),

            // The multi-sampling properties of the pipeline.
            multisample: MultisampleState {
                count: msaa_level,
                mask:  !0,
                alpha_to_coverage_enabled: false,
            },

            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
        })
    }

    /// Create a texture with a given width, height, format, usages, and MSAA
    fn create_texture_msaa(device: &mut Device,
            width: u32, height: u32, format: TextureFormat,
            usage: TextureUsages, msaa_level: u32) -> (Texture, TextureView) {
        // Create the texture
        let texture = device.create_texture(&TextureDescriptor {
            label:           None,
            mip_level_count: 1,
            sample_count:    msaa_level,
            dimension:       TextureDimension::D2,

            size: Extent3d {
                width:  width,
                height: height,
                depth_or_array_layers: 1
            },

            format,
            usage,
        });

        // Create the view
        let view = texture.create_view(&TextureViewDescriptor::default());

        // Return the result
        (texture, view)
    }

    /// Create a pair of color textures and depth textures
    fn create_texture_pair(device: &mut Device, width: u32, height: u32,
            msaa_level: u32, swapchain_format: TextureFormat)
                -> (Texture, TextureView, Texture, TextureView) {
        // Create the output texture
        let (texture, texture_view) = Self::create_texture_msaa(
            device,
            width,
            height,
            swapchain_format,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING |
            TextureUsages::COPY_DST | TextureUsages::COPY_SRC,
            msaa_level);

        // Create depth texture
        let (depth, depth_view) =
            Self::create_texture_msaa(
            device,
            width,
            height,
            TextureFormat::Depth32Float,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING |
            TextureUsages::COPY_DST | TextureUsages::COPY_SRC,
            msaa_level);

        // Return the color and depth textures
        (texture, texture_view, depth, depth_view)
    }

    /// Return the window's client area width and height
    pub fn client_area(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Handle the events from the event loop
    fn handle_event(&mut self, event: Event<UserEvent>,
            control_flow: &mut ControlFlow) -> Result<()> {
        // ControlFlow::Wait pauses the event loop if no events are
        // available to process.  This is ideal for non-game applications
        // that only update in response to user input, and uses
        // significantly less power/CPU time than ControlFlow::Poll.
        *control_flow = ControlFlow::Wait;

        // Get the handler
        let mut handler = self.handler.take().unwrap();

        // Handle events
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                // Exit when the user closes the window
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent { event: WindowEvent::ScaleFactorChanged {
                new_inner_size, ..
            }, .. } => {
                // Scale factor change (changed host resolution/DPI, moved
                // window to a monitor with a different DPI)
                self.width   = new_inner_size.width;
                self.height  = new_inner_size.height;
                self.resized = true;
            }
            Event::WindowEvent { event: WindowEvent::Resized(psize), .. } => {
                // Resized window
                self.width   = psize.width;
                self.height  = psize.height;
                self.resized = true;
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput {
                input, ..
            }, ..} => {
                match (input.state, input.virtual_keycode,
                        &mut self.camera_state) {
                    (ElementState::Pressed, Some(VirtualKeyCode::W),
                            CameraState::Flight3d { key_w, .. }) => {
                        *key_w = true;
                    }
                    (ElementState::Released, Some(VirtualKeyCode::W),
                            CameraState::Flight3d { key_w, .. }) => {
                        *key_w = false;
                    }
                    (ElementState::Pressed, Some(VirtualKeyCode::A),
                            CameraState::Flight3d { key_a, .. }) => {
                        *key_a = true;
                    }
                    (ElementState::Released, Some(VirtualKeyCode::A),
                            CameraState::Flight3d { key_a, .. }) => {
                        *key_a = false;
                    }
                    (ElementState::Pressed, Some(VirtualKeyCode::S),
                            CameraState::Flight3d { key_s, .. }) => {
                        *key_s = true;
                    }
                    (ElementState::Released, Some(VirtualKeyCode::S),
                            CameraState::Flight3d { key_s, .. }) => {
                        *key_s = false;
                    }
                    (ElementState::Pressed, Some(VirtualKeyCode::D),
                            CameraState::Flight3d { key_d, .. }) => {
                        *key_d = true;
                    }
                    (ElementState::Released, Some(VirtualKeyCode::D),
                            CameraState::Flight3d { key_d, .. }) => {
                        *key_d = false;
                    }
                    (ElementState::Pressed, Some(key), _) => {
                        // Pass through key
                        handler.key_down(self, key);
                    }
                    (ElementState::Released, Some(key), _) => {
                        // Pass through key
                        handler.key_up(self, key);
                    }
                    _ => {}
                }
            }
            Event::DeviceEvent { event: MouseMotion { delta: (x, y) }, .. } =>{
                match &mut self.camera_state {
                    CameraState::Pannable2d {
                        x: camx, y: camy, zoom, panning: true, ..
                    } => {
                        // Adjust panning a bit
                        *camx -= x as f32 / self.width  as f32 * 2. * *zoom;
                        *camy += y as f32 / self.height as f32 * 2. * *zoom;

                        // Update the camera
                        self.update_camera();
                    }
                    CameraState::Flight3d {
                        pitch, yaw, panning: true, ..
                    } => {
                        // Update the pitch (clamping to [-89, 89])
                        *pitch = Deg((pitch.0 + (-y as f32 / 5.))
                            .clamp(-89., 89.));

                        // Update yaw
                        *yaw += Deg(-x as f32 / 5.);

                        // Update the camera
                        self.update_camera();
                    }
                    _ => {
                        // Pass through movement
                        handler.mouse_move(self, x, y);
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::MouseInput {
                state: ElementState::Pressed, button, ..
            }, ..} => {
                match (button, &mut self.camera_state) {
                    (MouseButton::Left,
                     CameraState::Pannable2d { panning, .. } |
                     CameraState::Flight3d { panning, .. })
                            => {
                        // We're panning
                        self.window.set_cursor_grab(true)
                            .map_err(Error::CursorGrab)?;
                        self.window.set_cursor_visible(false);
                        *panning = true;
                    }
                    _ => {
                        // Pass through mouse event
                        handler.mouse_down(self, button);
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::MouseInput {
                state: ElementState::Released, button, ..
            }, ..} => {
                match (button, &mut self.camera_state) {
                    (MouseButton::Left,
                     CameraState::Pannable2d { panning, .. } |
                     CameraState::Flight3d { panning, .. })
                            => {
                        // We're no longer panning
                        self.window.set_cursor_grab(false)
                            .map_err(Error::CursorGrab)?;
                        self.window.set_cursor_visible(true);
                        *panning = false;
                    }
                    _ => {
                        // Pass through mouse event
                        handler.mouse_up(self, button);
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, y), .. }, ..
            } => {
                match &mut self.camera_state {
                    CameraState::Pannable2d { zoom, .. } => {
                        if y > 0. {
                            *zoom /= 1.25;
                        } else {
                            *zoom *= 1.25;
                        }

                        // Update the camera
                        self.update_camera();
                    }
                    CameraState::Flight3d { speed, .. } => {
                        // Update speed for camera movement
                        if y > 0. {
                            *speed *= 2.;
                        } else {
                            *speed /= 2.;
                        }
                    }
                }
            }
            Event::RedrawRequested(_) => {
                // Check if we've had a resize since the most recent rendering
                if self.resized {
                    // Re-configure the swap buffers
                    self.surface.configure(&self.device,
                            &SurfaceConfiguration {
                        // Usage for the swap chain. In this case, this is
                        // currently the only supported option.
                        usage: TextureUsages::RENDER_ATTACHMENT |
                            TextureUsages::COPY_SRC | TextureUsages::COPY_DST,

                        // Set the preferred texture format for the swap chain
                        // to be what the surface and adapter want.
                        format: self.swapchain_format,

                        // Set the width of the swap chain
                        width: self.width,

                        // Set the height of the swap chain
                        height: self.height,

                        // The way data is presented to the screen
                        // `Immediate` (no vsync)
                        // `Mailbox`   (no vsync for rendering,
                        //              but frames synced on vsync)
                        // `Fifo`      (full vsync)
                        present_mode: match self.vsync {
                            Vsync::Off => PresentMode::Immediate,
                            Vsync::On  => PresentMode::Fifo,
                        },
                    });

                    // Create output
                    let (output_texture, output_view, output_depth,
                            output_depth_view) = Self::create_texture_pair(
                        &mut self.device, self.width, self.height,
                        self.msaa_level as u32,
                        self.swapchain_format
                    );

                    // Create scene save buffer
                    let (scene_texture, _scene_view, scene_depth,
                            _scene_depth_view) = Self::create_texture_pair(
                        &mut self.device, self.width, self.height,
                        self.msaa_level as u32,
                        self.swapchain_format
                    );
                    
                    // Create background buffer
                    let (background_texture, _, background_depth, _) =
                            Self::create_texture_pair(
                        &mut self.device, self.width, self.height,
                        self.msaa_level as u32,
                        self.swapchain_format
                    );

                    // Replace textures with the new ones
                    self.output_texture = output_texture;
                    self.output_depth = output_depth;
                    self.output_view = output_view;
                    self.output_depth_view = output_depth_view;
                    self.scene_texture = scene_texture;
                    self.scene_depth = scene_depth;
                    self.background_texture = background_texture;
                    self.background_depth = background_depth;

                    // Need a scene redraw
                    self.incremental = false;

                    // No longer need a resize
                    self.resized = false;
                }

                // Get the next available surface in the swap chain
                let mut frame = self.surface
                    .get_current_texture()
                    .map_err(Error::GetSurfaceTexture)?;

                // Draw scene
                self.update_camera_int();
                self.render_internal(&mut handler, false, &mut frame)?;

                // Draw UI
                self.set_ui_camera();
                self.render_internal(&mut handler, true, &mut frame)?;

                // Present the frame to the output surface
                frame.present();
            }
            Event::MainEventsCleared => {
                if let CameraState::Flight3d {
                    key_w, key_a, key_s, key_d,
                    ref mut eye, speed, pitch, yaw, ..
                } = self.camera_state {
                    // Check for movement keypresses
                    if key_w || key_a || key_s || key_d {
                        // Key is pressed, so we have to redraw so we can
                        // handle movement

                        // Determine movement for forwards and strafing
                        let forwards = match (key_w, key_s) {
                            (true, true)   =>      0.,
                            (true, false)  =>  speed,
                            (false, true)  => -speed,
                            (false, false) =>      0.,
                        };
                        let strafe = match (key_a, key_d) {
                            (true, true)   =>      0.,
                            (true, false)  =>  speed,
                            (false, true)  => -speed,
                            (false, false) =>      0.,
                        };

                        // Update the camera eye position...

                        // Compute the vector which is the direction the camera
                        // is facing
                        let direction = Vector3::new(
                            pitch.cos() * yaw.sin(), pitch.sin(),
                            pitch.cos() * yaw.cos());

                        // Compute the vector which would give strafing to the
                        // left of the camera
                        let strafe_pitch = Deg(0.);
                        let strafe_yaw   = yaw + Deg(90.);
                        let strafe_direction = Vector3::new(
                            strafe_pitch.cos() * strafe_yaw.sin(),
                            strafe_pitch.sin(),
                            strafe_pitch.cos() * strafe_yaw.cos());

                        // Update camera position
                        *eye += direction * forwards;
                        *eye += strafe_direction * strafe;
                        self.update_camera();
                    }
                }

                // Check if we should schedule another frame for drawing
                handler.should_redraw(self);
            }
            Event::UserEvent(incremental) => {
                // Got a remote request to redraw the screen
                self.request_redraw(incremental);
            }
            _ => {
                // Unhandled event
            }
        }

        // Put the handler back
        self.handler = Some(handler);

        Ok(())
    }

    /// Perform rendering passes
    fn render_internal(&mut self, handler: &mut EH, ui: bool,
            frame: &mut SurfaceTexture) -> Result<()> {
        // Clear temporary storage for text commands
        self.text_temp.clear();

        if ui {
            // Notify about the frame
            handler.render_ui(self);
        } else {
            // Notify about the frame
            handler.render(self, self.incremental);
        }

        // Send text data to the GPU
        let buffer = self.create_texture_vertex_buffer();
        self.text_commands.push((buffer,
            0..self.text_temp.len() as u32));

        // Create a view of the texture used in the frame
        let view = frame.texture
            .create_view(&TextureViewDescriptor::default());

        // Create an encoder for a series of GPU operations
        let mut encoder = self.device.create_command_encoder(
            &CommandEncoderDescriptor::default());

        if ui {
            encoder.copy_texture_to_texture(
                self.background_texture.as_image_copy(),
                if self.msaa_level == Msaa::None {
                    frame.texture.as_image_copy()
                } else {
                    self.output_texture.as_image_copy()
                },
                Extent3d {
                    width:  self.width,
                    height: self.height,
                    depth_or_array_layers: 1
                });
        }

        if !ui {
            // If we're doing an incremental render, copy the last rendered
            // scene as the base for the new rendering
            if self.incremental {
                encoder.copy_texture_to_texture(
                    self.scene_texture.as_image_copy(),
                    if self.msaa_level == Msaa::None {
                        frame.texture.as_image_copy()
                    } else {
                        self.output_texture.as_image_copy()
                    },
                    Extent3d {
                        width:  self.width,
                        height: self.height,
                        depth_or_array_layers: 1
                    });

                encoder.copy_texture_to_texture(
                    self.scene_depth.as_image_copy(),
                    self.output_depth.as_image_copy(),
                    Extent3d {
                        width:  self.width,
                        height: self.height,
                        depth_or_array_layers: 1
                    });
            } else {
                // Start a render pass
                let mut render_pass = encoder.begin_render_pass(
                    &RenderPassDescriptor {
                        // Debug label
                        label: None,

                        // Description of the output color buffer
                        color_attachments: &[RenderPassColorAttachment {
                            // Draw either to the MSAA buffer or the view,
                            // depending on if MSAA is enabled
                            view: match self.msaa_level {
                                Msaa::None => &view,
                                _ =>          &self.output_view,
                            },

                            // Actual screen to render to
                            resolve_target: match self.msaa_level {
                                Msaa::None => None,
                                _          => Some(&view),
                            },

                            // Clear the screen to black at the start of
                            // the rendering pass
                            ops: Operations {
                                load: if !ui && !self.incremental {
                                    LoadOp::Clear(Color::BLACK)
                                } else {
                                    LoadOp::Load
                                },
                                store: true,
                            },
                        }],

                        // Description of the depth buffer
                        depth_stencil_attachment: Some(
                                RenderPassDepthStencilAttachment {
                            // Depth buffer
                            view: &self.output_depth_view,

                            // Reset the depth buffer to `1.0` for all
                            // values at the start of the rendering pass
                            depth_ops: Some(Operations {
                                // Clear to 1.0
                                load: if !ui && !self.incremental {
                                    LoadOp::Clear(1.0)
                                } else {
                                    LoadOp::Load
                                },
                                store: true,
                            }),

                            // Operations to perform on the stencil
                            stencil_ops: None,
                        }),
                    });

                // Bind the camera
                render_pass.set_bind_group(
                    0, &self.camera_bind_group, &[]);

                // Use the triangle pipeline
                render_pass.set_pipeline(&self.triangle_pipeline);

                // Render persistant data
                for (buffer, range) in &self.persist_tri_commands {
                    // Bind the vertex buffer
                    render_pass
                        .set_vertex_buffer(0, buffer.slice(..));

                    // Draw it!
                    render_pass.draw(range.clone(), 0..1);
                }

                // Use the line pipeline
                render_pass.set_pipeline(&self.line_pipeline);

                // Render persistant data
                for (buffer, range) in &self.persist_line_commands {
                    // Bind the vertex buffer
                    render_pass
                        .set_vertex_buffer(0, buffer.slice(..));

                    // Draw it!
                    render_pass.draw(range.clone(), 0..1);
                }

                // Use the point pipeline
                render_pass.set_pipeline(&self.point_pipeline);

                // Render persistant data
                for (buffer, range) in &self.persist_point_commands {
                    // Bind the vertex buffer
                    render_pass
                        .set_vertex_buffer(0, buffer.slice(..));

                    // Draw it!
                    render_pass.draw(range.clone(), 0..1);
                }

                // Done with the render pass
                drop(render_pass);

                // Save the rendering
                encoder.copy_texture_to_texture(
                    if self.msaa_level == Msaa::None {
                        frame.texture.as_image_copy()
                    } else {
                        self.output_texture.as_image_copy()
                    },
                    self.scene_texture.as_image_copy(),
                    Extent3d {
                        width:  self.width,
                        height: self.height,
                        depth_or_array_layers: 1
                    });

                encoder.copy_texture_to_texture(
                    self.output_depth.as_image_copy(),
                    self.scene_depth.as_image_copy(),
                    Extent3d {
                        width:  self.width,
                        height: self.height,
                        depth_or_array_layers: 1
                    });
            }
        }

        // Incremental render pass
        {
            // Start a render pass
            let mut render_pass = encoder.begin_render_pass(
                &RenderPassDescriptor {
                    // Debug label
                    label: None,

                    // Description of the output color buffer
                    color_attachments: &[RenderPassColorAttachment {
                        // Draw either to the MSAA buffer or the view,
                        // depending on if MSAA is enabled
                        view: match self.msaa_level {
                            Msaa::None => &view,
                            _ =>          &self.output_view,
                        },

                        // Actual screen to render to
                        resolve_target: match self.msaa_level {
                            Msaa::None => None,
                            _          => Some(&view),
                        },

                        // Don't clear color data
                        ops: Operations {
                            load:  LoadOp::Load,
                            store: true,
                        },
                    }],

                    // Description of the depth buffer
                    depth_stencil_attachment: Some(
                            RenderPassDepthStencilAttachment {
                        // Depth buffer
                        view: &self.output_depth_view,

                        // Reset the depth buffer to `1.0` for all
                        // values at the start of the rendering pass
                        depth_ops: Some(Operations {
                            // Clear the depth buffer if it's a UI render so
                            // we always draw on top of everything
                            load: if ui {
                                LoadOp::Clear(1.0)
                            } else {
                                LoadOp::Load
                            },
                            store: true,
                        }),

                        // Operations to perform on the stencil
                        stencil_ops: None,
                    }),
                });

            // Bind the camera
            render_pass.set_bind_group(
                0, &self.camera_bind_group, &[]);

            // Use the triangle pipeline
            render_pass.set_pipeline(&self.triangle_pipeline);

            // Render temporary data
            for (buffer, range) in &self.tri_commands {
                // Bind the vertex buffer
                render_pass
                    .set_vertex_buffer(0, buffer.slice(..));

                // Draw it!
                render_pass.draw(range.clone(), 0..1);
            }

            // Use the line pipeline
            render_pass.set_pipeline(&self.line_pipeline);

            // Render temporary data
            for (buffer, range) in &self.line_commands {
                // Bind the vertex buffer
                render_pass
                    .set_vertex_buffer(0, buffer.slice(..));

                // Draw it!
                render_pass.draw(range.clone(), 0..1);
            }

            // Use the point pipeline
            render_pass.set_pipeline(&self.point_pipeline);

            // Render temporary data
            for (buffer, range) in &self.point_commands {
                // Bind the vertex buffer
                render_pass
                    .set_vertex_buffer(0, buffer.slice(..));

                // Draw it!
                render_pass.draw(range.clone(), 0..1);
            }

            // Switch to text drawing and draw all text
            render_pass.set_bind_group(0,
                &self.texture_bind_group, &[]);
            render_pass.set_pipeline(&self.texture_pipeline);
            for (buffer, range) in &self.text_commands {
                render_pass.set_vertex_buffer(0, buffer.slice(..));
                render_pass.draw(range.clone(), 0..1);
            }
        }

        // Finalize the encoder and submit the buffer for execution
        self.queue.submit(Some(encoder.finish()));

        // Done with commands, discard them
        self.persist_tri_commands.clear();
        self.tri_commands.clear();
        self.persist_line_commands.clear();
        self.line_commands.clear();
        self.persist_point_commands.clear();
        self.point_commands.clear();
        self.text_commands.clear();

        // Now that the frame was drawn, we can go back to incremental
        // mode
        self.incremental = true;

        Ok(())
    }

    /// Set the window visible and start the event loop
    pub fn run<'a>(mut self) -> ! {
        // Register the event handler
        self.handler = Some(<EH>::create(&mut self));

        // Make the window visible
        self.window.set_visible(true);

        self.event_loop.take().unwrap().run(move |event, _, control_flow| {
            // Handle events forever, unless we get an error
            self.handle_event(event, control_flow)
                .expect("Failed to handle event");
        });
    }
}

