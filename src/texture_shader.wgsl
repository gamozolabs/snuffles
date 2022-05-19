struct CameraUniform {
    view_proj: mat4x4<f32>;
};

[[group(0), binding(0)]]
var<uniform> camera: CameraUniform;

[[group(0), binding(1)]]
var t_diffuse: texture_2d<f32>;

[[group(0), binding(2)]]
var s_diffuse: sampler;

struct VertexInput {
    [[location(0)]] position:   vec3<f32>;
    [[location(1)]] tex_coords: vec2<f32>;
    [[location(2)]] color:      vec4<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]]       tex_coords:    vec2<f32>;
    [[location(1)]]       color:         vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(model: VertexInput) -> VertexOutput {
    // Create output to fragment shader
    var out: VertexOutput;

    // Propagate the texture coords
    out.tex_coords = model.tex_coords;

    // Apply the transform
    out.clip_position =
        camera.view_proj * vec4<f32>(model.position, 1.0);

    out.color = model.color;

    // Return the output structure
    return out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords) * in.color;
}

