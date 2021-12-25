struct CameraUniform {
    view_proj: mat4x4<f32>;
};

[[group(0), binding(0)]]
var<uniform> camera: CameraUniform;

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] color:    vec4<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]]       color:         vec3<f32>;
};

[[stage(vertex)]]
fn vs_main(model: VertexInput) -> VertexOutput {
    // Create output to fragment shader
    var out: VertexOutput;

    // Set the fragment shader color
    out.color = vec3<f32>(model.color.x, model.color.y, model.color.z);

    // Apply the transform
    out.clip_position =
        camera.view_proj * vec4<f32>(model.position, 1.0);

    // Return the output structure
    return out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    // Just set the color!
    return vec4<f32>(in.color, 1.);
}

