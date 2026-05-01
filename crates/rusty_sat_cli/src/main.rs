use rusty_sat_core::Scene;

fn main() {
    let scene = Scene::new();
    println!(
        "rusty-sat 0.1.0 - Rust-native Satpy-compatible rewrite scaffold (datasets: {})",
        scene.len()
    );
}
