use std::collections::HashMap;
use std::path::PathBuf;

use tracel_xtask::prelude::*;

#[derive(clap::Args)]
pub struct ModelCheckArgs {
    /// Model to operate on (default: all models).
    #[arg(long)]
    pub model: Option<String>,

    #[command(subcommand)]
    pub command: Option<ModelCheckSubCommand>,
}

#[derive(clap::Subcommand)]
pub enum ModelCheckSubCommand {
    /// Download model artifacts (runs get_model.py).
    Download,
    /// Build the model-check crate.
    Build,
    /// Run the model-check binary.
    Run,
    /// Download, build, and run (default).
    All,
}

struct ModelInfo {
    dir: &'static str,
    name: &'static str,
    /// Optional (env_var, default_value) for models with selectable variants.
    env: Option<(&'static str, &'static str)>,
}

const MODELS: &[ModelInfo] = &[
    ModelInfo {
        dir: "silero-vad",
        name: "Silero VAD",
        env: None,
    },
    ModelInfo {
        dir: "all-minilm-l6-v2",
        name: "all-MiniLM-L6-v2",
        env: None,
    },
    ModelInfo {
        dir: "clip-vit-b-32-text",
        name: "CLIP ViT-B-32 text",
        env: None,
    },
    ModelInfo {
        dir: "clip-vit-b-32-vision",
        name: "CLIP ViT-B-32 vision",
        env: None,
    },
    ModelInfo {
        dir: "modernbert-base",
        name: "ModernBERT-base",
        env: None,
    },
    ModelInfo {
        dir: "rf-detr",
        name: "RF-DETR Small",
        env: None,
    },
    ModelInfo {
        dir: "albert",
        name: "ALBERT",
        env: Some(("ALBERT_MODEL", "albert-base-v2")),
    },
    ModelInfo {
        dir: "yolo",
        name: "YOLO",
        env: Some(("YOLO_MODEL", "yolov8n")),
    },
];

fn model_dir(model: &ModelInfo) -> PathBuf {
    PathBuf::from("crates/model-checks").join(model.dir)
}

fn model_envs(model: &ModelInfo) -> Option<HashMap<&str, &str>> {
    model.env.map(|(k, v)| {
        let mut m = HashMap::new();
        m.insert(k, v);
        m
    })
}

fn download(model: &ModelInfo) -> anyhow::Result<()> {
    let dir = model_dir(model);
    info!("Downloading {} artifacts...", model.name);
    run_process(
        "uv",
        &["run", "get_model.py"],
        None,
        Some(&dir),
        &format!("Failed to download {} model", model.name),
    )
}

fn build(model: &ModelInfo) -> anyhow::Result<()> {
    let dir = model_dir(model);
    let envs = model_envs(model);
    info!("Building {}...", model.name);
    run_process(
        "cargo",
        &["build"],
        envs,
        Some(&dir),
        &format!("Failed to build {} model check", model.name),
    )
}

fn run(model: &ModelInfo) -> anyhow::Result<()> {
    let dir = model_dir(model);
    let envs = model_envs(model);
    info!("Running {}...", model.name);
    run_process(
        "cargo",
        &["run"],
        envs,
        Some(&dir),
        &format!("Failed to run {} model check", model.name),
    )
}

fn all(model: &ModelInfo) -> anyhow::Result<()> {
    download(model)?;
    build(model)?;
    run(model)
}

pub fn handle_command(args: ModelCheckArgs) -> anyhow::Result<()> {
    let subcmd = args.command.unwrap_or(ModelCheckSubCommand::All);

    let models: Vec<&ModelInfo> = match &args.model {
        Some(name) => {
            let m = MODELS
                .iter()
                .find(|m| m.dir == name.as_str())
                .ok_or_else(|| {
                    let valid: Vec<&str> = MODELS.iter().map(|m| m.dir).collect();
                    anyhow::anyhow!(
                        "Unknown model '{}'. Valid models: {}",
                        name,
                        valid.join(", ")
                    )
                })?;
            vec![m]
        }
        None => MODELS.iter().collect(),
    };

    let action = match &subcmd {
        ModelCheckSubCommand::Download => download as fn(&ModelInfo) -> _,
        ModelCheckSubCommand::Build => build,
        ModelCheckSubCommand::Run => run,
        ModelCheckSubCommand::All => all,
    };

    for model in &models {
        action(model)?;
    }

    info!(
        "\x1B[32;1mModel check completed for {} model(s)\x1B[0m",
        models.len()
    );
    Ok(())
}
