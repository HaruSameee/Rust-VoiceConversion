use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use vc_signal::rmvpe_mel_from_audio;

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    keep_aligned: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut output = PathBuf::from("rust_mel.csv");
    let mut keep_aligned = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => {
                let v = args.next().ok_or("--input requires a path")?;
                input = Some(PathBuf::from(v));
            }
            "--output" => {
                let v = args.next().ok_or("--output requires a path")?;
                output = PathBuf::from(v);
            }
            "--aligned" => {
                keep_aligned = true;
            }
            "--help" | "-h" => {
                return Err(help_text());
            }
            other => {
                return Err(format!("unknown argument: {other}\n\n{}", help_text()));
            }
        }
    }

    let input = input.ok_or_else(help_text)?;
    Ok(Args {
        input,
        output,
        keep_aligned,
    })
}

fn help_text() -> String {
    "Usage: cargo run -p vc-signal --bin dump_rmvpe_mel -- \
--input <wav> [--output rust_mel.csv] [--aligned]

Default output is valid (non-tail-padded) frames only.
Use --aligned to dump full aligned frames (multiple of 32)."
        .to_string()
}

fn read_wav_mono(path: &PathBuf) -> Result<(Vec<f32>, u32, u16), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1);
    let sr = spec.sample_rate.max(1);

    let mut mono = Vec::<f32>::new();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut frame = Vec::<f32>::with_capacity(channels as usize);
            for sample in reader.samples::<f32>() {
                frame.push(sample?);
                if frame.len() == channels as usize {
                    let sum: f32 = frame.iter().copied().sum();
                    mono.push(sum / channels as f32);
                    frame.clear();
                }
            }
        }
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample.clamp(1, 32);
            let max = ((1_i64 << (bits - 1)) - 1) as f32;
            let mut frame = Vec::<f32>::with_capacity(channels as usize);
            for sample in reader.samples::<i32>() {
                let v = sample? as f32 / max.max(1.0);
                frame.push(v.clamp(-1.0, 1.0));
                if frame.len() == channels as usize {
                    let sum: f32 = frame.iter().copied().sum();
                    mono.push(sum / channels as f32);
                    frame.clear();
                }
            }
        }
    }

    Ok((mono, sr, channels))
}

fn stats(values: &[f32]) -> (f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    for &v in values {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
        sum += v as f64;
    }
    (min_v, max_v, (sum / values.len() as f64) as f32)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = match parse_args() {
        Ok(v) => v,
        Err(msg) => {
            if msg.starts_with("Usage:") {
                println!("{msg}");
                return Ok(());
            }
            return Err(format!("{msg}\n").into());
        }
    };
    let (samples, sample_rate, channels) = read_wav_mono(&args.input)?;
    let mel_input = rmvpe_mel_from_audio(&samples, sample_rate);

    let dims = mel_input.mel.dim();
    let aligned_frames = dims.2;
    let frame_count = if args.keep_aligned {
        aligned_frames
    } else {
        mel_input.valid_frames.min(aligned_frames)
    };
    let bins = dims.1;

    let mut flat = Vec::<f32>::with_capacity(bins * frame_count);
    for m in 0..bins {
        for t in 0..frame_count {
            flat.push(mel_input.mel[(0, m, t)]);
        }
    }
    let (min_v, max_v, mean_v) = stats(&flat);

    let file = File::create(&args.output)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "# rmvpe_mel_csv")?;
    writeln!(writer, "# input={}", args.input.display())?;
    writeln!(writer, "# sample_rate={sample_rate}")?;
    writeln!(writer, "# channels={channels}")?;
    writeln!(writer, "# bins={bins}")?;
    writeln!(writer, "# valid_frames={}", mel_input.valid_frames)?;
    writeln!(writer, "# aligned_frames={aligned_frames}")?;
    writeln!(writer, "# dumped_frames={frame_count}")?;
    writeln!(
        writer,
        "# stats_min={min_v:.6} stats_max={max_v:.6} stats_mean={mean_v:.6}"
    )?;

    for m in 0..bins {
        for t in 0..frame_count {
            if t > 0 {
                write!(writer, ",")?;
            }
            write!(writer, "{:.8}", mel_input.mel[(0, m, t)])?;
        }
        writeln!(writer)?;
    }
    writer.flush()?;

    eprintln!(
        "[dump_rmvpe_mel] wrote {} (bins={} frames={} valid={} aligned={} min={:.3} max={:.3} mean={:.3})",
        args.output.display(),
        bins,
        frame_count,
        mel_input.valid_frames,
        aligned_frames,
        min_v,
        max_v,
        mean_v
    );

    Ok(())
}
