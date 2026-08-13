use std::{fmt::Write as _, fs, path::Path};

use crate::{
    AppError, Result,
    benchmark::{BenchmarkReport, PqExperimentResult},
};

pub const REQUIRED_ARTIFACTS: [&str; 7] = [
    "results.json",
    "summary.csv",
    "report.html",
    "recall-vs-compression.svg",
    "recall-vs-quantization-error.svg",
    "search-time-vs-compression.svg",
    "memory-vs-recall.svg",
];

#[derive(Clone, Debug)]
struct ChartPoint {
    x: f64,
    y: f64,
    label: String,
}

pub fn write_artifacts(output: &Path, report: &BenchmarkReport) -> Result<()> {
    fs::create_dir_all(output)?;
    fs::write(
        output.join("results.json"),
        serde_json::to_string_pretty(report)?,
    )?;
    write_csv(&output.join("summary.csv"), report)?;

    let recall_compression = render_chart(
        "Recall versus raw compression",
        "Recall compared with the physical byte-per-subspace compression ratio.",
        "Raw compression ratio",
        "Recall",
        &chart_points(report, |experiment| {
            (
                experiment.memory.raw_compression_ratio,
                headline_recall(experiment),
            )
        }),
    )?;
    let recall_error = render_chart(
        "Recall versus reconstruction error",
        "Recall compared with mean total squared reconstruction error per vector.",
        "Mean squared reconstruction error per vector",
        "Recall",
        &chart_points(report, |experiment| {
            (
                experiment
                    .quality
                    .mean_squared_reconstruction_error_per_vector,
                headline_recall(experiment),
            )
        }),
    )?;
    let time_compression = render_chart(
        "ADC search time versus raw compression",
        "Simple end-to-end ADC query timing compared with raw compression.",
        "Raw compression ratio",
        "ADC search time (ms)",
        &chart_points(report, |experiment| {
            (
                experiment.memory.raw_compression_ratio,
                experiment.search.adc_total_time_ms,
            )
        }),
    )?;
    let memory_recall = render_chart(
        "Memory versus recall",
        "Amortized PQ bytes per database vector compared with recall.",
        "Amortized PQ bytes per vector",
        "Recall",
        &chart_points(report, |experiment| {
            (
                experiment.memory.amortized_pq_bytes_per_vector,
                headline_recall(experiment),
            )
        }),
    )?;

    for (name, svg) in [
        ("recall-vs-compression.svg", &recall_compression),
        ("recall-vs-quantization-error.svg", &recall_error),
        ("search-time-vs-compression.svg", &time_compression),
        ("memory-vs-recall.svg", &memory_recall),
    ] {
        fs::write(output.join(name), svg)?;
    }
    fs::write(
        output.join("report.html"),
        render_html(
            report,
            &recall_compression,
            &recall_error,
            &time_compression,
            &memory_recall,
        )?,
    )?;
    Ok(())
}

fn chart_points(
    report: &BenchmarkReport,
    values: impl Fn(&PqExperimentResult) -> (f64, f64),
) -> Vec<ChartPoint> {
    report
        .experiments
        .iter()
        .map(|experiment| {
            let (x, y) = values(experiment);
            ChartPoint {
                x,
                y,
                label: format!(
                    "M={} K={}",
                    experiment.config.subspaces, experiment.config.centroids_per_subspace
                ),
            }
        })
        .collect()
}

fn headline_recall(experiment: &PqExperimentResult) -> f64 {
    experiment
        .quality
        .recall_at_10
        .or(experiment.quality.recall_at_5)
        .or(experiment.quality.recall_at_1)
        .unwrap_or(0.0)
}

fn write_csv(path: &Path, report: &BenchmarkReport) -> Result<()> {
    const HEADERS: [&str; 32] = [
        "dataset",
        "dimension",
        "training_vectors",
        "database_vectors",
        "query_vectors",
        "seed",
        "top_k",
        "subspaces",
        "centroids_per_subspace",
        "subvector_dimension",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mean_squared_reconstruction_error_per_vector",
        "full_vector_bytes_per_vector",
        "actual_pq_code_bytes_per_vector",
        "theoretical_packed_bits_per_vector",
        "codebook_bytes",
        "amortized_codebook_bytes_per_vector",
        "amortized_pq_bytes_per_vector",
        "raw_compression_ratio",
        "amortized_compression_ratio",
        "pq_training_time_ms",
        "pq_encoding_time_ms",
        "adc_table_build_time_ms",
        "adc_code_scan_time_ms",
        "adc_total_time_ms",
        "exact_search_time_ms",
        "exact_coordinate_distance_operations",
        "adc_table_coordinate_distance_operations",
        "adc_table_lookups",
        "adc_distance_additions",
    ];
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(HEADERS)?;
    for experiment in &report.experiments {
        let optional =
            |value: Option<f64>| value.map(|number| number.to_string()).unwrap_or_default();
        writer.write_record(&[
            report.dataset.kind.to_string(),
            report.dataset.dimension.to_string(),
            report.dataset.training_vectors.to_string(),
            report.dataset.database_vectors.to_string(),
            report.dataset.query_vectors.to_string(),
            report.dataset.seed.to_string(),
            report.dataset.top_k.to_string(),
            experiment.config.subspaces.to_string(),
            experiment.config.centroids_per_subspace.to_string(),
            experiment.config.subvector_dimension.to_string(),
            optional(experiment.quality.recall_at_1),
            optional(experiment.quality.recall_at_5),
            optional(experiment.quality.recall_at_10),
            experiment
                .quality
                .mean_squared_reconstruction_error_per_vector
                .to_string(),
            experiment.memory.full_vector_bytes_per_vector.to_string(),
            experiment
                .memory
                .actual_pq_code_bytes_per_vector
                .to_string(),
            experiment
                .memory
                .theoretical_packed_bits_per_vector
                .to_string(),
            experiment.memory.codebook_bytes.to_string(),
            experiment
                .memory
                .amortized_codebook_bytes_per_vector
                .to_string(),
            experiment.memory.amortized_pq_bytes_per_vector.to_string(),
            experiment.memory.raw_compression_ratio.to_string(),
            experiment.memory.amortized_compression_ratio.to_string(),
            experiment.training.time_ms.to_string(),
            experiment.encoding.time_ms.to_string(),
            experiment.search.table_build_time_ms.to_string(),
            experiment.search.code_scan_time_ms.to_string(),
            experiment.search.adc_total_time_ms.to_string(),
            report.exact.search_time_ms.to_string(),
            report.exact.coordinate_distance_operations.to_string(),
            experiment
                .search
                .table_coordinate_distance_operations
                .to_string(),
            experiment.search.table_lookups.to_string(),
            experiment.search.distance_additions.to_string(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn render_chart(
    title: &str,
    description: &str,
    x_label: &str,
    y_label: &str,
    points: &[ChartPoint],
) -> Result<String> {
    if points.is_empty()
        || points
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return Err(AppError::InvalidConfig(
            "chart points must be non-empty and finite".into(),
        ));
    }
    let width = 900.0;
    let height = 520.0;
    let left = 90.0;
    let right = 35.0;
    let top = 55.0;
    let bottom = 80.0;
    let mut x_min = points
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let mut x_max = points
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut y_min = points
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min);
    let mut y_max = points
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    expand_domain(&mut x_min, &mut x_max);
    expand_domain(&mut y_min, &mut y_max);
    let plot_width = width - left - right;
    let plot_height = height - top - bottom;
    let map_x = |value: f64| left + (value - x_min) / (x_max - x_min) * plot_width;
    let map_y = |value: f64| top + (y_max - value) / (y_max - y_min) * plot_height;

    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width:.0} {height:.0}\" role=\"img\">"
    )
    .unwrap();
    writeln!(svg, "<title>{}</title>", escape_xml(title)).unwrap();
    writeln!(svg, "<desc>{}</desc>", escape_xml(description)).unwrap();
    writeln!(svg, "<rect width=\"100%\" height=\"100%\" fill=\"#fff\"/>").unwrap();
    writeln!(
        svg,
        "<text x=\"450\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"20\" font-weight=\"700\">{}</text>",
        escape_xml(title)
    )
    .unwrap();
    writeln!(
        svg,
        "<line x1=\"{left}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#334155\"/><line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{}\" stroke=\"#334155\"/>",
        top + plot_height,
        left + plot_width,
        top + plot_height,
        top + plot_height
    )
    .unwrap();
    for tick in 0..=5 {
        let fraction = tick as f64 / 5.0;
        let x_value = x_min + fraction * (x_max - x_min);
        let x = left + fraction * plot_width;
        let y_value = y_max - fraction * (y_max - y_min);
        let y = top + fraction * plot_height;
        writeln!(svg, "<text x=\"{x:.2}\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">{x_value:.3}</text>", top + plot_height + 22.0).unwrap();
        writeln!(svg, "<text x=\"{}\" y=\"{y:.2}\" text-anchor=\"end\" dominant-baseline=\"middle\" font-family=\"sans-serif\" font-size=\"12\">{y_value:.3}</text>", left - 10.0).unwrap();
    }
    writeln!(svg, "<text x=\"450\" y=\"500\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">{}</text>", escape_xml(x_label)).unwrap();
    writeln!(svg, "<text x=\"20\" y=\"260\" text-anchor=\"middle\" transform=\"rotate(-90 20 260)\" font-family=\"sans-serif\" font-size=\"14\">{}</text>", escape_xml(y_label)).unwrap();
    for (index, point) in points.iter().enumerate() {
        let x = map_x(point.x);
        let y = map_y(point.y);
        let label_y = if index % 2 == 0 { y - 10.0 } else { y + 19.0 };
        writeln!(svg, "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"5\" fill=\"#2563eb\"><title>{}</title></circle>", escape_xml(&format!("{}: x={:.6}, y={:.6}", point.label, point.x, point.y))).unwrap();
        writeln!(svg, "<text x=\"{x:.2}\" y=\"{label_y:.2}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\">{}</text>", escape_xml(&point.label)).unwrap();
    }
    svg.push_str("</svg>\n");
    Ok(svg)
}

fn expand_domain(minimum: &mut f64, maximum: &mut f64) {
    if (*maximum - *minimum).abs() < f64::EPSILON {
        let padding = minimum.abs().max(1.0) * 0.05;
        *minimum -= padding;
        *maximum += padding;
    } else {
        let padding = (*maximum - *minimum) * 0.08;
        *minimum -= padding;
        *maximum += padding;
    }
}

fn render_html(
    report: &BenchmarkReport,
    recall_compression: &str,
    recall_error: &str,
    time_compression: &str,
    memory_recall: &str,
) -> Result<String> {
    let representative = report
        .experiments
        .iter()
        .find(|experiment| {
            experiment.config.subspaces == 8 && experiment.config.centroids_per_subspace == 256
        })
        .or_else(|| report.experiments.last())
        .ok_or_else(|| AppError::InvalidConfig("HTML report needs an experiment".into()))?;
    let exact_query = report.exact.queries.first();
    let approximate_query = representative.queries.first();
    let best = report
        .experiments
        .iter()
        .max_by(|left, right| headline_recall(left).total_cmp(&headline_recall(right)))
        .expect("an experiment exists");

    let mut html = String::from(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Product Quantization Vector Search Laboratory</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1120px;margin:auto;padding:2rem;color:#172033;background:#f8fafc}h1,h2{color:#0f172a}.card{background:white;border:1px solid #dbe3ef;border-radius:12px;padding:1rem;margin:1rem 0}table{border-collapse:collapse;width:100%}th,td{padding:.5rem;border-bottom:1px solid #dbe3ef;text-align:left}.chart svg{width:100%;height:auto}.overlap{font-weight:700;color:#166534}code{background:#e2e8f0;padding:.1rem .3rem;border-radius:4px}.muted{color:#475569}</style></head><body>",
    );
    html.push_str("<h1>Product Quantization Vector Search Laboratory</h1>");
    html.push_str("<section><h2>Overview</h2><p>This educational experiment compares exhaustive full-precision squared-Euclidean search with Product Quantization and Asymmetric Distance Computation. Exact search is the ground truth. PQ compresses each database vector into one physical <code>u8</code> centroid ID per subspace; ADC keeps the query uncompressed and scans every code without reconstructing vectors.</p></section>");
    write!(html, "<section><h2>Dataset</h2><div class=\"card\"><p>Kind: <strong>{}</strong>; dimension: {}; training vectors: {}; database vectors: {}; queries: {}; seed: {}; top-k: {}.</p></div></section>", escape_html(&report.dataset.kind.to_string()), report.dataset.dimension, report.dataset.training_vectors, report.dataset.database_vectors, report.dataset.query_vectors, report.dataset.seed, report.dataset.top_k).unwrap();
    write!(html, "<section><h2>Exact baseline</h2><p>Exact search scanned every full <code>f32</code> vector. Its simple end-to-end wall-clock time was {:.3} ms, with {} vectors scanned and {} coordinate squared-difference operations.</p></section>", report.exact.search_time_ms, report.exact.vectors_scanned, report.exact.coordinate_distance_operations).unwrap();
    html.push_str("<section><h2>Product Quantization configurations</h2>");
    for experiment in &report.experiments {
        write!(html, "<div class=\"card\"><h3>M={} K={}</h3><p>Recall: {:.4}; reconstruction error per vector: {:.6}; physical code: {} bytes/vector; theoretical packed code: {} bits/vector; raw compression: {:.2}×; amortized compression: {:.2}×.</p><p class=\"muted\">Offline training {:.3} ms; offline encoding {:.3} ms; ADC query time {:.3} ms.</p></div>", experiment.config.subspaces, experiment.config.centroids_per_subspace, headline_recall(experiment), experiment.quality.mean_squared_reconstruction_error_per_vector, experiment.memory.actual_pq_code_bytes_per_vector, experiment.memory.theoretical_packed_bits_per_vector, experiment.memory.raw_compression_ratio, experiment.memory.amortized_compression_ratio, experiment.training.time_ms, experiment.encoding.time_ms, experiment.search.adc_total_time_ms).unwrap();
    }
    html.push_str("</section>");
    for (heading, svg) in [
        ("Recall versus compression", recall_compression),
        ("Recall versus reconstruction error", recall_error),
        ("Search time", time_compression),
        ("Memory versus recall", memory_recall),
    ] {
        write!(
            html,
            "<section class=\"chart\"><h2>{}</h2>{}</section>",
            heading,
            svg.replace(" xmlns=\"http://www.w3.org/2000/svg\"", "")
        )
        .unwrap();
    }
    html.push_str("<section><h2>Computational work</h2><p>Exact coordinate squared-difference operations and ADC table-building coordinate operations are reported separately from compact ADC table lookups and additions. A lookup is not presented as equivalent to coordinate arithmetic. Training and database encoding are offline costs and are also separate.</p>");
    write!(html, "<p>For representative M={} K={}: table coordinate operations {}; lookups {}; additions {}; exhaustive codes scanned {}.</p></section>", representative.config.subspaces, representative.config.centroids_per_subspace, representative.search.table_coordinate_distance_operations, representative.search.table_lookups, representative.search.distance_additions, representative.search.codes_scanned).unwrap();
    html.push_str("<section><h2>Selected query</h2><p>Query 0 compares exact and approximate ranks for the representative configuration.</p><table><thead><tr><th>Rank</th><th>Exact ID</th><th>Exact distance</th><th>Approximate ID</th><th>ADC distance</th></tr></thead><tbody>");
    if let (Some(exact), Some(approximate)) = (exact_query, approximate_query) {
        for rank in 0..exact.neighbors.len().min(approximate.neighbors.len()) {
            let exact_neighbor = &exact.neighbors[rank];
            let approximate_neighbor = &approximate.neighbors[rank];
            let class = if exact
                .neighbors
                .iter()
                .any(|neighbor| neighbor.vector_id == approximate_neighbor.vector_id)
            {
                " class=\"overlap\""
            } else {
                ""
            };
            write!(
                html,
                "<tr><td>{}</td><td>{}</td><td>{:.6}</td><td{}>{}</td><td>{:.6}</td></tr>",
                rank + 1,
                exact_neighbor.vector_id,
                exact_neighbor.distance,
                class,
                approximate_neighbor.vector_id,
                approximate_neighbor.distance
            )
            .unwrap();
        }
    }
    html.push_str("</tbody></table></section>");
    write!(html, "<section><h2>Interpretation</h2><p>The highest observed recall in this run was {:.4} for M={} K={}. This is a measured outcome for this synthetic configuration, not a universal ranking. Raw compression excludes shared codebooks, while amortized compression distributes codebook bytes across the database.</p></section>", headline_recall(best), best.config.subspaces, best.config.centroids_per_subspace).unwrap();
    html.push_str("<section><h2>Caveats</h2><p>The vectors are synthetic. ADC remains an exhaustive scan and does not use IVF or HNSW. Timings are simple end-to-end measurements, not rigorous microbenchmarks. This implementation has no SIMD-specific path or GPU support. Physical codes use one byte per subspace even when theoretical packing needs fewer bits.</p></section><section><h2>Future work</h2><p>Possible extensions are bit packing, SIMD ADC, IVF-PQ, OPQ, real embedding import, and a cosine or normalized-vector experiment. None is implemented here.</p></section></body></html>");
    Ok(html)
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn escape_html(value: &str) -> String {
    escape_xml(value)
}

pub fn inspect_query(
    path: &Path,
    query_id: usize,
    subspaces: Option<usize>,
    centroids: Option<usize>,
) -> Result<()> {
    let report: BenchmarkReport = serde_json::from_str(&fs::read_to_string(path)?)?;
    let exact = report
        .exact
        .queries
        .get(query_id)
        .ok_or_else(|| AppError::InvalidConfig("query index is out of range".into()))?;
    let experiment = report
        .experiments
        .iter()
        .find(|experiment| {
            subspaces.is_none_or(|value| value == experiment.config.subspaces)
                && centroids.is_none_or(|value| value == experiment.config.centroids_per_subspace)
        })
        .ok_or_else(|| AppError::InvalidConfig("matching experiment was not found".into()))?;
    let approximate = experiment
        .queries
        .get(query_id)
        .ok_or_else(|| AppError::InvalidConfig("query index is out of range".into()))?;
    println!(
        "Query {} using M={} K={} (Recall@{}={:.3})",
        query_id,
        experiment.config.subspaces,
        experiment.config.centroids_per_subspace,
        report.dataset.top_k,
        approximate.recall_at_k
    );
    for rank in 0..exact.neighbors.len().min(approximate.neighbors.len()) {
        let exact_neighbor = &exact.neighbors[rank];
        let approximate_neighbor = &approximate.neighbors[rank];
        let overlap = exact
            .neighbors
            .iter()
            .any(|neighbor| neighbor.vector_id == approximate_neighbor.vector_id);
        println!(
            "Rank {}: exact id={} distance={:.6}; approximate id={} distance={:.6}{}",
            rank + 1,
            exact_neighbor.vector_id,
            exact_neighbor.distance,
            approximate_neighbor.vector_id,
            approximate_neighbor.distance,
            if overlap { " [overlap]" } else { "" }
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        benchmark::{BenchmarkConfig, Preset, run_pipeline},
        config::{KMeansConfig, PqConfig},
        synthetic::DatasetKind,
    };

    #[test]
    fn complete_report_writes_contract_artifacts_without_nonfinite_values() {
        let unique = format!(
            "pqvs-report-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let output = std::env::temp_dir().join(unique);
        let mut config =
            BenchmarkConfig::from_preset(Preset::Quick, DatasetKind::Uniform, output.clone());
        config.dataset.training_vectors = 16;
        config.dataset.database_vectors = 20;
        config.dataset.query_vectors = 2;
        config.pq_configs = vec![PqConfig {
            dimension: 32,
            subspaces: 4,
            centroids_per_subspace: 16,
            kmeans: KMeansConfig {
                max_iterations: 2,
                convergence_epsilon: 1.0e-4,
            },
        }];
        let report = run_pipeline(&config).unwrap();
        write_artifacts(&output, &report).unwrap();
        for artifact in REQUIRED_ARTIFACTS {
            let text = fs::read_to_string(output.join(artifact)).unwrap();
            assert!(!text.contains("NaN"));
            assert!(!text.contains("inf"));
        }
        let html = fs::read_to_string(output.join("report.html")).unwrap();
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("<svg"));
        assert!(!html.contains("http://"));
        assert!(!html.contains("https://"));
        let svg = fs::read_to_string(output.join("memory-vs-recall.svg")).unwrap();
        assert!(svg.contains("<title>"));
        assert!(svg.contains("<desc>"));
        let json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(output.join("results.json")).unwrap())
                .unwrap();
        assert_eq!(json["schema_version"], 1);
        let csv = fs::read_to_string(output.join("summary.csv")).unwrap();
        assert!(csv.starts_with("dataset,dimension,training_vectors"));
        fs::remove_dir_all(output).unwrap();
    }
}
