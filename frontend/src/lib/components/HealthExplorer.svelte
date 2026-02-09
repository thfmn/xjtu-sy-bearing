<!--
  Copyright (C) 2026 by Tobias Hoffmann
  thoffmann-ml@proton.me | MIT License | 2025-2026
  xjtu-sy-bearing onset and RUL prediction ML Pipeline
-->
<script lang="ts">
  import PlotlyChart from "./PlotlyChart.svelte";
  import { getHealthIndicators, getPanoramicScalogram } from "../api";
  import { CONDITION_LABELS } from "../constants";
  import type { HealthIndicatorData, PanoramicScalogramData } from "../types";

  let { condition, bearing }: { condition: string; bearing: string } = $props();

  let loading = $state(false);
  let error = $state("");
  let hiData: HealthIndicatorData | null = $state(null);
  let panoramicLoading = $state(false);
  let panoramicData: PanoramicScalogramData | null = $state(null);

  let plotData = $derived(buildPlotData(hiData));
  let plotLayout = $derived(buildLayout(hiData));
  let hPanoData = $derived(buildPanoPlotData(panoramicData, "h"));
  let vPanoData = $derived(buildPanoPlotData(panoramicData, "v"));
  let hPanoLayout = $derived(buildPanoLayout(panoramicData, "Horizontal"));
  let vPanoLayout = $derived(buildPanoLayout(panoramicData, "Vertical"));
  let panoInfo = $derived(buildPanoInfo(panoramicData));

  $effect(() => {
    if (condition && bearing) {
      loadData(condition, bearing);
    }
  });

  async function loadData(cond: string, bear: string) {
    loading = true;
    error = "";
    hiData = null;
    panoramicData = null;

    try {
      hiData = await getHealthIndicators(cond, bear);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }

    // Load panoramic scalogram in the background (may be slow for large bearings)
    panoramicLoading = true;
    try {
      panoramicData = await getPanoramicScalogram(cond, bear);
    } catch {
      // Panoramic is non-critical — silently skip if CWT files missing
    } finally {
      panoramicLoading = false;
    }
  }

  function buildPlotData(
    d: HealthIndicatorData | null,
  ): Record<string, unknown>[] {
    if (!d) return [];
    return [
      {
        x: d.file_indices,
        y: d.kurtosis_avg,
        name: "Kurtosis (H+V avg)",
        type: "scatter",
        mode: "lines",
        line: { color: "#d62728" },
        yaxis: "y",
      },
      {
        x: d.file_indices,
        y: d.rms_avg,
        name: "RMS (H+V avg)",
        type: "scatter",
        mode: "lines",
        line: { color: "#1f77b4" },
        yaxis: "y2",
      },
    ];
  }

  function buildLayout(d: HealthIndicatorData | null): Record<string, unknown> {
    if (!d) return {};

    const condLabel = CONDITION_LABELS[d.condition] ?? d.condition;
    const lastIdx = d.file_indices[d.file_indices.length - 1] ?? 0;

    const shapes: Record<string, unknown>[] = [];
    const annotations: Record<string, unknown>[] = [];

    if (d.onset_manual) {
      const idx = d.onset_manual.file_idx;
      shapes.push({
        type: "line",
        x0: idx, x1: idx, y0: 0, y1: 1, yref: "paper",
        line: { color: "#4a9eff", width: 2, dash: "dash" },
      });
      shapes.push({
        type: "rect",
        x0: 0, x1: idx, y0: 0, y1: 1, yref: "paper",
        fillcolor: "rgba(0,128,0,0.05)", line: { width: 0 }, layer: "below",
      });
      shapes.push({
        type: "rect",
        x0: idx, x1: lastIdx, y0: 0, y1: 1, yref: "paper",
        fillcolor: "rgba(255,0,0,0.05)", line: { width: 0 }, layer: "below",
      });
      annotations.push({
        x: idx, y: 1, yref: "paper", xanchor: "left", yanchor: "top",
        text: `Manual (${idx})`, showarrow: false,
        font: { color: "#4a9eff", size: 11 }, xshift: 4, yshift: -4,
      });
    }

    if (d.onset_auto) {
      const idx = d.onset_auto.file_idx;
      shapes.push({
        type: "line",
        x0: idx, x1: idx, y0: 0, y1: 1, yref: "paper",
        line: { color: "#d62728", width: 2, dash: "dashdot" },
      });
      annotations.push({
        x: idx, y: 1, yref: "paper", xanchor: "right", yanchor: "top",
        text: `Auto (${idx})`, showarrow: false,
        font: { color: "#d62728", size: 11 }, xshift: -4, yshift: -4,
      });
    }

    return {
      title: `Health Indicators \u2014 ${d.bearing_id} (${condLabel})`,
      xaxis: { title: "File Index" },
      yaxis: { title: "Kurtosis (avg)", titlefont: { color: "#d62728" } },
      yaxis2: {
        title: "RMS (avg)",
        titlefont: { color: "#1f77b4" },
        overlaying: "y",
        side: "right",
      },
      hovermode: "x unified",
      height: 500,
      shapes,
      annotations,
      margin: { t: 50, b: 50, l: 60, r: 60 },
    };
  }

  // --- Panoramic scalogram helpers ---

  function buildPanoPlotData(
    d: PanoramicScalogramData | null,
    channel: "h" | "v",
  ): Record<string, unknown>[] {
    if (!d) return [];
    return [
      {
        z: channel === "h" ? d.h_panorama : d.v_panorama,
        x: d.file_indices,
        type: "heatmap",
        colorscale: "Viridis",
        zsmooth: "best",
        colorbar: { title: "Mean Power (dB)" },
      },
    ];
  }

  function buildPanoLayout(
    d: PanoramicScalogramData | null,
    chLabel: string,
  ): Record<string, unknown> {
    if (!d) return {};

    const freqTicks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
    const logMin = Math.log10(d.freq_min);
    const logMax = Math.log10(d.freq_max);
    const numScales = d.num_scales;
    const freqTickPositions = freqTicks.map((f) => {
      const logF = Math.log10(f);
      return ((logF - logMin) / (logMax - logMin)) * (numScales - 1);
    });

    const shapes: Record<string, unknown>[] = [];
    const annotations: Record<string, unknown>[] = [];

    if (d.onset_manual) {
      const idx = d.onset_manual.file_idx;
      shapes.push({
        type: "line",
        x0: idx, x1: idx, y0: 0, y1: numScales - 1,
        line: { color: "#4a9eff", width: 2, dash: "dash" },
      });
      annotations.push({
        x: idx, y: numScales - 1, xanchor: "left", yanchor: "top",
        text: `Manual (${idx})`, showarrow: false,
        font: { color: "#4a9eff", size: 11 }, xshift: 4, yshift: -4,
      });
    }

    if (d.onset_auto) {
      const idx = d.onset_auto.file_idx;
      shapes.push({
        type: "line",
        x0: idx, x1: idx, y0: 0, y1: numScales - 1,
        line: { color: "#ff4444", width: 2, dash: "dashdot" },
      });
      annotations.push({
        x: idx, y: numScales - 1, xanchor: "right", yanchor: "top",
        text: `Auto (${idx})`, showarrow: false,
        font: { color: "#ff4444", size: 11 }, xshift: -4, yshift: -4,
      });
    }

    return {
      title: `CWT Lifetime \u2014 ${chLabel} Channel`,
      yaxis: {
        title: "Frequency (Hz)",
        tickvals: freqTickPositions,
        ticktext: freqTicks.map((f) =>
          f >= 1000 ? `${f / 1000}k` : String(f),
        ),
      },
      xaxis: { title: "File Index (Bearing Lifetime)" },
      height: 350,
      margin: { t: 40, b: 50, l: 60, r: 20 },
      shapes,
      annotations,
    };
  }

  function buildPanoInfo(d: PanoramicScalogramData | null): string {
    if (!d) return "";
    const sampled = d.file_indices.length;
    if (d.step > 1) {
      return `${d.total_files} files total, showing ${sampled} (every ${d.step}th file)`;
    }
    return `${d.total_files} files (all loaded)`;
  }
</script>

<div class="health-explorer">
  {#if loading}
    <div class="status-msg">Loading health indicators...</div>
  {:else if error}
    <div class="status-msg error">{error}</div>
  {:else if hiData}
    <PlotlyChart data={plotData} layout={plotLayout} />

    {#if panoramicLoading}
      <div class="status-msg subtle">Loading lifetime scalogram...</div>
    {:else if panoramicData}
      <div class="panoramic-section">
        <div class="section-header">
          <h3>CWT Lifetime Scalogram</h3>
          <span class="info-text">{panoInfo}</span>
        </div>
        <PlotlyChart data={hPanoData} layout={hPanoLayout} />
        <PlotlyChart data={vPanoData} layout={vPanoLayout} />
      </div>
    {/if}
  {:else}
    <div class="status-msg">Select a bearing to view health indicators.</div>
  {/if}
</div>

<style>
  .health-explorer {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .panoramic-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
  }

  .section-header h3 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .info-text {
    font-size: 0.8rem;
    color: var(--text-secondary);
  }

  .status-msg {
    text-align: center;
    padding: 60px 16px;
    color: var(--text-secondary);
    font-size: 0.95rem;
  }

  .status-msg.subtle {
    padding: 30px 16px;
    font-size: 0.85rem;
  }

  .status-msg.error {
    color: var(--accent-red);
  }
</style>
