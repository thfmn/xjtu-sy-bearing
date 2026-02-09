<!--
  Copyright (C) 2026 by Tobias Hoffmann
  thoffmann-ml@proton.me | MIT License | 2025-2026
  xjtu-sy-bearing onset and RUL prediction ML Pipeline
-->
<script lang="ts">
  import PlotlyChart from "./PlotlyChart.svelte";
  import { getScalogram, getScalogramFileCount } from "../api";
  import type { ScalogramData } from "../types";

  let { condition, bearing }: { condition: string; bearing: string } = $props();

  let loading = $state(false);
  let error = $state("");
  let fileCount = $state(0);
  let fileIdx = $state(0);
  let scalogramData: ScalogramData | null = $state(null);

  let debounceTimer: ReturnType<typeof setTimeout>;

  let hPlotData = $derived(buildPlotData(scalogramData, "h"));
  let vPlotData = $derived(buildPlotData(scalogramData, "v"));
  let hLayout = $derived(buildLayout(scalogramData, "Horizontal"));
  let vLayout = $derived(buildLayout(scalogramData, "Vertical"));

  $effect(() => {
    if (condition && bearing) {
      resetAndLoad(condition, bearing);
    }
  });

  async function resetAndLoad(cond: string, bear: string) {
    fileIdx = 0;
    scalogramData = null;
    error = "";

    try {
      fileCount = await getScalogramFileCount(cond, bear);
      if (fileCount > 0) {
        await loadScalogram();
      }
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  async function loadScalogram() {
    if (!condition || !bearing) return;
    loading = true;
    error = "";
    try {
      scalogramData = await getScalogram(condition, bearing, fileIdx);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      scalogramData = null;
    } finally {
      loading = false;
    }
  }

  function handleSlider(e: Event) {
    const value = parseInt((e.target as HTMLInputElement).value, 10);
    fileIdx = value;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      loadScalogram();
    }, 100);
  }

  function buildPlotData(
    d: ScalogramData | null,
    channel: "h" | "v",
  ): Record<string, unknown>[] {
    if (!d) return [];
    return [
      {
        z: channel === "h" ? d.h_channel : d.v_channel,
        type: "heatmap",
        colorscale: "Viridis",
        zsmooth: "best",
        colorbar: { title: "Power (dB)" },
      },
    ];
  }

  function buildLayout(
    d: ScalogramData | null,
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

    return {
      title: `${chLabel} \u2014 ${bearing} (File ${fileIdx})`,
      yaxis: {
        title: "Frequency (Hz)",
        tickvals: freqTickPositions,
        ticktext: freqTicks.map((f) =>
          f >= 1000 ? `${f / 1000}k` : String(f),
        ),
      },
      xaxis: { title: "Time Bin" },
      height: 400,
      margin: { t: 40, b: 50, l: 60, r: 20 },
    };
  }
</script>

<div class="scalogram-viewer">
  {#if fileCount > 0}
    <div class="slider-row card">
      <label for="file-slider">File Index: {fileIdx} / {fileCount - 1}</label>
      <input
        id="file-slider"
        type="range"
        min="0"
        max={fileCount - 1}
        value={fileIdx}
        oninput={handleSlider}
      />
    </div>
  {/if}

  {#if loading}
    <div class="status-msg">Loading scalogram...</div>
  {:else if error}
    <div class="status-msg error">{error}</div>
  {:else if scalogramData}
    <div class="dual-scalogram">
      <PlotlyChart data={hPlotData} layout={hLayout} />
      <PlotlyChart data={vPlotData} layout={vLayout} />
    </div>
  {:else}
    <div class="status-msg">Select a bearing to view scalograms.</div>
  {/if}
</div>

<style>
  .scalogram-viewer {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .slider-row {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .dual-scalogram {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .status-msg {
    text-align: center;
    padding: 60px 16px;
    color: var(--text-secondary);
    font-size: 0.95rem;
  }

  .status-msg.error {
    color: var(--accent-red);
  }
</style>
