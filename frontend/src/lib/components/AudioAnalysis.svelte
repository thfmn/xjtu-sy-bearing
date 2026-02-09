<!--
  Copyright (C) 2026 by Tobias Hoffmann
  thoffmann-ml@proton.me | MIT License | 2025-2026
  xjtu-sy-bearing onset and RUL prediction ML Pipeline
-->
<script lang="ts">
  import PlotlyChart from "./PlotlyChart.svelte";
  import { getAudioFiles, getAudioData, getWaveform, getScalogram } from "../api";
  import { AUDIO_STAGE_LABELS } from "../constants";
  import type { AudioStage, WaveformData, ScalogramData } from "../types";

  let { condition, bearing }: { condition: string; bearing: string } = $props();

  interface StagePanel {
    stage: AudioStage;
    audioSrc: string;
    waveform: WaveformData | null;
    scalogram: ScalogramData | null;
    loading: boolean;
  }

  const STAGE_COLORS: Record<string, string> = {
    healthy_0pct: "#2ca02c",
    degrading_50pct: "#ff7f0e",
    failed_100pct: "#d62728",
  };

  let loading = $state(false);
  let error = $state("");
  let panels: StagePanel[] = $state([]);

  $effect(() => {
    if (condition && bearing) {
      loadAudioData(condition, bearing);
    }
  });

  async function loadAudioData(cond: string, bear: string) {
    loading = true;
    error = "";
    panels = [];

    try {
      const resp = await getAudioFiles(cond, bear);

      panels = resp.stages.map((stage) => ({
        stage,
        audioSrc: "",
        waveform: null,
        scalogram: null,
        loading: true,
      }));

      const stagePromises = resp.stages.map(async (stage, i) => {
        const [audioResult, wfResult, scResult] = await Promise.allSettled([
          getAudioData(cond, bear, stage.stage_key),
          getWaveform(cond, bear, stage.stage_key),
          getScalogram(cond, bear, stage.file_idx),
        ]);

        if (audioResult.status === "fulfilled") {
          panels[i].audioSrc = `data:${audioResult.value.mime_type};base64,${audioResult.value.base64}`;
        }
        if (wfResult.status === "fulfilled") {
          panels[i].waveform = wfResult.value;
        }
        if (scResult.status === "fulfilled") {
          panels[i].scalogram = scResult.value;
        }
        panels[i].loading = false;
      });

      await Promise.all(stagePromises);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }

  function waveformPlotData(
    wf: WaveformData,
    stageKey: string,
  ): Record<string, unknown>[] {
    const label = AUDIO_STAGE_LABELS[stageKey] ?? stageKey;
    return [
      {
        x: wf.time_ms,
        y: wf.amplitude,
        type: "scatter",
        mode: "lines",
        line: { color: STAGE_COLORS[stageKey] ?? "#e0e0e0", width: 0.5 },
        name: label,
      },
    ];
  }

  function waveformLayout(stageKey: string): Record<string, unknown> {
    const label = AUDIO_STAGE_LABELS[stageKey] ?? stageKey;
    return {
      title: `Waveform \u2014 ${label}`,
      xaxis: { title: "Time (ms)" },
      yaxis: { title: "Amplitude", range: [-1, 1] },
      height: 220,
      margin: { t: 35, b: 40, l: 50, r: 20 },
    };
  }

  function scalogramPlotData(
    sc: ScalogramData,
    channel: "h" | "v",
  ): Record<string, unknown>[] {
    return [
      {
        z: channel === "h" ? sc.h_channel : sc.v_channel,
        type: "heatmap",
        colorscale: "Viridis",
        zsmooth: "best",
        colorbar: { title: "Power (dB)", len: 0.9 },
      },
    ];
  }

  function scalogramLayout(
    sc: ScalogramData,
    channel: "h" | "v",
    fileIdx: number,
  ): Record<string, unknown> {
    const chLabel = channel === "h" ? "Horizontal" : "Vertical";

    const freqTicks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
    const logMin = Math.log10(sc.freq_min);
    const logMax = Math.log10(sc.freq_max);
    const numScales = sc.num_scales;
    const freqTickPositions = freqTicks.map((f) => {
      const logF = Math.log10(f);
      return ((logF - logMin) / (logMax - logMin)) * (numScales - 1);
    });

    return {
      title: `CWT Scalogram \u2014 ${chLabel} (File ${fileIdx})`,
      yaxis: {
        title: "Frequency (Hz)",
        tickvals: freqTickPositions,
        ticktext: freqTicks.map((f) =>
          f >= 1000 ? `${f / 1000}k` : String(f),
        ),
      },
      xaxis: { title: "Time Bin" },
      height: 280,
      margin: { t: 35, b: 40, l: 60, r: 20 },
    };
  }
</script>

<div class="audio-analysis">
  {#if loading}
    <div class="status-msg">Loading audio files...</div>
  {:else if error}
    <div class="status-msg error">{error}</div>
  {:else if panels.length > 0}
    <div class="audio-grid">
      {#each panels as panel}
        <div class="card audio-panel">
          <h3 class="stage-label">
            {AUDIO_STAGE_LABELS[panel.stage.stage_key] ??
              panel.stage.stage_key}
            <span class="file-idx-badge">File {panel.stage.file_idx}</span>
          </h3>
          <audio controls src={panel.audioSrc}>
            <track kind="captions" />
          </audio>
          {#if panel.loading}
            <div class="waveform-loading">Loading waveform &amp; scalogram...</div>
          {:else}
            {#if panel.waveform}
              <PlotlyChart
                data={waveformPlotData(panel.waveform, panel.stage.stage_key)}
                layout={waveformLayout(panel.stage.stage_key)}
              />
            {/if}
            {#if panel.scalogram}
              <div class="scalogram-pair">
                <PlotlyChart
                  data={scalogramPlotData(panel.scalogram, "h")}
                  layout={scalogramLayout(panel.scalogram, "h", panel.stage.file_idx)}
                />
                <PlotlyChart
                  data={scalogramPlotData(panel.scalogram, "v")}
                  layout={scalogramLayout(panel.scalogram, "v", panel.stage.file_idx)}
                />
              </div>
            {/if}
          {/if}
        </div>
      {/each}
    </div>
  {:else}
    <div class="status-msg">Select a bearing to explore audio.</div>
  {/if}
</div>

<style>
  .audio-analysis {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .audio-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .stage-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .file-idx-badge {
    font-size: 0.75rem;
    font-weight: 400;
    color: var(--text-secondary);
    background: var(--bg-secondary);
    padding: 2px 8px;
    border-radius: 4px;
  }

  .scalogram-pair {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .waveform-loading {
    text-align: center;
    padding: 20px;
    color: var(--text-secondary);
    font-size: 0.85rem;
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
