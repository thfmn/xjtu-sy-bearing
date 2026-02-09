<!--
  Copyright (C) 2026 by Tobias Hoffmann
  thoffmann-ml@proton.me | MIT License | 2025-2026
  xjtu-sy-bearing onset and RUL prediction ML Pipeline
-->
<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import Plotly from "plotly.js-dist-min";

  let {
    data,
    layout = {},
    config = {},
  }: {
    data: Record<string, unknown>[];
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
  } = $props();

  let container: HTMLDivElement;
  let mounted = false;

  const defaultConfig: Record<string, unknown> = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
  };

  const darkLayout: Record<string, unknown> = {
    paper_bgcolor: "#1e2a45",
    plot_bgcolor: "#1e2a45",
    font: { color: "#e0e0e0" },
  };

  function mergedLayout(): Record<string, unknown> {
    return { ...darkLayout, ...layout };
  }

  function mergedConfig(): Record<string, unknown> {
    return { ...defaultConfig, ...config };
  }

  onMount(() => {
    Plotly.newPlot(container, data, mergedLayout(), mergedConfig());
    mounted = true;
  });

  $effect(() => {
    if (mounted && container) {
      const _d = data;
      const _l = layout;
      Plotly.react(container, _d, mergedLayout(), mergedConfig());
    }
  });

  onDestroy(() => {
    if (container) Plotly.purge(container);
  });
</script>

<div class="plot-container" bind:this={container}></div>
