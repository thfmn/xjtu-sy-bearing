<!--
  Copyright (C) 2026 by Tobias Hoffmann
  thoffmann-ml@proton.me | MIT License | 2025-2026
  xjtu-sy-bearing onset and RUL prediction ML Pipeline
-->
<script lang="ts">
  import TabBar from "./lib/components/TabBar.svelte";
  import BearingSelector from "./lib/components/BearingSelector.svelte";
  import BearingOverview from "./lib/components/BearingOverview.svelte";
  import HealthExplorer from "./lib/components/HealthExplorer.svelte";
  import ScalogramViewer from "./lib/components/ScalogramViewer.svelte";
  import AudioAnalysis from "./lib/components/AudioAnalysis.svelte";

  let activeTab = $state("Overview");
  let selectedCondition = $state("");
  let selectedBearing = $state("");

  function handleBearingSelect(condition: string, bearing: string) {
    selectedCondition = condition;
    selectedBearing = bearing;
  }
</script>

<header>
  <div class="header-top">
    <h1>XJTU-SY Bearing Explorer</h1>
    <BearingSelector
      condition={selectedCondition}
      bearing={selectedBearing}
      onSelect={handleBearingSelect}
    />
  </div>
  <TabBar {activeTab} onTabChange={(tab) => (activeTab = tab)} />
</header>

<main>
  {#if activeTab === "Overview"}
    <BearingOverview />
  {:else if activeTab === "Health Explorer"}
    <HealthExplorer condition={selectedCondition} bearing={selectedBearing} />
  {:else if activeTab === "Scalogram Viewer"}
    <ScalogramViewer condition={selectedCondition} bearing={selectedBearing} />
  {:else}
    <AudioAnalysis condition={selectedCondition} bearing={selectedBearing} />
  {/if}
</main>

<style>
  header {
    padding: 16px 16px 0;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    flex-shrink: 0;
  }

  .header-top {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 24px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }

  .header-top h1 {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  main {
    flex: 1;
    padding: 16px;
    overflow-y: auto;
  }
</style>
