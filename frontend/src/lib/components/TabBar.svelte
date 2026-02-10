<!--
  Copyright (C) 2026 by Tobias Hoffmann
  thoffmann-ml@proton.me | MIT License | 2025-2026
  xjtu-sy-bearing onset and RUL prediction ML Pipeline
-->
<script lang="ts">
  let {
    activeTab,
    onTabChange,
  }: {
    activeTab: string;
    onTabChange: (tab: string) => void;
  } = $props();

  const tabs = ["Overview", "Health Explorer", "Scalogram Viewer", "Audio Analysis"];
</script>

<nav class="tab-bar">
  {#each tabs as tab}
    <button
      class="tab-btn"
      class:active={activeTab === tab}
      onclick={() => onTabChange(tab)}
    >
      {tab}
    </button>
  {/each}
</nav>

<style>
  .tab-bar {
    display: flex;
    gap: 4px;
    padding: 0 16px;
  }

  .tab-btn {
    position: relative;
    background: transparent;
    border: none;
    border-radius: 6px 6px 0 0;
    padding: 10px 20px;
    font-size: 0.875rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .tab-btn:hover {
    color: var(--text-primary);
    background: var(--hover-bg);
  }

  .tab-btn.active {
    color: var(--accent-blue);
    background: var(--bg-card);
  }

  .tab-btn.active::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--accent-blue);
    border-radius: 2px 2px 0 0;
  }
</style>
