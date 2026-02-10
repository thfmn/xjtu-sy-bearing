<!--
  Copyright (C) 2026 by Tobias Hoffmann
  thoffmann-ml@proton.me | MIT License | 2025-2026
  xjtu-sy-bearing onset and RUL prediction ML Pipeline
-->
<script lang="ts">
  import { getBearingOverview } from "../api";
  import { CONDITION_LABELS } from "../constants";
  import type { BearingOverviewRow } from "../types";

  let loading = $state(true);
  let error = $state("");
  let rows: BearingOverviewRow[] = $state([]);

  $effect(() => {
    loadOverview();
  });

  async function loadOverview() {
    loading = true;
    error = "";
    try {
      const resp = await getBearingOverview();
      rows = resp.rows;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }
</script>

<div class="bearing-overview">
  {#if loading}
    <div class="status-msg">Loading bearing overview...</div>
  {:else if error}
    <div class="status-msg error">{error}</div>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Bearing</th>
          <th>Condition</th>
          <th>Failure Mode</th>
          <th>Onset Index</th>
          <th>Confidence</th>
          <th>Method</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as row}
          <tr>
            <td class="mono">{row.bearing_id}</td>
            <td>{CONDITION_LABELS[row.condition] ?? row.condition}</td>
            <td class="failure-mode">{row.failure_mode || "—"}</td>
            <td class="mono">{row.onset_file_idx}</td>
            <td>
              <span class="badge badge-{row.confidence}">{row.confidence}</span>
            </td>
            <td>{row.detection_method}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</div>

<style>
  .bearing-overview {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
  }

  thead {
    background: var(--bg-secondary);
  }

  th {
    padding: 10px 14px;
    text-align: left;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border-color);
  }

  td {
    padding: 8px 14px;
    font-size: 0.875rem;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-color);
  }

  tr:last-child td {
    border-bottom: none;
  }

  tr:hover td {
    background: var(--hover-bg);
  }

  .mono {
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.825rem;
  }

  .failure-mode {
    color: var(--accent-blue);
    font-weight: 500;
  }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: capitalize;
  }

  .badge-high {
    background: rgba(44, 160, 44, 0.15);
    color: var(--accent-green);
  }

  .badge-medium {
    background: rgba(255, 127, 14, 0.15);
    color: #ff7f0e;
  }

  .badge-low {
    background: rgba(214, 39, 40, 0.15);
    color: var(--accent-red);
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
