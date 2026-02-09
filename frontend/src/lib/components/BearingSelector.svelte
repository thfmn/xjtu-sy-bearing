<!--
  Copyright (C) 2026 by Tobias Hoffmann
  thoffmann-ml@proton.me | MIT License | 2025-2026
  xjtu-sy-bearing onset and RUL prediction ML Pipeline
-->
<script lang="ts">
  import { onMount } from "svelte";
  import { getDatasetInfo } from "../api";
  import { CONDITIONS, CONDITION_LABELS } from "../constants";
  import type { ConditionInfo } from "../types";

  let {
    condition,
    bearing,
    onSelect,
  }: {
    condition: string;
    bearing: string;
    onSelect: (condition: string, bearingId: string) => void;
  } = $props();

  let conditions: ConditionInfo[] = $state([]);

  let bearingsForCondition = $derived(
    conditions.find((c) => c.condition === condition)?.bearings ?? [],
  );

  onMount(async () => {
    try {
      const info = await getDatasetInfo();
      conditions = info.conditions;
    } catch {
      conditions = Object.entries(CONDITIONS).map(([cond, bearings]) => ({
        condition: cond,
        bearings,
      }));
    }
    // Auto-select first bearing if nothing is selected yet
    if (!condition && conditions.length > 0) {
      onSelect(conditions[0].condition, conditions[0].bearings[0]);
    }
  });

  function handleConditionChange(e: Event) {
    const value = (e.target as HTMLSelectElement).value;
    const entry = conditions.find((c) => c.condition === value);
    if (entry && entry.bearings.length > 0) {
      onSelect(value, entry.bearings[0]);
    }
  }

  function handleBearingChange(e: Event) {
    const value = (e.target as HTMLSelectElement).value;
    onSelect(condition, value);
  }
</script>

<div class="bearing-selector">
  <div class="selector-group">
    <label for="condition-select">Condition</label>
    <select
      id="condition-select"
      value={condition}
      onchange={handleConditionChange}
    >
      {#each conditions as cond}
        <option value={cond.condition}>
          {CONDITION_LABELS[cond.condition] ?? cond.condition}
        </option>
      {/each}
    </select>
  </div>

  <div class="selector-group">
    <label for="bearing-select">Bearing</label>
    <select
      id="bearing-select"
      value={bearing}
      onchange={handleBearingChange}
    >
      {#each bearingsForCondition as b}
        <option value={b}>{b}</option>
      {/each}
    </select>
  </div>
</div>

<style>
  .bearing-selector {
    display: flex;
    gap: 16px;
    align-items: flex-end;
  }

  .selector-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
</style>
