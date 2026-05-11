<script lang="ts">
  import Tabular from '../tabular/Tabular.svelte';
  let mode: 'f1_routed' | 'v5_global' = 'f1_routed';
</script>

<style lang="scss">
  @import './WebShap.scss';
</style>

<div class="webshap-page">
  <!-- <Tooltip {tooltipStore} /> -->

  <div class="app-wrapper">
    <div class="mode-switch">
      <button
        class:selected={mode === 'f1_routed'}
        on:click={() => (mode = 'f1_routed')}
      >
        F1 Routed
      </button>
      <button
        class:selected={mode === 'v5_global'}
        on:click={() => (mode = 'v5_global')}
      >
        V5 Global
      </button>
    </div>
    <div class="main-app">
      <div class="main-app-container">
        {#key mode}
          <Tabular {mode} />
        {/key}
      </div>
    </div>
  </div>
</div>
