<script lang="ts">
import d3 from '../../utils/d3-import';
import { Tabular } from './Tabular';
  import { onMount } from 'svelte';
  import iconBox from '../../imgs/icon-box.svg?raw';
  import iconRefresh from '../../imgs/icon-refresh2.svg?raw';
  import iconCheck from '../../imgs/icon-check.svg?raw';
  import iconCross from '../../imgs/icon-cross.svg?raw';
  import iconOpen from '../../imgs/icon-open.svg?raw';

  let component: HTMLElement | null = null;
  let mounted = false;
  let initialized = false;
  let myTabular: Tabular | null = null;
  let selectedChannel = 'Speed';
  let displayedFeatureByKey: Record<string, number> = {};
  const channelOptions = [
    'Speed',
    'Throttle',
    'RPM',
    'nGear',
    'DRS',
    'Distance'
  ];
  const sortingFeatures: Array<{
    key: string;
    label: string;
    step: number;
  }> = [
    { key: 'n_elements', label: 'Array length', step: 1 },
    { key: 'length_norm', label: 'Length norm', step: 0.0001 },
    {
      key: 'adj_sorted_ratio',
      label: 'Adjacent sorted ratio',
      step: 0.0001
    },
    {
      key: 'duplicate_ratio',
      label: 'Duplicate ratio',
      step: 0.0001
    },
    {
      key: 'dispersion_ratio',
      label: 'Dispersion ratio',
      step: 0.0001
    },
    { key: 'runs_ratio', label: 'Runs ratio', step: 0.0001 },
    {
      key: 'inversion_ratio',
      label: 'Inversion ratio',
      step: 0.0001
    },
    {
      key: 'entropy_ratio',
      label: 'Entropy ratio',
      step: 0.0001
    },
    { key: 'skewness_t', label: 'Skewness', step: 0.0001 },
    {
      key: 'kurtosis_excess_t',
      label: 'Kurtosis excess',
      step: 0.0001
    },
    {
      key: 'longest_run_ratio',
      label: 'Longest run ratio',
      step: 0.0001
    },
    { key: 'iqr_norm', label: 'IQR norm', step: 0.0001 },
    { key: 'mad_norm', label: 'MAD norm', step: 0.0001 },
    {
      key: 'top1_freq_ratio',
      label: 'Top-1 frequency',
      step: 0.0001
    },
    {
      key: 'top5_freq_ratio',
      label: 'Top-5 frequency',
      step: 0.0001
    },
    { key: 'outlier_ratio', label: 'Outlier ratio', step: 0.0001 },
    {
      key: 'mean_abs_diff_norm',
      label: 'Mean absolute diff',
      step: 0.0001
    }
  ];

  const benefits = ['Channel', 'Prediction', 'VBS/SBS'];
  let shownBenefits: string[] = [];
  const visibleSortingFeatures = () =>
    sortingFeatures.filter((f) => f.key !== 'outlier_ratio');
  const formatFeatureValue = (key: string, value: number): number => {
    if (!Number.isFinite(value)) return 0;
    if (key === 'n_elements') return Math.round(value);
    return Math.round(value * 100) / 100;
  };
  const refreshDisplayedFeatures = () => {
    if (!myTabular || !myTabular.data) return;
    const names = myTabular.data.featureNames || [];
    const vals =
      myTabular.featureValues && myTabular.featureValues.length > 0
        ? myTabular.featureValues
        : myTabular.curX || [];
    const next: Record<string, number> = {};
    for (let i = 0; i < names.length; i++) {
      next[names[i]] = Number.isFinite(vals[i]) ? Number(vals[i]) : 0;
    }
    displayedFeatureByKey = next;
  };
  const onFeatureChange = (key: string, event: Event) => {
    if (!myTabular) return;
    const input = event.currentTarget as HTMLInputElement | null;
    if (!input) return;
    const names = myTabular.data?.featureNames || [];
    const idx = names.indexOf(key);
    if (idx < 0) return;
    myTabular.setFeatureValue(
      idx,
      Number(input.value)
    );
    refreshDisplayedFeatures();
  };

  onMount(() => {
    mounted = true;

    const timeGap = 420;
    for (let i = 0; i < benefits.length; i++) {
      setTimeout(() => {
        shownBenefits.push(benefits[i]);
        shownBenefits = shownBenefits;
      }, 500 + timeGap * i);
    }
  });

  const tabularUpdated = () => {
    myTabular = myTabular;
    refreshDisplayedFeatures();
  };

  const predFormatter = d3.format('.2%');

  /**
   * Initialize the embedding view.
   */
  const initView = () => {
    initialized = true;

    if (component) {
      myTabular = new Tabular({ component, tabularUpdated });
      myTabular.setChannel(selectedChannel);
    }
  };

  $: mounted && !initialized && component && initView();
</script>

<style lang="scss">
  @import './Tabular.scss';
</style>

<div class="tabular-wrapper" bind:this="{component}">
  <div class="tabular">
    <div class="top-section feature">
      <span class="section-name">Input Data</span>
      <span class="section-description"
        >F1 {selectedChannel} row #{String(myTabular ? myTabular.curIndex : 0).padStart(3, '0')} info
      </span>
      <div
        class="svg-icon rect-button"
        on:click="{() => {
          if (myTabular) myTabular.sampleClicked();
        }}"
      >
        {@html iconRefresh}
      </div>
    </div>

    <div class="feature-box">
      <div class="feature-section">
        <span class="feature-header cat">Feature Flag</span>
        <div class="content-cat">
          <div class="input-wrapper">
            <span class="name">Channel</span>
            <select
              class="feature-select"
              bind:value="{selectedChannel}"
              on:change="{() => {
                if (myTabular) myTabular.setChannel(selectedChannel);
              }}"
            >
              {#each channelOptions as channel}
                <option value="{channel}">{channel}</option>
              {/each}
            </select>
          </div>
        </div>
      </div>

      <div class="feature-section">
        <span class="feature-header cont">Extracted Features</span>
        <div class="content-cont">
          {#each visibleSortingFeatures() as item}
            <div class="input-wrapper">
              <span class="name">{item.label}</span>
              <input
                class="feature-input"
                type="number"
                step="{item.step}"
                value="{formatFeatureValue(
                  item.key,
                  Number.isFinite(displayedFeatureByKey[item.key])
                    ? displayedFeatureByKey[item.key]
                    : 0
                )}"
                on:change="{(e) => onFeatureChange(item.key, e)}"
              />
            </div>
          {/each}
        </div>
      </div>
    </div>

    <div class="data-model-arrow">
      <div class="background">
        <span class="line-loader hidden"></span>
        <div class="start-rectangle"></div>
        <div class="content-box">
          <div class="line">
            <span class="svg-icon no-pointer">
              {@html iconBox}
            </span>
            <span class="name"> ML Model </span>
          </div>

          <div class="line">
            <span class="model"> XGBoost </span>
          </div>

          <div class="loader-container hidden">
            <div class="circle-loader"></div>
            <span class="loader-label">Loading model</span>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>
    </div>

    <div class="model-explain-arrow">
      <div class="background">
        <span class="line-loader hidden"></span>
        <div class="start-rectangle"></div>
        <div class="content-box">
          <span class="name">
            {#if myTabular && myTabular.isPredictionCorrect() === true}
              Correct
            {:else if myTabular && myTabular.isPredictionCorrect() === false}
              Incorrect
            {:else}
              --
            {/if}
          </span>
          <div class="loader-container hidden">
            <div class="circle-loader"></div>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>

      <div class="benefit-panel">
        {#each benefits as benefit}
          <div class="line" class:hidden="{!shownBenefits.includes(benefit)}">
            <span class="svg-icon no-pointer">{@html iconCheck}</span>
            <span>{benefit}</span>
          </div>
        {/each}
      </div>
    </div>

    <div class="top-section output">
      <span class="section-name">Model Output</span>
      <span class="section-description">fastest sorting algorithm </span>
    </div>

    <div class="output-box">
      <div class="pred-number">
        {myTabular && myTabular.curPred !== null
          ? predFormatter(myTabular.curPred)
          : '00.00%'}
      </div>

      <div class="pred-bar">
        <svg class="pred-bar-svg"></svg>
      </div>

      <div class="label-container">
        <div class="label placeholder hidden">
          <span class="label-icon svg-icon no-pointer">
            {@html iconCross}
          </span>
          <span class="label-name"> -- </span>
        </div>

        <div
          class="label approval"
          class:hidden="{myTabular ? !myTabular.curPredLabel : true}"
        >
          <span class="label-icon svg-icon no-pointer">
            {@html iconCheck}
          </span>
          <span class="label-name">{myTabular?.curPredLabel || '--'}</span>
        </div>
      </div>
    </div>

    <div class="explain-content">
      <div class="arrow"></div>

      <div class="explain-component">
        <div class="top-section explain">
          <span class="svg-icon no-pointer">
            {@html iconOpen}
          </span>
          <span class="section-name"> Algorithm Selection</span>
          <span class="section-description"
            >Each feature's contribution to model's prediction
          </span>
        </div>

        <div class="explain-box">
          <div class="header">
            <span>Top 10 Important Features and</span>
            <span class="shap-label">Their SHAP Values</span>
          </div>
          <svg class="shap-svg"></svg>
          <div class="loader-container hidden">
            <div class="circle-loader"></div>
            <span class="loader-label">Computing SHAP values</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
