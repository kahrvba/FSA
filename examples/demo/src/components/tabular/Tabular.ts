import d3 from '../../utils/d3-import';
import { tick } from 'svelte';
import type {
  TabularData,
  TabularContFeature,
  TabularCatFeature,
  Size,
  Padding,
  SHAPRow,
  TabularWorkerMessage
} from '../../types/common-types';
import { getLatoTextWidth } from '../../utils/text-width';
import TabularWorker from './tabular-worker?worker';

const LCG = d3.randomLcg(0.20230101);

// SVG constants
const GAP = 20;
const K = 10;
const ROW_HEIGHT = 28;
const FORMAT_2 = d3.format('.4f');
const BAR_HEIGHT = ROW_HEIGHT - 8;
const SHAP_DOMAIN_PAD = 1.08;

type ChannelRuntimeMetrics = {
  rows: number;
  sbs_algorithm: string;
  t_selector_sec: number;
  t_sbs_sec: number;
  t_vbs_sec: number;
  speedup_vs_sbs: number;
  regret_vs_vbs_sec: number;
  gap_closed: number;
};

type RowExample = {
  file: string;
  trueLabel: string;
  x: number[];
  supportedLabels: string[];
  timesSec?: Record<string, number>;
  domain?: string;
  id?: string;
  n_elements?: number;
};

type F1ChannelConfig = {
  model: string;
  classes: string[];
  rows: number;
  runtimeMetrics?: ChannelRuntimeMetrics | null;
  backgroundData?: number[][];
  examples: Array<RowExample>;
};

type SortingF1Config = {
  defaultChannel?: string;
  featureNames: string[];
  featureInfo: Record<string, {
    displayName: string;
    description: string;
    group?: string;
    requiresInt?: boolean;
  }>;
  channels: Record<string, F1ChannelConfig>;
};

type SortingV5Config = {
  mode: 'v5_global';
  featureOrder: string[];
  featureInfo: Record<string, {
    displayName: string;
    description: string;
    group?: string;
    requiresInt?: boolean;
  }>;
  classLabels: string[];
  model: string;
  runtimeMetrics?: ChannelRuntimeMetrics | null;
  backgroundData: number[][];
  rows: Array<RowExample>;
};

/**
 * Class for the Tabular WebSHAP demo
 */

export class Tabular {
  mode: 'f1_routed' | 'v5_global';
  component: HTMLElement;
  tabularUpdated: () => void;
  tabularWorker: Worker;

  // SVGs
  predBarSVG: d3.Selection<HTMLElement, unknown, null, undefined>;
  predBarSVGSize: Size;
  predBarSVGPadding: Padding;
  predBarScale: d3.ScaleLinear<number, number, never>;

  shapSVG: d3.Selection<HTMLElement, unknown, null, undefined>;
  shapSVGSize: Size;
  shapSVGPadding: Padding;
  shapScale: d3.ScaleLinear<number, number, never>;
  maxTextWidth = 200;
  maxBarWidth = 200;
  shapPlotInitialized = false;

  // Dataset
  data: TabularData | null = null;
  contFeatures: Map<string, TabularContFeature> | null = null;
  catFeatures: Map<string, TabularCatFeature> | null = null;
  curX: number[] | null = null;
  curY: number | null = null;
  curIndex = 0;

  // Model information
  curPred: number | null = null;
  curPredLabel = '';
  curClassProbs: number[] = [];
  curClassLabels: string[] = [];

  // WebSHAP data
  backgroundData: number[][] = [];
  curShapValues: number[] | null = null;
  sortingConfig: SortingF1Config | SortingV5Config | null = null;
  selectedChannel = 'Speed';
  featureValues: number[] = [];
  modelReady = false;
  loadRequestId = 0;
  pendingInference = false;
  currentRuntimeMetrics: ChannelRuntimeMetrics | null = null;
  currentExample: RowExample | null = null;

  isPredictionCorrect = (): boolean | null => {
    if (!this.currentExample || !this.currentExample.trueLabel) return null;
    if (!this.curPredLabel) return null;
    return this.curPredLabel === this.currentExample.trueLabel;
  };

  /**
   * @param args Named parameters
   * @param args.component The component
   * @param args.tabularUpdated A function to trigger updates
   */
  constructor({
    component,
    tabularUpdated,
    mode
  }: {
    component: HTMLElement;
    tabularUpdated: () => void;
    mode: 'f1_routed' | 'v5_global';
  }) {
    this.mode = mode;
    this.component = component;
    this.tabularUpdated = tabularUpdated;

    // Workers
    this.tabularWorker = new TabularWorker();
    this.tabularWorker.onmessage = (e: MessageEvent<TabularWorkerMessage>) => {
      this.tabularWorkerMessageHandler(e);
    };

    // SVGs
    this.predBarSVG = d3
      .select<HTMLElement, unknown>(this.component)
      .select('svg.pred-bar-svg');
    this.predBarSVGSize = { width: 0, height: 0 };
    this.predBarSVGPadding = { top: 4, bottom: 4, left: 10, right: 10 };
    this.predBarScale = d3.scaleLinear();

    this.shapSVG = d3
      .select<HTMLElement, unknown>(this.component)
      .select('svg.shap-svg');
    this.shapSVGSize = { width: 0, height: 0 };
    this.shapSVGPadding = { top: 1, bottom: 1, left: 10, right: 25 };
    this.shapScale = d3.scaleLinear();

    // Load the training and test dataset
    this.initData().then(() => {
      // Initialize the SVGs
      tick().then(() => {
        this.initPredBar();
      });
    });
  }

  /**
   * Flip the loading spinner for the data model arrow
   * @param isLoading If the model is loading
   */
  updateModelLoader = (isLoading: boolean, controlCircle = true) => {
    const lineLoader = this.component.querySelector(
      '.data-model-arrow .line-loader'
    ) as HTMLElement;

    const circleLoader = this.component.querySelector(
      '.data-model-arrow .loader-container'
    ) as HTMLElement;

    if (isLoading) {
      lineLoader.classList.remove('hidden');

      if (controlCircle) {
        circleLoader.classList.remove('hidden');
      }
    } else {
      lineLoader.classList.add('hidden');

      if (controlCircle) {
        circleLoader.classList.add('hidden');
      }
    }
  };

  destroy = () => {
    this.tabularWorker.terminate();
  };

  /**
   * Flip the loading spinner for the explain loaders
   * @param isLoading If the model is loading
   */
  updateExplainLoader = (isLoading: boolean) => {
    const circleLoader = this.component.querySelector(
      '.model-explain-arrow .loader-container'
    ) as HTMLElement;

    const explainBoxLoader = this.component.querySelector(
      '.explain-box .loader-container'
    ) as HTMLElement;

    const lineLoader = this.component.querySelector(
      '.model-explain-arrow .line-loader'
    ) as HTMLElement;

    if (isLoading) {
      lineLoader.classList.remove('hidden');

      if (!this.shapPlotInitialized) {
        explainBoxLoader.classList.remove('hidden');
      } else {
        circleLoader.classList.remove('hidden');
      }
    } else {
      lineLoader.classList.add('hidden');
      circleLoader.classList.add('hidden');
      explainBoxLoader.classList.add('hidden');
    }
  };

  /**
   * Handling worker messages
   * @param e Message event
   */
  tabularWorkerMessageHandler = (e: MessageEvent<TabularWorkerMessage>) => {
    switch (e.data.command) {
      case 'finishLoadModel': {
        const reqId = e.data.payload.requestId ?? -1;
        if (reqId !== this.loadRequestId) {
          break;
        }
        this.modelReady = true;
        this.updateModelLoader(false, true);
        if (this.pendingInference) {
          this.pendingInference = false;
          this.getNewExplanation();
        }
        break;
      }
      case 'failLoadModel': {
        const reqId = e.data.payload.requestId ?? -1;
        if (reqId !== this.loadRequestId) {
          break;
        }
        this.modelReady = false;
        console.error(
          `Model load failed [${e.data.payload.code || 'LOAD_MODEL_FAILED'}]:`,
          e.data.payload.error
        );
        this.updateModelLoader(false, true);
        break;
      }

      case 'finishPredict': {
        const posProbs = e.data.payload.posProbs;
        const predIndexes = e.data.payload.predIndexes || [];
        const classProbs = e.data.payload.classProbs || [];
        const firstRow = classProbs[0] || [];
        if (firstRow.length !== this.curClassLabels.length) {
          console.error(
            '[ROW_DIM_MISMATCH] class count mismatch',
            {
              expected: this.curClassLabels.length,
              got: firstRow.length,
              debug: e.data.payload.debug
            }
          );
          this.updateModelLoader(false, false);
          break;
        }
        this.curPred = posProbs[0][0];
        this.curClassProbs = firstRow;
        const idx = predIndexes[0] ?? -1;
        this.curPredLabel =
          idx >= 0 && idx < this.curClassLabels.length
            ? this.curClassLabels[idx]
            : '';
        if (e.data.payload.debug) {
          console.debug('[Predict debug]', e.data.payload.debug);
        }
        this.updatePred();
        this.updateModelLoader(false, false);

        break;
      }
      case 'failPredict': {
        console.error(
          `Prediction failed [${e.data.payload.code || 'PREDICT_FAILED'}]:`,
          e.data.payload.error
        );
        this.updateModelLoader(false, false);
        this.updateExplainLoader(false);
        break;
      }

      case 'finishExplain': {
        const shapValues = e.data.payload.shapValues;
        this.curShapValues = shapValues[0];

        if (this.shapPlotInitialized) {
          this.updateShapPlot();
        } else {
          this.initShapPlot();
        }
        this.updateExplainLoader(false);

        break;
      }
      case 'failExplain': {
        console.error(
          `Explain failed [${e.data.payload.code || 'EXPLAIN_FAILED'}]:`,
          e.data.payload.error
        );
        this.updateExplainLoader(false);
        this.updateModelLoader(false, false);
        break;
      }

      default: {
        console.error('Worker: unknown message', e.data.command);
        break;
      }
    }
  };

  getRuntimeMetricRows = (): SHAPRow[] => {
    const ex = this.currentExample;
    const predLabel = this.curPredLabel;
    const rowTimes = ex?.timesSec || {};
    const m = this.currentRuntimeMetrics;

    const labelsFromRow = Object.keys(rowTimes).filter((k) =>
      Number.isFinite(rowTimes[k])
    );
    const sbsAlgoFromMetrics = m?.sbs_algorithm || null;

    const selectedTimeFromRow =
      predLabel && Number.isFinite(rowTimes[predLabel])
        ? Number(rowTimes[predLabel])
        : null;
    const sbsTimeFromRow =
      sbsAlgoFromMetrics && Number.isFinite(rowTimes[sbsAlgoFromMetrics])
        ? Number(rowTimes[sbsAlgoFromMetrics])
        : null;
    const vbsTimeFromRow =
      labelsFromRow.length > 0
        ? Math.min(...labelsFromRow.map((k) => Number(rowTimes[k])))
        : null;

    const selectedTime =
      selectedTimeFromRow ?? m?.t_selector_sec ?? 0;
    const sbsTime = sbsTimeFromRow ?? m?.t_sbs_sec ?? selectedTime;
    const vbsTime = vbsTimeFromRow ?? m?.t_vbs_sec ?? sbsTime;

    const speedupVsSbs = sbsTime > 0 ? (sbsTime - selectedTime) / sbsTime : 0;
    const regretVsVbsSec = selectedTime - vbsTime;
    const gapDen = sbsTime - vbsTime;
    const gapClosed = Math.abs(gapDen) > 1e-18 ? (sbsTime - selectedTime) / gapDen : 0;

    return [
      {
        index: 100001,
        shap: Number.isFinite(speedupVsSbs) ? speedupVsSbs : 0,
        name: 'Selector vs SBS',
        fullName: `Selector vs SBS speedup: ${(speedupVsSbs * 100).toFixed(2)}%`
      },
      {
        index: 100002,
        shap: Number.isFinite(regretVsVbsSec)
          ? -regretVsVbsSec * 1000.0
          : 0,
        name: 'Regret vs VBS',
        fullName: `Regret vs VBS: ${(regretVsVbsSec * 1000).toFixed(4)} ms`
      },
      {
        index: 100003,
        shap: Number.isFinite(gapClosed) ? gapClosed : 0,
        name: 'Gap Closed',
        fullName: `Gap closed: ${(gapClosed * 100).toFixed(2)}%`
      }
    ];
  };

  initPredBar = () => {
    if (this.predBarSVG === null) throw Error('predBarSVG is null.');

    // Get the SVG size
    const svgBBox = this.predBarSVG.node()?.getBoundingClientRect();
    if (svgBBox !== undefined) {
      this.predBarSVGSize.width =
        svgBBox.width -
        this.predBarSVGPadding.left -
        this.predBarSVGPadding.right;
      this.predBarSVGSize.height =
        svgBBox.height -
        this.predBarSVGPadding.top -
        this.predBarSVGPadding.bottom;
    }

    const content = this.predBarSVG
      .append('g')
      .attr('class', 'content')
      .attr(
        'transform',
        `translate(${this.predBarSVGPadding.left}, ${this.predBarSVGPadding.top})`
      );

    // Create scales
    this.predBarScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([0, this.predBarSVGSize.width]);

    // Init rectangles
    content
      .append('rect')
      .attr('class', 'back-rect')
      .attr('rx', this.predBarSVGSize.height / 2)
      .attr('ry', this.predBarSVGSize.height / 2)
      .attr('width', this.predBarScale(1))
      .attr('height', this.predBarSVGSize.height);

    // Init with 0 as default pred score
    const curPred = 0;

    content
      .append('rect')
      .attr('class', 'top-rect')
      .classed('approval', curPred ? curPred >= 0.5 : true)
      .attr('rx', this.predBarSVGSize.height / 2)
      .attr('ry', this.predBarSVGSize.height / 2)
      .attr('width', this.predBarScale(curPred || 0))
      .attr('height', this.predBarSVGSize.height);

    // Add a threshold bar
    content
      .append('rect')
      .attr('class', 'threshold')
      .attr('x', this.predBarScale(0.5) - 1)
      .attr('width', 2)
      .attr('height', this.predBarSVGSize.height);
  };

  initShapPlot = () => {
    if (this.shapSVG === null) throw Error('shapSVG is null.');
    if (this.curShapValues === null) throw Error('curShapValues is null.');
    if (this.data === null) throw Error('data is null.');
    if (this.contFeatures === null) throw Error('contFeatures is null.');
    if (this.catFeatures === null) throw Error('catFeatures is null.');
    if (this.curX === null) throw Error('curX is null.');
    if (this.shapPlotInitialized)
      throw Error('shap plot is already initailized.');

    // Get the SVG size
    const svgBBox = this.shapSVG.node()?.getBoundingClientRect();
    if (svgBBox !== undefined) {
      this.shapSVGSize.width =
        svgBBox.width - this.shapSVGPadding.left - this.shapSVGPadding.right;
      this.shapSVGSize.height =
        svgBBox.height - this.shapSVGPadding.top - this.shapSVGPadding.bottom;
    }

    const content = this.shapSVG
      .append('g')
      .attr('class', 'content')
      .attr(
        'transform',
        `translate(${this.shapSVGPadding.left}, ${this.shapSVGPadding.top})`
      );

    // Decide the text and bar widths
    let maxTextWidth = 200;
    let maxBarWidth = 200;

    if (this.shapSVGSize.width - 260 - GAP > 200) {
      maxTextWidth = 260;
      maxBarWidth = this.shapSVGSize.width - GAP - maxTextWidth;
    } else {
      maxBarWidth = 220;
      maxTextWidth = this.shapSVGSize.width - GAP - maxBarWidth;
    }

    this.maxTextWidth = maxTextWidth;
    this.maxBarWidth = maxBarWidth;

    // Create scales (include runtime metric rows so bars never overflow)
    const metricRowsForScale = this.getRuntimeMetricRows();
    const absValues = [
      ...this.curShapValues.map(x => Math.abs(x)),
      ...metricRowsForScale.map((r) => Math.abs(r.shap))
    ];
    const maxAbs = d3.max(absValues)! * SHAP_DOMAIN_PAD;
    this.shapScale = d3
      .scaleLinear()
      .domain([0, maxAbs])
      .range([0, maxBarWidth / 2]);

    const shapValueScale = d3
      .scaleLinear()
      .domain([-maxAbs, maxAbs])
      .range([0, maxBarWidth]);

    // Organize all shap values
    const allShaps: SHAPRow[] = [];

    for (let i = 0; i < this.data.featureNames.length; i++) {
      const curFeatureType = this.data.featureTypes[i];
      const curFeatureName = this.data.featureNames[i];

      // Get the display name
      let displayName = '';
      let fullName = '';

      if (curFeatureType === 'cont') {
        displayName = this.contFeatures.get(curFeatureName)!.displayName;
        fullName = displayName;
      } else {
        const curName = curFeatureName.replace(/(.+)-(.+)/, '$1');
        const curLevel = curFeatureName.replace(/(.+)-(.+)/, '$2');
        const catInfo = this.catFeatures.get(curName)!;
        const dummy = this.curX[i] === 1 ? 'T' : 'F';
        const dummyFull = this.curX[i] === 1 ? 'True' : 'False';
        displayName = `${catInfo.displayName} (${catInfo.levelInfo[curLevel][0]}=${dummy})`;
        fullName = `${catInfo.displayName} (${catInfo.levelInfo[curLevel][0]}=${dummyFull})`;
      }

      // Truncate displayName until it fits the limit
      let nameWidth = getLatoTextWidth(displayName, 15);

      while (nameWidth > maxTextWidth) {
        displayName = displayName.replace('...', '');
        displayName = displayName
          .slice(0, displayName.length - 1)
          .concat('...');
        nameWidth = getLatoTextWidth(displayName, 15);
      }

      allShaps.push({
        index: i,
        shap: this.curShapValues[i],
        name: displayName,
        fullName: fullName
      });
    }

    // Sort all shaps based on their absolute shap values
    allShaps.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
    const metricRows = metricRowsForScale;
    const topRows = [...metricRows, ...allShaps].slice(0, K);
    const topRowIndexSet = new Set(topRows.map((r) => r.index));

    const rowContent = content
      .append('g')
      .attr('class', 'row-content')
      .attr('transform', 'translate(0, 20)');

    // Draw the background grid
    rowContent
      .append('rect')
      .attr('class', 'grid-rect')
      .attr('x', maxTextWidth + GAP + maxBarWidth / 2)
      .attr('y', -BAR_HEIGHT / 2)
      .attr('width', 0.2)
      .attr('height', 10 * ROW_HEIGHT + 5);

    /**
     * Helper function to add a SHAP row
     * @param shap SHAP value
     * @param y Y of this row
     * @param opacity The initial opacity value
     */
    const addShapRow = (shap: SHAPRow, y: number, opacity: number) => {
      const row = rowContent
        .append('g')
        .attr('class', `row row-${shap.index}`)
        .attr('transform', `translate(0, ${y})`)
        .style('opacity', opacity);

      // Add background grid
      row
        .append('line')
        .attr('class', 'grid-line')
        .attr('x1', maxTextWidth + GAP / 2)
        .attr('y1', 0)
        .attr('x2', maxTextWidth + GAP + maxBarWidth + GAP / 2)
        .attr('y2', 0);

      row
        .append('text')
        .attr('class', 'feature-name')
        .attr('x', maxTextWidth)
        .text(shap.name)
        .append('title')
        .text(shap.fullName);

      // Add the rectangle
      const rect = row.append('rect').attr('class', 'shap-bar');
      const curRectWidth = this.shapScale(Math.abs(shap.shap));
      if (shap.shap < 0) {
        rect
          .classed('negative', true)
          .attr('x', maxTextWidth + GAP + maxBarWidth / 2 - curRectWidth)
          .attr('y', -BAR_HEIGHT / 2)
          .attr('width', curRectWidth)
          .attr('height', BAR_HEIGHT);
      } else {
        rect
          .attr('x', maxTextWidth + GAP + maxBarWidth / 2)
          .attr('y', -BAR_HEIGHT / 2)
          .attr('width', curRectWidth)
          .attr('height', BAR_HEIGHT);
      }

      // Add the shap number
      row
        .append('text')
        .attr('class', 'shap-number')
        .classed('negative', shap.shap < 0)
        .text(FORMAT_2(shap.shap))
        .attr(
          'x',
          shap.shap < 0
            ? maxTextWidth + GAP + maxBarWidth / 2 + 5
            : maxTextWidth + GAP + maxBarWidth / 2 - 5
        );
    };

    // Add the top K in a list
    for (let i = 0; i < topRows.length; i++) {
      const shap = topRows[i];
      addShapRow(shap, i * ROW_HEIGHT, 1);
    }

    // Draw the rest shap values off the screen
    for (let i = 0; i < allShaps.length; i++) {
      const shap = allShaps[i];
      if (!topRowIndexSet.has(shap.index)) {
        addShapRow(shap, this.shapSVGSize.height + 5, 0);
      }
    }

    // Draw the axis
    const axisGroup = content
      .append('g')
      .attr('class', 'axis-group')
      .attr(
        'transform',
        `translate(${maxTextWidth + GAP}, ${this.shapSVGSize.height - 20})`
      );
    const axis = d3.axisBottom(shapValueScale).tickValues([-maxAbs, 0, maxAbs]);
    axisGroup.call(axis);

    this.shapPlotInitialized = true;
  };

  updateShapPlot = () => {
    if (this.shapSVG === null) throw Error('shapSVG is null.');
    if (this.curShapValues === null) throw Error('curShapValues is null.');
    if (this.data === null) throw Error('data is null.');
    if (this.contFeatures === null) throw Error('contFeatures is null.');
    if (this.catFeatures === null) throw Error('catFeatures is null.');

    const curX = this.getCurX();

    // Create scales (include runtime metric rows so bars never overflow)
    const metricRowsForScale = this.getRuntimeMetricRows();
    const absValues = [
      ...this.curShapValues.map(x => Math.abs(x)),
      ...metricRowsForScale.map((r) => Math.abs(r.shap))
    ];
    const maxAbs = d3.max(absValues)! * SHAP_DOMAIN_PAD;
    this.shapScale = d3
      .scaleLinear()
      .domain([0, maxAbs])
      .range([0, this.maxBarWidth / 2]);

    const shapValueScale = d3
      .scaleLinear()
      .domain([-maxAbs, maxAbs])
      .range([0, this.maxBarWidth]);

    // Organize all shap values
    const allShaps: SHAPRow[] = [];

    for (let i = 0; i < this.data.featureNames.length; i++) {
      const curFeatureType = this.data.featureTypes[i];
      const curFeatureName = this.data.featureNames[i];

      // Get the display name
      let displayName = '';
      let fullName = '';

      if (curFeatureType === 'cont') {
        displayName = this.contFeatures.get(curFeatureName)!.displayName;
        fullName = displayName;
      } else {
        const curName = curFeatureName.replace(/(.+)-(.+)/, '$1');
        const curLevel = curFeatureName.replace(/(.+)-(.+)/, '$2');
        const catInfo = this.catFeatures.get(curName)!;
        const dummy = curX[i] === 1 ? 'T' : 'F';
        const dummyFull = curX[i] === 1 ? 'True' : 'False';
        displayName = `${catInfo.displayName} (${catInfo.levelInfo[curLevel][0]}=${dummy})`;
        fullName = `${catInfo.displayName} (${catInfo.levelInfo[curLevel][0]}=${dummyFull})`;
      }

      // Truncate displayName until it fits the limit
      let nameWidth = getLatoTextWidth(displayName, 15);

      while (nameWidth > this.maxTextWidth) {
        displayName = displayName.replace('...', '');
        displayName = displayName
          .slice(0, displayName.length - 1)
          .concat('...');
        nameWidth = getLatoTextWidth(displayName, 15);
      }

      allShaps.push({
        index: i,
        shap: this.curShapValues[i],
        name: displayName,
        fullName: fullName
      });
    }

    // Sort all shaps based on their absolute shap values
    allShaps.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
    const metricRows = metricRowsForScale;
    const topRows = [...metricRows, ...allShaps].slice(0, K);
    const topRowIndexSet = new Set(topRows.map((r) => r.index));

    const content = this.shapSVG.select('g.content');
    const rowContent = this.shapSVG.select('g.row-content');

    const trans = d3.transition('update').duration(300).ease(d3.easeCubicInOut);

    const updateShapRow = (shap: SHAPRow, y: number, opacity: number) => {
      const row = rowContent.select(`g.row-${shap.index}`);

      row
        .transition(trans)
        .attr('transform', `translate(0, ${y})`)
        .style('opacity', opacity);

      // Update the feature name
      row
        .select('text.feature-name')
        .attr('x', this.maxTextWidth)
        .text(shap.name)
        .select('title')
        .text(shap.fullName);

      // Update the rectangle
      const rect = row.select('rect.shap-bar');
      const curRectWidth = this.shapScale(Math.abs(shap.shap));
      if (shap.shap < 0) {
        rect
          .classed('negative', true)
          .transition(trans)
          .attr(
            'x',
            this.maxTextWidth + GAP + this.maxBarWidth / 2 - curRectWidth
          )
          .attr('y', -BAR_HEIGHT / 2)
          .attr('width', curRectWidth);
      } else {
        rect
          .classed('negative', false)
          .transition(trans)
          .attr('x', this.maxTextWidth + GAP + this.maxBarWidth / 2)
          .attr('y', -BAR_HEIGHT / 2)
          .attr('width', curRectWidth);
      }

      // Update the shap number
      row
        .select('text.shap-number')
        .classed('negative', shap.shap < 0)
        .text(FORMAT_2(shap.shap))
        .transition(trans)
        .attr(
          'x',
          shap.shap < 0
            ? this.maxTextWidth + GAP + this.maxBarWidth / 2 + 5
            : this.maxTextWidth + GAP + this.maxBarWidth / 2 - 5
        );
    };

    // Update the top 10 rows first (first 3 reserved for runtime metrics)
    for (let i = 0; i < topRows.length; i++) {
      const shap = topRows[i];
      updateShapRow(shap, i * ROW_HEIGHT, 1);
    }

    // Draw the rest shap values off the screen
    for (let i = 0; i < allShaps.length; i++) {
      const shap = allShaps[i];
      if (!topRowIndexSet.has(shap.index)) {
        updateShapRow(shap, this.shapSVGSize.height + 5, 0);
      }
    }

    // Update the axis
    const axisGroup = content.select<SVGGElement>('g.axis-group');
    const axis = d3.axisBottom(shapValueScale).tickValues([-maxAbs, 0, maxAbs]);
    axisGroup.call(axis);
  };

  /**
   * Load the lending club dataset.
   */
  initData = async () => {
    const dataFile =
      this.mode === 'v5_global'
        ? 'data/sorting-v5.json'
        : 'data/sorting-f1.json';
    this.sortingConfig = (await d3.json(
      `${import.meta.env.BASE_URL}${dataFile}`
    )) as SortingF1Config | SortingV5Config;

    const featureNames =
      this.mode === 'v5_global'
        ? (this.sortingConfig as SortingV5Config).featureOrder
        : (this.sortingConfig as SortingF1Config).featureNames;

    const featureInfo: Record<string, [string, string]> = {};
    for (const name of featureNames) {
      const entry = this.sortingConfig.featureInfo[name];
      featureInfo[name] = [entry.displayName, entry.description];
    }

    this.data = {
      xTrain: [],
      yTrain: [],
      xTest: [],
      yTest: [],
      featureNames,
      featureTypes: featureNames.map(() => 'cont'),
      featureInfo,
      featureLevelInfo: {},
      featureRequiresLog: [],
      featureRequireInt: []
    } as TabularData;

    this.contFeatures = new Map();
    for (const name of featureNames) {
      const entry = this.sortingConfig.featureInfo[name];
      this.contFeatures.set(name, {
        name,
        displayName: entry.displayName,
        desc: entry.description,
        value: 0,
        requiresInt: !!entry.requiresInt,
        requiresLog: false
      });
    }
    this.catFeatures = new Map();

    if (this.mode === 'v5_global') {
      const v5 = this.sortingConfig as SortingV5Config;
      this.selectedChannel = 'Global';
      this.curClassLabels = v5.classLabels.slice();
      this.backgroundData = v5.backgroundData || [];
      this.currentRuntimeMetrics = v5.runtimeMetrics || null;
      const modelUrl = `${import.meta.env.BASE_URL}${v5.model}`;
      this.modelReady = false;
      this.pendingInference = false;
      this.loadRequestId += 1;
      const loadMsg: TabularWorkerMessage = {
        command: 'startLoadModel',
        payload: { url: modelUrl, requestId: this.loadRequestId }
      };
      this.updateModelLoader(true, true);
      this.tabularWorker.postMessage(loadMsg);
      this.loadRandomSample();
      this.getNewExplanation();
      return;
    }

    this.selectedChannel = (this.sortingConfig as SortingF1Config).defaultChannel || 'Speed';
    this.setChannel(this.selectedChannel);
  };

  /**
   * Load a random sample from the test dataset.
   */
  loadRandomSample = () => {
    if (this.data === null || this.sortingConfig === null) {
      throw Error('this.data is null');
    }

    const examples =
      this.mode === 'v5_global'
        ? ((this.sortingConfig as SortingV5Config).rows as Array<RowExample>)
        : ((this.sortingConfig as SortingF1Config).channels[this.selectedChannel]
            .examples as Array<RowExample>);
    const randomIndex = d3.randomInt(examples.length)();
    const ex = examples[randomIndex];
    this.currentExample = ex;
    this.curX = ex.x.slice();
    this.featureValues = ex.x.slice();
    this.curY = 0;
    this.curIndex = randomIndex;
    this.syncContFeatureValues();

    this.tabularUpdated();
  };

  /**
   * Get the current x values from the user inputs
   */
  getCurX = () => {
    if (this.featureValues.length === 0) {
      throw Error('Feature vector is not initialized');
    }
    return this.featureValues.slice();
  };

  /**
   * Event handler for the sample button clicking.
   */
  sampleClicked = () => {
    this.loadRandomSample();

    // Predict this example
    this.getNewExplanation();
  };

  getNewExplanation = () => {
    if (this.curX === null) {
      throw new Error('curX is null');
    }
    if (!this.modelReady) {
      this.pendingInference = true;
      return;
    }

    this.updateModelLoader(true, false);
    this.updateExplainLoader(true);

    const predictMessage: TabularWorkerMessage = {
      command: 'startPredict',
      payload: {
        x: [this.curX]
      }
    };
    this.tabularWorker.postMessage(predictMessage);
    // Explain the prediction
    const explainMessage: TabularWorkerMessage = {
      command: 'startExplain',
      payload: {
        x: this.curX,
        backgroundData: this.backgroundData
      }
    };
    this.tabularWorker.postMessage(explainMessage);
  };

  /**
   * Helper function to update the view with the new prediction result
   */
  updatePred = () => {
    if (this.curPred === null) {
      throw Error('curPred is null');
    }

    // Update the bar
    const content = this.predBarSVG.select('g.content');
    content
      .select('rect.top-rect')
      .classed('approval', this.curPred >= 0.5)
      .attr('width', this.predBarScale(this.curPred));

    this.tabularUpdated();
  };

  setChannel = (channel: string) => {
    if (this.sortingConfig === null) return;
    if (this.mode === 'v5_global') return;
    const f1 = this.sortingConfig as SortingF1Config;
    if (!f1.channels[channel]) return;
    this.selectedChannel = channel;
    this.curClassLabels = (f1.channels[channel].classes ||
      []) as string[];
    this.currentRuntimeMetrics =
      f1.channels[channel].runtimeMetrics || null;

    const modelRel = f1.channels[channel].model as string;
    const modelUrl = `${import.meta.env.BASE_URL}${modelRel}`;
    this.modelReady = false;
    this.pendingInference = false;
    this.loadRequestId += 1;
    const loadMsg: TabularWorkerMessage = {
      command: 'startLoadModel',
      payload: { url: modelUrl, requestId: this.loadRequestId }
    };
    this.updateModelLoader(true, true);
    this.tabularWorker.postMessage(loadMsg);

    this.backgroundData = f1.channels[channel].backgroundData || [];
    this.loadRandomSample();
    this.getNewExplanation();
  };

  setFeatureValue = (idx: number, value: number) => {
    if (!Number.isFinite(value)) return;
    if (idx < 0 || idx >= this.featureValues.length) return;
    this.featureValues[idx] = value;
    this.curX = this.featureValues.slice();
    this.syncContFeatureValues();
    this.getNewExplanation();
  };

  syncContFeatureValues = () => {
    if (!this.data || !this.contFeatures) return;
    for (let i = 0; i < this.data.featureNames.length; i++) {
      const name = this.data.featureNames[i];
      const cf = this.contFeatures.get(name);
      if (cf) {
        cf.value = this.featureValues[i] ?? 0;
      }
    }
  };
}
