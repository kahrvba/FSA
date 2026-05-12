import d3 from '../../utils/d3-import';
import { config } from '../../config/config';
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
import { KernelSHAP } from 'webshap';
import { round, timeit, downloadJSON } from '../../utils/utils';
import { getLatoTextWidth } from '../../utils/text-width';

import * as ort from 'onnxruntime-web/wasm';
import ortWasm from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import ortWasmJsep from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';
import ortWasmMjs from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import ortWasmJsepMjs from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url';

const DEBUG = config.debug;

let loadModel: Promise<ort.InferenceSession>;
let activeLoadRequestId = 0;

/**
 * Handle message events from the main thread
 * @param e Message event
 */
self.onmessage = (e: MessageEvent<TabularWorkerMessage>) => {
  switch (e.data.command) {
    case 'startLoadModel': {
      const modelUrl = e.data.payload.url;
      const requestId = e.data.payload.requestId ?? 0;
      activeLoadRequestId = requestId;
      loadModel = startLoadModel(modelUrl, requestId);
      break;
    }

    case 'startPredict': {
      const x = e.data.payload.x;
      predict(x, true);
      break;
    }

    case 'startExplain': {
      const x = e.data.payload.x;
      const backgroundData = e.data.payload.backgroundData;
      explain(x, backgroundData);
      break;
    }

    default: {
      console.error('Worker: unknown message', e.data.command);
      break;
    }
  }
};

const startLoadModel = async (url: string, requestId: number) => {
  try {
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = false;
    ort.env.wasm.wasmPaths = {
      wasm: ortWasm,
      mjs: ortWasmMjs,
      'ort-wasm-simd-threaded.wasm': ortWasm,
      'ort-wasm-simd-threaded.jsep.wasm': ortWasmJsep,
      'ort-wasm-simd-threaded.mjs': ortWasmMjs,
      'ort-wasm-simd-threaded.jsep.mjs': ortWasmJsepMjs
    };

    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    };
    const session = await ort.InferenceSession.create(url, options);

    const message: TabularWorkerMessage = {
      command: 'finishLoadModel',
      payload: { requestId }
    };
    postMessage(message);
    return session;
  } catch (err) {
    const errorText = err instanceof Error ? err.message : String(err);
    const message: TabularWorkerMessage = {
      command: 'failLoadModel',
      payload: {
        error: errorText,
        code: 'LOAD_MODEL_FAILED',
        requestId
      }
    };
    postMessage(message);
    throw err;
  }
};

/**
 * Run XGBoost on the given input data x
 * @param x Input data instances (n, k)
 * @param notifyMainThread True if it needs to send message to main thread
 * @returns Predicted positive label probabilities (n)
 */
const predict = async (x: number[][], notifyMainThread: boolean) => {
  const model = await loadModel;
  const posProbs: number[][] = [];
  const classProbs: number[][] = [];
  const predIndexes: number[] = [];

  try {
    // First need to flatten the x array
    const xFlat = Float32Array.from(x.flat());

    // Prepare feeds, use model input names as keys.
    const xTensor = new ort.Tensor('float32', xFlat, [x.length, x[0].length]);
    const inputName = model.inputNames?.[0] ?? 'float_input';
    const feeds = { [inputName]: xTensor };

    // Feed inputs and run
    const results = await model.run(feeds);
    const outputNames = Object.keys(results);
    const probabilitiesRaw =
      (results.probabilities as unknown) ??
      (results.output_probability as unknown) ??
      Object.values(results)[1] ??
      Object.values(results)[0];
    const selectedOutput =
      results.probabilities !== undefined
        ? 'probabilities'
        : results.output_probability !== undefined
          ? 'output_probability'
          : outputNames[1] || outputNames[0] || 'unknown';

    const rows: number[][] = [];

    // Case 1: ONNX Tensor-style output
    if (probabilitiesRaw === undefined || probabilitiesRaw === null) {
      throw new Error('NO_OUTPUT: probability output is empty');
    } else if (
      probabilitiesRaw &&
      typeof probabilitiesRaw === 'object' &&
      'data' in (probabilitiesRaw as Record<string, unknown>)
    ) {
      const tensorLike = probabilitiesRaw as {
        data: Float32Array | Float64Array | number[];
        dims?: number[];
      };
      const flat = Array.from(tensorLike.data as ArrayLike<number>);
      const classesCount =
        tensorLike.dims && tensorLike.dims.length >= 2 && tensorLike.dims[1]
          ? tensorLike.dims[1]
          : Math.max(1, Math.floor(flat.length / x.length));
      for (let r = 0; r < x.length; r++) {
        const start = r * classesCount;
        rows.push(flat.slice(start, start + classesCount).map((v) => Number(v)));
      }
    }
    // Case 2: ZipMap-style output: array of objects/maps
    else if (Array.isArray(probabilitiesRaw)) {
      for (const item of probabilitiesRaw as Array<unknown>) {
        // ZipMap can be returned as Map<class, prob>
        if (item instanceof Map) {
          const vals = Array.from(item.values())
            .map((v) => Number(v))
            .filter((v) => Number.isFinite(v));
          rows.push(vals);
          continue;
        }
        // Or plain object {class: prob}
        if (item && typeof item === 'object') {
          const vals = Object.values(item as Record<string, unknown>)
            .map((v) => Number(v))
            .filter((v) => Number.isFinite(v));
          rows.push(vals);
          continue;
        }
        rows.push([]);
      }
    } else {
      throw new Error('UNSUPPORTED_OUTPUT_FORMAT: ONNX probability output');
    }

    if (rows.length !== x.length) {
      throw new Error(
        `ROW_DIM_MISMATCH: parsed_rows=${rows.length}, expected_rows=${x.length}`
      );
    }

    for (let row = 0; row < rows.length; row++) {
      const rowProbs = rows[row];
      if (!rowProbs || rowProbs.length === 0) {
        throw new Error('EMPTY_PROB_ROW: parsed row has zero classes');
      }
      let maxProb = Number.NEGATIVE_INFINITY;
      let maxIdx = 0;
      for (let j = 0; j < rowProbs.length; j++) {
        const p = Number(rowProbs[j] ?? 0);
        if (j === 0 || p > maxProb) {
          maxProb = p;
          maxIdx = j;
        }
      }
      posProbs.push([Number.isFinite(maxProb) ? maxProb : 0]);
      classProbs.push(rowProbs);
      predIndexes.push(maxIdx);
    }

    if (notifyMainThread) {
      const message: TabularWorkerMessage = {
        command: 'finishPredict',
        payload: {
          posProbs: posProbs,
          classProbs,
          predIndexes,
          debug: {
            outputNames,
            selectedOutput,
            classCount: classProbs[0]?.length || 0,
            firstRowProbs: classProbs[0] || []
          }
        }
      };
      postMessage(message);
    }
  } catch (e) {
    console.error(`Failed model prediction: ${e}.`);
    if (notifyMainThread) {
      const message: TabularWorkerMessage = {
        command: 'failPredict',
        payload: {
          error: e instanceof Error ? e.message : String(e),
          code:
            e instanceof Error
              ? e.message.split(':')[0]
              : 'PREDICT_FAILED'
        }
      };
      postMessage(message);
    }
  }

  return posProbs;
};

/**
 * Run WebSHAP to explain the given input data x
 * @param x Input data instance
 */
const explain = async (x: number[], backgroundData: number[][]) => {
  try {
    const explainer = new KernelSHAP(
      (x: number[][]) => predict(x, false),
      backgroundData,
      0.2022
    );

    timeit('Explain tabular', DEBUG);
    const shapValues = await explainer.explainOneInstance(x, 512);
    timeit('Explain tabular', DEBUG);

    const message: TabularWorkerMessage = {
      command: 'finishExplain',
      payload: {
        shapValues
      }
    };
    postMessage(message);
    return shapValues;
  } catch (e) {
    const message: TabularWorkerMessage = {
      command: 'failExplain',
      payload: {
        error: e instanceof Error ? e.message : String(e),
        code: 'EXPLAIN_FAILED'
      }
    };
    postMessage(message);
    return [];
  }
};
