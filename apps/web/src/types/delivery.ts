export type DeliveryKind = "training" | "inference";

/**
 * 当前交付对象。URL 是唯一真实来源，页面之间只传递这些稳定 ID，
 * 具体名称和状态仍由各页面从后端读取，避免在前端复制业务数据。
 */
export type DeliveryContext = {
  deliveryKind?: DeliveryKind;
  modelId?: string;
  modelVersionId?: string;
  configItemId?: string;
  configVersion?: string;
  trainingJobId?: string;
  benchmarkRunId?: string;
  deploymentId?: string;
  diagnosisId?: string;
};

export type DeliveryContextPatch = {
  [K in keyof DeliveryContext]?: DeliveryContext[K] | null;
};

const PARAMS: Record<keyof DeliveryContext, string> = {
  deliveryKind: "kind",
  modelId: "model_id",
  modelVersionId: "model_version_id",
  configItemId: "config_item_id",
  configVersion: "config_version",
  trainingJobId: "training_job_id",
  benchmarkRunId: "benchmark_run_id",
  deploymentId: "deployment_id",
  diagnosisId: "diagnosis_id",
};

export function readDeliveryContext(params: URLSearchParams): DeliveryContext {
  const result: DeliveryContext = {};
  (Object.keys(PARAMS) as Array<keyof DeliveryContext>).forEach((key) => {
    const value = params.get(PARAMS[key]);
    if (value) Object.assign(result, { [key]: value });
  });
  if (result.deliveryKind !== "training" && result.deliveryKind !== "inference") {
    delete result.deliveryKind;
  }
  return result;
}

export function writeDeliveryContext(params: URLSearchParams, patch: DeliveryContextPatch): URLSearchParams {
  const next = new URLSearchParams(params);
  (Object.keys(patch) as Array<keyof DeliveryContext>).forEach((key) => {
    const value = patch[key];
    if (value === undefined) return;
    if (value === null || value === "") next.delete(PARAMS[key]);
    else next.set(PARAMS[key], value);
  });
  next.delete("page");
  return next;
}

export function clearDeliveryContext(params: URLSearchParams): URLSearchParams {
  const next = new URLSearchParams(params);
  Object.values(PARAMS).forEach((key) => next.delete(key));
  next.delete("page");
  return next;
}
