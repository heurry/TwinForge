import type { ReactNode } from "react";

export type DisplayTone = "success" | "warning" | "danger";

// 迷你趋势 / 拓扑节点色调（含 info；曾定义在 data/platformSnapshots）。
export type SparkTone = "info" | "success" | "warning" | "danger";

export type CardTone = "success" | "warning" | "danger" | "ai" | "executing";

export type KpiItem = {
  id: string;
  label: string;
  value: string;
  detail: string;
  tone?: "success" | "warning" | "danger" | "ai";
  onClick?: () => void;
  // 样例图 KPI 卡：数值 + 迷你趋势 + 环比。均可选，向后兼容旧调用。
  trend?: number[];
  delta?: string;
  deltaTone?: "up" | "down" | "flat";
  unit?: string;
};

export type KeyValueItem = {
  id: string;
  label: string;
  value: string;
  detail?: string;
  status?: string;
  tone?: CardTone;
};

export type StrategyCardItem = {
  id: string;
  label: string;
  value: string;
  description: string;
  status?: string;
  tone?: CardTone;
};

export type EventListItem = {
  id: string;
  title: string;
  status?: string;
  meta?: string;
  description?: string;
  time?: string;
  tone?: CardTone;
};

export type DataTableCell =
  | ReactNode
  | {
      value: ReactNode;
      strong?: boolean;
    };

export type DataTableRow = {
  id: string;
  cells: DataTableCell[];
};
