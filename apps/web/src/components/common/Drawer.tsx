// 通用右侧抽屉（§8.7 设计系统 / §7.3 行详情）
//
// 资源详情统一用右侧抽屉承载，避免详情堆在主页面。
// 关闭：点击遮罩 / 关闭按钮 / Esc。

import { X } from "lucide-react";
import { useEffect, type ReactNode } from "react";

export function Drawer({
  open,
  onClose,
  title,
  subtitle,
  children,
  footer,
  className,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  subtitle?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  className?: string;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="infra-drawer-overlay" onClick={onClose} role="presentation">
      <aside
        className={["infra-drawer", className].filter(Boolean).join(" ")}
        role="dialog"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="infra-drawer-header">
          <div>
            <strong>{title}</strong>
            {subtitle && <span className="infra-drawer-subtitle">{subtitle}</span>}
          </div>
          <button className="infra-drawer-close" type="button" aria-label="关闭" onClick={onClose}>
            <X size={16} />
          </button>
        </header>
        <div className="infra-drawer-body">{children}</div>
        {footer && <footer className="infra-drawer-footer">{footer}</footer>}
      </aside>
    </div>
  );
}

/** 抽屉内常用的键值明细行。 */
export function DrawerField({ label, value }: { label: ReactNode; value: ReactNode }) {
  return (
    <div className="infra-drawer-field">
      <span className="infra-drawer-field-label">{label}</span>
      <span className="infra-drawer-field-value">{value}</span>
    </div>
  );
}
