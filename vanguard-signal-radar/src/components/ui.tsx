"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

/** Fit score rendered as a figure plus a hairline meter scaled over the 86–98 band. */
export function FitMeter({ score, className, wide }: { score: number; className?: string; wide?: boolean }) {
  const pct = Math.max(8, Math.min(100, ((score - 84) / 14) * 100));
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <span className="font-mono text-[12px] tabular-nums text-ink">{score}</span>
      <span className={cn("relative h-[3px] overflow-hidden rounded-full bg-line-strong", wide ? "w-24" : "w-10")}>
        <span
          className={cn("animate-grow-x absolute inset-y-0 left-0 rounded-full", score >= 95 ? "bg-brass" : "bg-muted")}
          style={{ width: `${pct}%` }}
        />
      </span>
    </span>
  );
}

export function SectionLabel({ children, className }: { children: React.ReactNode; className?: string }) {
  return <h3 className={cn("text-[11px] font-medium uppercase tracking-[0.14em] text-faint", className)}>{children}</h3>;
}

/** Animates a number from its previous value to `value`. */
export function CountUp({ value, format, duration = 700 }: { value: number; format: (n: number) => string; duration?: number }) {
  const [display, setDisplay] = useState(value);
  const from = useRef(value);

  useEffect(() => {
    const start = performance.now();
    const origin = from.current;
    let frame = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(origin + (value - origin) * eased);
      if (t < 1) frame = requestAnimationFrame(tick);
      else from.current = value;
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value, duration]);

  return <>{format(display)}</>;
}

/** Parses "€145,000/yr" → 145000. */
export function parseEur(value: string): number {
  const m = value.replace(/[^\d.,kKmM]/g, "");
  const digits = m.replace(/[kKmM]$/, "");
  // "45.000" / "1.250.000" use dots as thousands separators (EU style).
  const normalised = /^\d{1,3}(\.\d{3})+$/.test(digits) ? digits.replace(/\./g, "") : digits.replace(/,/g, "");
  let n = Number(normalised);
  if (/k$/i.test(m)) n *= 1_000;
  if (/m$/i.test(m)) n *= 1_000_000;
  return Number.isFinite(n) ? n : 0;
}

export function formatEur(n: number): string {
  if (n >= 1_000_000) return `€${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `€${Math.round(n / 1_000)}k`;
  return `€${Math.round(n)}`;
}
