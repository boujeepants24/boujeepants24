"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, Crosshair, Database, Loader2, Radar, Rocket, Zap } from "lucide-react";
import { DEFAULT_PRESET, PRESETS, getPreset } from "@/lib/presets";
import { buildCampaignCsv, downloadCsv, type LeadDraft } from "@/lib/csv";
import type { Lead, PresetId, SweepEvent, SweepMode } from "@/lib/types";
import { cn } from "@/lib/utils";
import { InspectorDrawer } from "./inspector-drawer";
import { ResultsTable } from "./results-table";
import { TerminalLog, type LogLine } from "./terminal-log";
import { formatEur, parseEur } from "./ui";

export function RadarConsole() {
  const [presetId, setPresetId] = useState<PresetId>(DEFAULT_PRESET.id);
  const [query, setQuery] = useState(DEFAULT_PRESET.query);
  const [offer, setOffer] = useState(DEFAULT_PRESET.offer);
  const [mode, setMode] = useState<SweepMode>("live");

  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [terminalOpen, setTerminalOpen] = useState(true);
  const [leads, setLeads] = useState<Lead[]>([]);
  const [source, setSource] = useState<"live" | "cache" | null>(null);
  const [fallbackReason, setFallbackReason] = useState<string | undefined>();
  const [drafts, setDrafts] = useState<Record<string, LeadDraft>>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const logId = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 2400);
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const pushLog = useCallback((tag: LogLine["tag"], message: string, ts = Date.now()) => {
    setLogs((prev) => [...prev, { id: logId.current++, tag, message, ts }]);
  }, []);

  const applyPreset = (id: PresetId) => {
    setPresetId(id);
    const p = getPreset(id);
    if (id !== "custom") {
      setQuery(p.query);
      setOffer(p.offer);
    }
  };

  const handleEvent = useCallback(
    (event: SweepEvent) => {
      if (event.type === "log") pushLog(event.tag, event.message, event.ts);
      else if (event.type === "error") pushLog("ERROR", event.message, event.ts);
      else {
        setLeads(event.leads);
        setSource(event.source);
        setFallbackReason(event.fallbackReason);
        setDrafts(
          Object.fromEntries(
            event.leads.map((l) => [
              l.id,
              { subject: l.sniperSubject, body: l.sniperEmail, firstName: "", lastName: "", pushed: false },
            ]),
          ),
        );
      }
    },
    [pushLog],
  );

  const runSweep = async () => {
    if (running || !query.trim() || !offer.trim()) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setRunning(true);
    setTerminalOpen(true);
    setLogs([]);
    setSelectedId(null);

    try {
      const res = await fetch("/api/sweep", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, offer, mode, presetId }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        const err = (await res.json().catch(() => null)) as { error?: string } | null;
        throw new Error(err?.error ?? `Sweep failed (HTTP ${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (line.trim()) handleEvent(JSON.parse(line) as SweepEvent);
        }
      }
      if (buffer.trim()) handleEvent(JSON.parse(buffer) as SweepEvent);
    } catch (err) {
      if (controller.signal.aborted) return;
      pushLog("ERROR", err instanceof Error ? err.message : "Sweep failed");
    } finally {
      if (abortRef.current === controller) setRunning(false);
    }
  };

  const updateDraft = useCallback((id: string, patch: Partial<LeadDraft>) => {
    setDrafts((prev) => ({ ...prev, [id]: { ...prev[id], ...patch } }));
  }, []);

  const exportCsv = () => {
    if (leads.length === 0) return;
    const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
    downloadCsv(`vanguard-campaign-${stamp}.csv`, buildCampaignCsv(leads, drafts));
    showToast(`Exported ${leads.length} leads to CSV`);
  };

  const stats = useMemo(() => {
    const pipeline = leads.reduce((sum, l) => sum + parseEur(l.estimatedAnnualValue), 0);
    const avgFit = leads.length ? leads.reduce((s, l) => s + l.fitScore, 0) / leads.length : 0;
    const queued = Object.values(drafts).filter((d) => d.pushed).length;
    return { pipeline, avgFit, queued };
  }, [leads, drafts]);

  const selectedIndex = leads.findIndex((l) => l.id === selectedId);
  const selected = selectedIndex >= 0 ? leads[selectedIndex] : null;
  const closeInspector = useCallback(() => setSelectedId(null), []);
  const prevLead = selectedIndex > 0 ? () => setSelectedId(leads[selectedIndex - 1].id) : undefined;
  const nextLead =
    selectedIndex >= 0 && selectedIndex < leads.length - 1 ? () => setSelectedId(leads[selectedIndex + 1].id) : undefined;

  return (
    <div className="bg-grid min-h-screen">
      {/* Top header */}
      <header className="sticky top-0 z-40 border-b border-line bg-obsidian/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="relative grid size-8 place-items-center overflow-hidden rounded-md border border-emerald-500/30 bg-emerald-500/10">
              <Radar className="size-4 text-emerald-400" />
              {running && (
                <span className="animate-radar absolute inset-0 bg-[conic-gradient(from_0deg,transparent_0deg,rgb(52_211_153/0.35)_60deg,transparent_70deg)]" />
              )}
            </div>
            <div className="font-mono text-[13px] font-semibold tracking-[0.14em] text-ink">
              VANGUARD <span className="text-faint">{"//"}</span> SIGNAL RADAR{" "}
              <span className="text-emerald-400">v2.4</span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2">
              <span className="hidden font-mono text-[10px] uppercase tracking-[0.18em] text-faint md:inline">Preset</span>
              <select
                value={presetId}
                onChange={(e) => applyPreset(e.target.value as PresetId)}
                className="max-w-[78vw] rounded-md border border-line bg-panel px-2.5 py-1.5 text-xs text-ink focus:border-emerald-500/50 focus:outline-none sm:max-w-none"
              >
                {PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>
            <span className="hidden items-center gap-1.5 rounded-md border border-line px-2 py-1.5 font-mono text-[11px] text-muted sm:inline-flex">
              <Rocket className="size-3 text-emerald-400" />
              Campaign queue <span className="tabular-nums text-ink">{stats.queued}</span>
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1440px] space-y-4 px-4 py-5 sm:px-6">
        {/* Control panel */}
        <section className="rounded-lg border border-line bg-panel p-4">
          <div className="grid gap-4 lg:grid-cols-[1fr_1.4fr]">
            <Field label="Target Buyer Signal / Niche Search" icon={<Crosshair className="size-3.5 text-cyan-400" />}>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && runSweep()}
                placeholder="industrial IoT hardware manufacturers Germany UK"
                maxLength={200}
                className="w-full rounded-md border border-line bg-obsidian px-3 py-2.5 font-mono text-[13px] text-ink placeholder:text-faint focus:border-emerald-500/50 focus:outline-none"
              />
            </Field>
            <Field label="Client Offer & Capability" icon={<Zap className="size-3.5 text-emerald-400" />}>
              <textarea
                value={offer}
                onChange={(e) => {
                  setOffer(e.target.value);
                  if (presetId !== "custom") setPresetId("custom");
                }}
                rows={2}
                maxLength={800}
                placeholder="What you sell, with the concrete cost / speed / quality edge"
                className="w-full resize-none rounded-md border border-line bg-obsidian px-3 py-2 text-[13px] leading-relaxed text-ink placeholder:text-faint focus:border-emerald-500/50 focus:outline-none"
              />
            </Field>
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
            <div role="radiogroup" aria-label="Sweep mode" className="inline-flex rounded-md border border-line bg-obsidian p-0.5">
              <ModeButton active={mode === "live"} onClick={() => setMode("live")} icon={<Activity className="size-3.5" />}>
                Live Web Scrape + Groq AI
              </ModeButton>
              <ModeButton active={mode === "cache"} onClick={() => setMode("cache")} icon={<Database className="size-3.5" />}>
                Cached Instant Demo
              </ModeButton>
            </div>

            <button
              type="button"
              onClick={runSweep}
              disabled={running || !query.trim() || !offer.trim()}
              className="inline-flex items-center gap-2 rounded-md bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-[#04120c] shadow-[0_0_24px_-6px_rgb(16_185_129/0.7)] transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {running ? <Loader2 className="size-4 animate-spin" /> : <Radar className="size-4" />}
              {running ? "Sweeping..." : "Execute Signal Sweep"}
            </button>
          </div>
        </section>

        <TerminalLog lines={logs} running={running} open={terminalOpen} onToggle={() => setTerminalOpen((o) => !o)} />

        {leads.length > 0 && (
          <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-4">
            <Stat label="Qualified Buyers" value={String(leads.length)} />
            <Stat label="Avg Fit Score" value={stats.avgFit.toFixed(1)} />
            <Stat label="Pipeline Value" value={`${formatEur(stats.pipeline)}/yr`} accent />
            <Stat label="Queued for Outbound" value={`${stats.queued} / ${leads.length}`} />
          </div>
        )}

        {leads.length > 0 ? (
          <ResultsTable
            leads={leads}
            drafts={drafts}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onExport={exportCsv}
            source={source}
            fallbackReason={fallbackReason}
          />
        ) : (
          <EmptyState running={running} />
        )}
      </main>

      {selected && drafts[selected.id] && (
        <InspectorDrawer
          key={selected.id}
          lead={selected}
          draft={drafts[selected.id]}
          onChange={(patch) => updateDraft(selected.id, patch)}
          onClose={closeInspector}
          onPrev={prevLead}
          onNext={nextLead}
          onToast={showToast}
        />
      )}

      {toast && (
        <div
          role="status"
          className="animate-fade-in fixed bottom-5 left-1/2 z-[60] -translate-x-1/2 rounded-md border border-emerald-500/30 bg-panel px-4 py-2 font-mono text-xs text-emerald-200 shadow-xl shadow-black/50"
        >
          {toast}
        </div>
      )}
    </div>
  );
}

function Field({ label, icon, children }: { label: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-muted">
        {icon}
        {label}
      </span>
      {children}
    </label>
  );
}

function ModeButton({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-medium transition",
        active ? "bg-panel-2 text-ink shadow-[inset_0_0_0_1px_#2a3042]" : "text-faint hover:text-muted",
        active && "[&>svg]:text-emerald-400",
      )}
    >
      {icon}
      {children}
    </button>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="bg-panel px-4 py-3">
      <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-faint">{label}</div>
      <div className={cn("mt-1 font-mono text-xl tabular-nums", accent ? "text-emerald-300" : "text-ink")}>{value}</div>
    </div>
  );
}

function EmptyState({ running }: { running: boolean }) {
  return (
    <section className="grid place-items-center rounded-lg border border-dashed border-line bg-panel/50 px-6 py-16 text-center">
      <div className="relative grid size-24 place-items-center rounded-full border border-line">
        <div className="absolute inset-3 rounded-full border border-line/70" />
        <div className="absolute inset-7 rounded-full border border-line/50" />
        <div
          className={cn(
            "absolute inset-0 rounded-full bg-[conic-gradient(from_0deg,transparent_0deg,rgb(52_211_153/0.28)_70deg,transparent_80deg)]",
            running ? "animate-radar" : "opacity-40",
          )}
        />
        <span className="size-1.5 rounded-full bg-emerald-400" />
      </div>
      <p className="mt-5 text-sm text-ink">{running ? "Sweeping the market for buying signals..." : "No signals acquired yet"}</p>
      <p className="mt-1 max-w-md text-xs text-faint">
        Pick a preset or describe your niche and offer, then execute a sweep. Live mode scrapes real company sites and drafts
        outreach with Groq; the instant cache renders a pre-qualified dataset in 1.5s.
      </p>
    </section>
  );
}
