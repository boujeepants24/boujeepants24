import type { PresetId } from "./types";

export interface Preset {
  id: PresetId;
  label: string;
  query: string;
  offer: string;
  /** Sentence 2 of the cached sniper email: our capability / speed advantage. */
  pitchLine: string;
  /** Sentence 3 of the cached sniper email: the low-friction ask. */
  ask: string;
}

export const PRESETS: Preset[] = [
  {
    id: "pcb",
    label: "Custom PCB & Motherboard Manufacturing (Export OEM)",
    query: "industrial IoT hardware manufacturers Germany UK",
    offer:
      "High-reliability multilayer PCB assembly & circuit manufacturing with 25% lower BOM cost and 10-day turnaround",
    pitchLine:
      "We build IPC Class 3 multilayer boards and full PCBA at roughly 25% under typical EU BOM cost, with a 10-day turnaround from Gerbers to delivered units.",
    ask: "Worth sending over a sample quote against your current board?",
  },
  {
    id: "cnc",
    label: "Precision CNC & Industrial Components",
    query: "industrial automation equipment manufacturers Germany Netherlands",
    offer:
      "5-axis precision CNC machining of aluminium & stainless enclosures, heatsinks and brackets with ±0.01mm tolerance, ISO 9001, 12-day lead time",
    pitchLine:
      "We run 5-axis CNC on aluminium and stainless housings, heatsinks and brackets at ±0.01mm, ISO 9001 certified, shipping in 12 days.",
    ask: "Worth sending over a spec sheet and a first-article quote?",
  },
  {
    id: "logistics",
    label: "B2B Logistics & Supply Chain Fulfillment",
    query: "medical device and electronics OEM manufacturers Europe export",
    offer:
      "Bonded EU/US cross-border B2B fulfillment for hardware OEMs: customs brokerage, kitting, 48h pan-EU delivery and consolidated freight",
    pitchLine:
      "We handle bonded EU/US warehousing, customs brokerage and kitting for hardware OEMs, with 48-hour pan-EU delivery on consolidated freight.",
    ask: "Worth sending over a lane-by-lane rate card?",
  },
  {
    id: "custom",
    label: "Custom Query",
    query: "",
    offer: "",
    pitchLine: "",
    ask: "Worth sending over a short capability sheet?",
  },
];

export const DEFAULT_PRESET = PRESETS[0];

export function getPreset(id: PresetId): Preset {
  return PRESETS.find((p) => p.id === id) ?? DEFAULT_PRESET;
}
