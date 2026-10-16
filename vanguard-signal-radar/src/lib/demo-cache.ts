import type { Lead, PresetId } from "./types";
import { getPreset } from "./presets";

/**
 * Instant Demo Cache.
 *
 * Eight fictional-but-realistic EU/US hardware OEM buyers used when the user
 * toggles "Cached Instant Demo" or when the live pipeline (DuckDuckGo / Groq)
 * is rate-limited or unavailable. Company names and domains are illustrative.
 */

interface Variant {
  signal: string;
  role: string;
  value: string;
  subject: string;
  fit: number;
}

interface CachedCompany {
  id: string;
  companyName: string;
  website: string;
  hq: string;
  sector: string;
  siteTitle: string;
  siteDescription: string;
  /** Sentence 1 of the sniper email: a concrete detail from their site. */
  hook: string;
  variants: Record<Exclude<PresetId, "custom">, Variant>;
}

const COMPANIES: CachedCompany[] = [
  {
    id: "c-nordlicht",
    companyName: "Nordlicht Sensorik GmbH",
    website: "https://www.nordlicht-sensorik.de",
    hq: "Stuttgart, DE",
    sector: "Industrial IoT edge gateways",
    siteTitle: "Nordlicht Sensorik | Industrial IoT Gateways & Condition Monitoring",
    siteDescription:
      "Rugged edge gateways and vibration sensors for predictive maintenance on factory floors across DACH.",
    hook: "Saw the NX-4 edge gateway is now shipping with dual-band LoRaWAN and a 24V DIN-rail I/O board.",
    variants: {
      pcb: {
        signal:
          "Scaling the NX-4 gateway into volume production with a new 8-layer I/O board — classic trigger to second-source PCBA before the ramp.",
        role: "VP of Hardware Engineering",
        value: "€145,000/yr",
        subject: "nx-4 i/o board supply",
        fit: 97,
      },
      cnc: {
        signal:
          "NX-4 moves to an IP67 die-cast housing for outdoor installs, opening a slot for machined aluminium enclosures and heatsinks.",
        role: "Head of Mechanical Engineering",
        value: "€88,000/yr",
        subject: "nx-4 enclosure machining",
        fit: 93,
      },
      logistics: {
        signal:
          "Expanding NX-4 distribution into the UK and Nordics post-launch, which means new cross-border customs and kitting flows.",
        role: "Supply Chain Director",
        value: "€72,000/yr",
        subject: "nx-4 uk distribution",
        fit: 91,
      },
    },
  },
  {
    id: "c-brightwell",
    companyName: "Brightwell Medical Devices Ltd",
    website: "https://www.brightwellmedical.co.uk",
    hq: "Cambridge, UK",
    sector: "Wearable patient monitoring (Class IIa)",
    siteTitle: "Brightwell Medical | Continuous Patient Monitoring",
    siteDescription:
      "VitalLink wireless patch monitors for continuous ward and remote patient monitoring. CE marked under EU MDR.",
    hook: "Noticed the VitalLink patch monitor just picked up CE marking under MDR and is moving into NHS trust rollouts.",
    variants: {
      pcb: {
        signal:
          "Post-MDR certification ramp of the VitalLink patch needs a high-reliability flex-rigid PCB supplier with full lot traceability.",
        role: "Head of Global Procurement",
        value: "€210,000/yr",
        subject: "vitallink flex-rigid boards",
        fit: 96,
      },
      cnc: {
        signal:
          "VitalLink charging dock and clinical mounts are moving from 3D-printed to production parts ahead of NHS volume orders.",
        role: "Manufacturing Engineering Manager",
        value: "€54,000/yr",
        subject: "vitallink dock parts",
        fit: 88,
      },
      logistics: {
        signal:
          "Post-Brexit EU shipments of a CE-marked device need bonded EU warehousing and an importer-of-record setup.",
        role: "Head of Operations & Logistics",
        value: "€118,000/yr",
        subject: "vitallink eu importer",
        fit: 95,
      },
    },
  },
  {
    id: "c-voltaris",
    companyName: "Voltaris Automotive Controls S.p.A.",
    website: "https://www.voltaris-controls.it",
    hq: "Turin, IT",
    sector: "EV battery management systems",
    siteTitle: "Voltaris | Battery Management & Powertrain Controllers",
    siteDescription:
      "ASIL-C battery management systems and powertrain control units for commercial electric vehicles.",
    hook: "Saw Voltaris is sampling the BMS-800 controller for 800V commercial EV packs.",
    variants: {
      pcb: {
        signal:
          "BMS-800 moving from sampling to SOP needs heavy-copper, high-voltage-clearance boards under IATF 16949 — a supplier qualification window is open now.",
        role: "Supply Chain Director",
        value: "€320,000/yr",
        subject: "bms-800 pcb capacity",
        fit: 98,
      },
      cnc: {
        signal:
          "800V BMS-800 needs machined aluminium cold plates and busbar housings for thermal management at SOP.",
        role: "Head of Purchasing, Mechanical Components",
        value: "€190,000/yr",
        subject: "bms-800 cold plates",
        fit: 95,
      },
      logistics: {
        signal:
          "SOP of BMS-800 for German and French truck OEMs requires JIT delivery and sequenced kitting into customer lines.",
        role: "Logistics & Supply Chain Manager",
        value: "€140,000/yr",
        subject: "bms-800 jit delivery",
        fit: 92,
      },
    },
  },
  {
    id: "c-helix",
    companyName: "Helix Grid Systems AB",
    website: "https://www.helixgrid.se",
    hq: "Gothenburg, SE",
    sector: "Smart metering & grid edge",
    siteTitle: "Helix Grid Systems | Smart Meters for Nordic Utilities",
    siteDescription:
      "HG-Meter smart electricity meters and grid-edge analytics deployed across Nordic and Baltic utilities.",
    hook: "Read that the HG-Meter 3 rollout with Nordic utilities is scaling toward 400k units.",
    variants: {
      pcb: {
        signal:
          "HG-Meter 3 rollout to 400k units means multi-year volume PCBA contracts and a push to cut per-unit BOM cost.",
        role: "Director of Strategic Sourcing",
        value: "€275,000/yr",
        subject: "hg-meter 3 bom cost",
        fit: 95,
      },
      cnc: {
        signal:
          "HG-Meter 3 utility cabinets and DIN mounts are being re-sourced for the 400k-unit rollout.",
        role: "Category Manager, Mechanics",
        value: "€66,000/yr",
        subject: "hg-meter mounting parts",
        fit: 87,
      },
      logistics: {
        signal:
          "Meter deliveries to Baltic utilities need consolidated freight and staged regional depots to hit install schedules.",
        role: "Head of Supply Chain",
        value: "€98,000/yr",
        subject: "baltic meter freight",
        fit: 90,
      },
    },
  },
  {
    id: "c-arcadia",
    companyName: "Arcadia Robotics Inc.",
    website: "https://www.arcadiarobotics.com",
    hq: "Pittsburgh, PA, US",
    sector: "Autonomous mobile robots (AMR)",
    siteTitle: "Arcadia Robotics | Heavy-Payload Autonomous Mobile Robots",
    siteDescription:
      "Atlas-series AMRs moving pallets up to 1,500kg in warehouses and automotive plants.",
    hook: "Saw the Atlas-series AMR now runs a new motor-control board for 1,500kg payloads.",
    variants: {
      pcb: {
        signal:
          "New Atlas motor-control board plus EU expansion — they need a second-source, tariff-friendly PCBA partner outside Asia.",
        role: "VP of Hardware Engineering",
        value: "€180,000/yr",
        subject: "atlas motor board",
        fit: 94,
      },
      cnc: {
        signal:
          "1,500kg payload chassis upgrade drives demand for machined drive-train brackets and steel lift components.",
        role: "Director of Mechanical Engineering",
        value: "€160,000/yr",
        subject: "atlas chassis brackets",
        fit: 96,
      },
      logistics: {
        signal:
          "Opening EU customer deployments from a US factory requires bonded import, spare-parts depots and 48h service logistics.",
        role: "VP of Operations",
        value: "€130,000/yr",
        subject: "atlas eu spares depot",
        fit: 93,
      },
    },
  },
  {
    id: "c-kestrel",
    companyName: "Kestrel Instruments B.V.",
    website: "https://www.kestrel-instruments.nl",
    hq: "Eindhoven, NL",
    sector: "Semiconductor test & measurement",
    siteTitle: "Kestrel Instruments | Thermal Test Systems for Semiconductor Fabs",
    siteDescription:
      "Precision thermal test heads and handler systems for wafer-level and package-level semiconductor test.",
    hook: "Noticed the KX-200 thermal test head launched with a redesigned 12-layer control board.",
    variants: {
      pcb: {
        signal:
          "KX-200's redesigned 12-layer control board with tight-impedance traces is a high-mix, low-volume job most EMS houses deprioritise.",
        role: "Head of Electronics Engineering",
        value: "€95,000/yr",
        subject: "kx-200 control board",
        fit: 93,
      },
      cnc: {
        signal:
          "KX-200 thermal heads use precision-machined copper and stainless parts at micron tolerances — a core CNC buyer.",
        role: "Head of Global Procurement",
        value: "€240,000/yr",
        subject: "kx-200 machined parts",
        fit: 98,
      },
      logistics: {
        signal:
          "Shipping fab tools to Taiwan, Korea and Arizona needs high-value, shock-monitored export freight and customs handling.",
        role: "Global Logistics Manager",
        value: "€115,000/yr",
        subject: "kx-200 export freight",
        fit: 91,
      },
    },
  },
  {
    id: "c-fieldpulse",
    companyName: "FieldPulse AgriSystems SAS",
    website: "https://www.fieldpulse-agri.fr",
    hq: "Lyon, FR",
    sector: "Precision agriculture sensors",
    siteTitle: "FieldPulse | Soil & Crop Sensing for Precision Agriculture",
    siteDescription:
      "Solar-powered soil moisture and nutrient sensor nodes with NB-IoT connectivity for growers and cooperatives.",
    hook: "Saw the FieldPulse soil-sensor node is shifting to solar-powered NB-IoT units for the 2027 season.",
    variants: {
      pcb: {
        signal:
          "Re-spinning the sensor node to NB-IoT + solar before the 2027 season compresses their PCB lead time — speed is the buying lever.",
        role: "Hardware Engineering Lead",
        value: "€62,000/yr",
        subject: "nb-iot node boards",
        fit: 91,
      },
      cnc: {
        signal:
          "New solar node needs weatherproof machined probe housings and mounting stakes for field deployment.",
        role: "Operations Director",
        value: "€38,000/yr",
        subject: "probe housing quote",
        fit: 86,
      },
      logistics: {
        signal:
          "Seasonal pre-planting deliveries to cooperatives across FR/ES/IT need peak-capacity fulfillment and kitting.",
        role: "Supply Chain Manager",
        value: "€58,000/yr",
        subject: "pre-season fulfillment",
        fit: 89,
      },
    },
  },
  {
    id: "c-summit",
    companyName: "Summit Energy Storage Corp.",
    website: "https://www.summitenergystorage.com",
    hq: "Austin, TX, US",
    sector: "Commercial battery storage & inverters",
    siteTitle: "Summit Energy Storage | Commercial & Industrial Battery Systems",
    siteDescription:
      "PowerStack modular battery and inverter systems for commercial and industrial sites in the US and EU.",
    hook: "Saw the PowerStack 250 commercial inverter launched with a modular 6-layer power board.",
    variants: {
      pcb: {
        signal:
          "PowerStack 250's EU launch needs CE-compliant, heavy-copper power boards sourced closer to EU installers.",
        role: "Head of Global Procurement",
        value: "€230,000/yr",
        subject: "powerstack power boards",
        fit: 95,
      },
      cnc: {
        signal:
          "Modular PowerStack cabinets use machined busbars and heatsinks that scale linearly with installed MWh.",
        role: "Director of Supply Chain",
        value: "€175,000/yr",
        subject: "powerstack busbars",
        fit: 94,
      },
      logistics: {
        signal:
          "Shipping lithium battery systems into the EU needs ADR/DG-certified freight and bonded staging near installers.",
        role: "VP of Supply Chain",
        value: "€205,000/yr",
        subject: "powerstack dg freight",
        fit: 97,
      },
    },
  },
];

/** Turns the user's free-text offer into a sentence-2 capability line. */
function offerToPitch(offer: string): string {
  const clean = offer.trim().replace(/\s+/g, " ").replace(/[.\s]+$/, "");
  if (!clean) return "We can quote and ship faster than your current supplier.";
  return `We do ${clean.charAt(0).toLowerCase()}${clean.slice(1)}.`;
}

export function getCachedLeads(presetId: PresetId, offer: string): Lead[] {
  const preset = getPreset(presetId);
  const variantKey = presetId === "custom" ? "pcb" : presetId;
  const pitch = presetId === "custom" || !preset.pitchLine ? offerToPitch(offer) : preset.pitchLine;

  return COMPANIES.map((c) => {
    const v = c.variants[variantKey];
    return {
      id: c.id,
      companyName: c.companyName,
      website: c.website,
      detectedSignal: v.signal,
      targetRole: v.role,
      estimatedAnnualValue: v.value,
      fitScore: v.fit,
      sniperSubject: v.subject,
      sniperEmail: `${c.hook} ${pitch} ${preset.ask}`,
      hq: c.hq,
      sector: c.sector,
      siteTitle: c.siteTitle,
      siteDescription: c.siteDescription,
      source: "cache" as const,
    };
  }).sort((a, b) => b.fitScore - a.fitScore);
}
