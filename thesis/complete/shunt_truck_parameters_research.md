# Shunt-Truck / Terminal-Tractor Parameters to Replace the "Tesla Model S" Set

**Purpose:** Source a real-world physical parameter set for a shunt truck (terminal tractor / yard hostler)
to replace the generic `tesla_model_s_vehicle_params` dict used in the RL tractor-trailer dynamic bicycle model.
**Status of the values being replaced:** the current `m=1500, Iz=3000, Cf=Cr=80000, lf=1.2, lr=1.6, Cd=0.208, A=2.4`
dict has **no source comment anywhere in the repo** — it is a generic Rajamani-textbook bicycle-model example,
not measured Tesla data. Replacing it with sourced/estimated hostler numbers is a strict improvement.

**Recommended target vehicle:** **Kalmar Ottawa T2 4x2 terminal tractor** — its DOT/EPA spec sheets and EV
brochure are genuine primary sources, and its 4x2 layout (one steered front axle, one driven rear axle) maps
cleanly onto the dynamic bicycle model. Primary set = **diesel T2** (most representative standard hostler);
alternative = **T2 EV**.

> Research method: multi-source web search → primary-source fetch → 3-vote adversarial verification
> (22 claims confirmed, 3 refuted). Confidence flags below reflect that verification.

---

## 1. Primary-sourced parameters (HIGH confidence)

| Param | Diesel T2 | T2 EV (distribution) | Source | Notes |
|---|---|---|---|---|
| Bobtail mass `m` | **6,577 kg** (14,500 lb) | 7,983 kg (17,600 lb) | Kalmar 4x2 DOT/EPA sheet; T2 EV brochure | "Approximate weight: 14,500 lbs" verbatim |
| Wheelbase `L=lf+lr` | **2.946 m** (116 in) | 3.20 m (126 in) | DOT sheet + T2 4x2 product page | EV options 126–136 in |
| Overall length | 5.11 m (201 in) | 5.11 m | T2 EV brochure | |
| Width | 2.54 m (100 in) | 2.54 m | T2 EV brochure | |
| Cab height | 3.20 m (126 in) | 3.20 m | T2 EV brochure | cabin, not overall height |
| Max road speed | **20.1 m/s** (45 mph / 72 km/h) | 20.1 m/s | DOT sheets | yard-only variants governed to ~11 m/s (25 mph) |
| Fifth-wheel vertical cap | 80,000 lb (36,287 kg) | 80,000 lb | Holland FW35-TT rating | binds the loaded state |
| Max GCW | 80–81,000 lb (36,741 kg) | 81,000 → up to 195,000 lb (88,450 kg) on heavy EV | DOT sheet; EV brochure | |
| Max steering angle | **0.87 rad** (50°) | 0.87 rad | Terberg YT series (class-typical) | hydraulic steering; far above on-road tractors |

**Medium-confidence geometry:** tire size **11R22.5 → wheel radius ≈ 0.50–0.52 m**. The specific tire spec
was *refuted* as not sourceable from the T2 sheet, but 11R22.5 is the standard heavy-truck tire and a defensible estimate.

---

## 2. Estimated parameters (formulas + worked numbers)

These are **not published by any manufacturer** and must be extrapolated.

### `lf` / `lr` — CG-to-axle split  (MEDIUM-LOW confidence — weakest params)
Formula: **`lf = L·(Wr/W)`, `lr = L·(Wf/W)`**, with Wf/Wr the static front/rear axle loads.

No reliable static bobtail split exists for a standard 4x2 (the T2's "12,000/30,000 lb" figures were **refuted** —
those are axle *capacity ratings*, not static weights). Bobtail hostlers are modestly **front-biased**
(cab + engine forward, no trailer on the rear). Using ≈ **55% front / 45% rear**:

- `lf = 2.946 × 0.45 ≈ **1.33 m**`
- `lr = 2.946 × 0.55 ≈ **1.62 m**`

(Coincidentally almost identical to the old placeholder 1.2/1.6.) **This flips hard when a loaded trailer
is added** (kingpin load over the rear axle) — see §3.

### `Iz` — yaw moment of inertia  (MEDIUM confidence — physics textbook, inputs are envelope estimates)
Two accepted estimators (McHenry review, citing SAE 970951):

- **Box/slab:** `Iz = m·(L²+W²)/12` (optionally × 0.92 empirical correction — a uniform box overestimates real vehicles)
  → `6,577 × (5.11² + 2.54²)/12 = 6,577 × 2.71 ≈ **17,800 kg·m²**`  (≈ **16,400** with 0.92 factor)
- **"mab" rule** (`Iz ≈ m·lf·lr`, described as the most commonly cited estimator, "very good results"):
  `6,577 × 1.33 × 1.62 ≈ **14,200 kg·m²**`

Recommended bobtail diesel: **`Iz ≈ 16,000 kg·m²`** (bracketed 14,000–18,000).

### `Cf` / `Cr` — cornering stiffness  (LOW confidence — no surviving primary tire source for this exact truck)
Anchored heavy-truck tire data (UMTRI/HSRI; Ervin & Winkler, DOT-HS-802-141, 1976; SMAC/msmac model):
**~600–722 lb/deg per tire ≈ 152,000–184,000 N/rad per fully-loaded tire.**
Axle stiffness = per-tire × number of tires (McHenry rule: ×2 for a dual location, ×4 for tandem):

- **Front axle** (2 single tires): `Cf ≈ 2 × 160,000 ≈ **320,000 N/rad**` (loaded)
- **Rear/drive axle** (4 dual tires): `Cr ≈ 4 × 160,000 ≈ **640,000 N/rad**` (loaded)

Load-scaling rule (cornering coefficient ≈ 8–13 % of vertical load per degree for truck tires, higher % at
lighter load): for the lighter **bobtail** state, drop to per-tire ≈ 90,000–100,000 N/rad →
`Cf ≈ 190,000`, `Cr ≈ 200,000 N/rad`. Since the trained policy is noted as "kinematic-dominant," exact Cf/Cr
mostly shape transients.

### `Cd` / `A` — aerodynamics  (LOW confidence — and largely irrelevant at governed ≤72 km/h)
- **`A` = width × cab height ≈ 2.54 × 3.20 ≈ 8.1 m²`** (envelope); tighter frontal-area estimate ≈ **7 m²**.
- **`Cd`**: no primary value for a blunt yard tractor. Class 8 streamlined tractors are CD ≈ 0.60 (DOE);
  a blunt terminal-tractor cab is higher — estimate **`Cd ≈ 0.8`** (range 0.7–0.9).

Keep `rho = 1.225`, `dt = 0.1` unchanged.

---

## 3. The three load states (trailer modeled as added mass on the kinematic tractor)

| State | Total mass `m` | How to get it | `Iz` | `lf / lr` shift |
|---|---|---|---|---|
| **① Bobtail** (required) | **6,577 kg** | Spec sheet, primary | ≈ 16,000 kg·m² | 1.33 / 1.62 m (≈55/45 front) |
| **② + empty trailer** | **≈ 13,600 kg** | bobtail + empty 53′ van tare ≈ 7,000 kg * | ≈ 16,000 + 7,000·d² | CG shifts rearward; recompute with new Wr |
| **③ + loaded trailer** | **≈ 36,740 kg** (= GCW 81,000 lb) | add up to fifth-wheel/GCW cap ≈ 30,160 kg | ≈ 16,000 + 30,160·d² ≈ **~93,000 kg·m²** | strongly rear-biased |

\* Empty-trailer tare (~15,000 lb / ~7,000 kg for a 53′ dry van) is general-knowledge, **not** in the verified
set — flag as an estimate; a reefer is heavier (~7,700 kg).

**Formulas for the loaded shifts** (added kingpin mass `m_add` sits at the fifth wheel, ≈ over the rear axle,
distance `d ≈ lr ≈ 1.6 m` from the bobtail CG):

- **Mass:** `m_new = m_bobtail + m_add`
- **Yaw inertia (parallel-axis):** `Iz_new ≈ Iz_bobtail + m_add · d²`
- **CG split:** recompute `Wf_new, Wr_new` (kingpin load adds almost entirely to `Wr`), then
  `lf_new = L·(Wr_new/W_new)`, `lr_new = L·(Wf_new/W_new)` — CG moves toward the rear axle, so `lr` shrinks and `lf` grows.
- **Cornering stiffness:** `Cr` rises with the heavier rear-axle load (scale per-tire stiffness by the new rear
  load, up to the SMAC ~600 lb/deg-per-tire ceiling near full load); `Cf` barely changes.

**Cap:** fifth-wheel vertical rating is 80,000 lb (36,287 kg), but GCW (81,000 lb) binds total combination mass.
Heavy container-terminal EV models reach GCW 195,000 lb (88,450 kg) for an extreme case.

---

## 4. On-road Class 8 comparison (Tesla Semi / Freightliner Cascadia)

| Param | Value | Source |
|---|---|---|
| Bobtail mass `m` | ~9,000 kg (Cascadia) / 10,430 kg (Tesla Semi LR, 23,000 lb) | Tesla Semi specs; Ritchie/Cascadia |
| Wheelbase | **5.49 m** (216 in, Cascadia) — ~1.9× the hostler | Ritchie Cascadia 125 |
| Max GCW | 80,000 lb (36,287 kg) federal; Tesla Semi 82,000 lb | DOE; Tesla |
| `Cd` | **0.60** (streamlined baseline); Tesla Semi **0.40** | DOE/LLNL; Tesla |
| Width / cab height | 2.59 m / 4.11 m (Tesla Semi) → `A` ≈ 10 m² | Tesla Semi |
| Max speed | 113 km/h (70 mph) | Tesla Semi |
| Accel | ~1.34 m/s² (0–60 mph in 20 s @ 80,000 lb) | Tesla Semi |

**Key contrast:** the yard hostler is short-wheelbase (2.95 vs 5.49 m), blunt (Cd ~0.8 vs 0.6), governed slow
(72 vs 113 km/h), steers to 50° (vs ~30–40°), and is lighter bobtail — but reaches the same ~36,700 kg loaded GCW.

---

## 5. Suggested drop-in values (mirroring the `tesla_model_s_vehicle_params` keys)

For a **bobtail diesel Kalmar Ottawa T2** — reported here for reference, not yet applied to any config file:

```python
m  = 6577      # kg   — primary (spec sheet)
Iz = 16000     # kg·m² — ESTIMATE, box formula ×0.92
Cf = 190000    # N/rad — ESTIMATE, 2× per-tire (bobtail load)
Cr = 200000    # N/rad — ESTIMATE, 4× per-tire (bobtail load)
lf = 1.33      # m    — ESTIMATE, 55/45 static split
lr = 1.62      # m    — ESTIMATE
Cd = 0.8       # –    — ESTIMATE (blunt cab; aero ~negligible at 72 km/h)
A  = 7.0       # m²   — ESTIMATE (frontal area)
dt = 0.1       # s    — unchanged
# also: wheelbase L=2.946 m, max_steer=0.87 rad (50°), v_max=20.1 m/s, rho=1.225
```

---

## 6. Confidence summary — what's solid vs. what to treat cautiously

- **HIGH (primary-sourced):** mass, wheelbase, dimensions, top speed, GCW, fifth-wheel rating, 50° steering.
- **MEDIUM:** `Iz` (textbook formula, envelope inputs); wheel radius (~0.52 m, tire size refuted for this exact truck).
- **LOW / weakest — flag in the thesis:**
  - `lf/lr` split — no static bobtail axle weights published for a standard 4x2; only real split found was a
    front-heavy 4x4 variant (Kalmar TR618i: 68% front / 32% rear, not representative).
  - `Cf/Cr` — no per-tire data survived verification for this truck; built from UMTRI heavy-truck rule of thumb.
  - `Cd` — no blunt-hostler value exists; only the Class 8 CD ≈ 0.60 is sourced.
  - Empty-trailer tare mass — general knowledge, unverified.

### Refuted claims (do NOT use)
- Front axle 12,000 lb / rear 30,000 lb "static split" — these are axle **capacity ratings**, not static weights (1-2).
- 11R22.5 tire / 0.526 m wheel radius as sourced from the T2 sheet — unverified (1-2); use only as a general estimate.
- Kalmar Ottawa T2E bobtail = 20,000 lb — refuted; correct EV figure is 17,600 lb / 7,983 kg (1-2).

### Open questions worth chasing if higher fidelity is needed
1. Measured per-tire cornering stiffness (or normalized cornering coefficient) for 11R22.5-class tires; confirmed tire size / wheel radius for the T2.
2. True static bobtail axle split (front/rear) for a standard 4x2 hostler, vs. axle capacity ratings — to fix `lf/lr`.
3. Track width, turning radius, rated max accel/decel for the T2/T2E (none surfaced in verified claims).
4. A defensible blunt-cab drag coefficient (distinct from Class 8 CD ≈ 0.60) — though aero is minor at governed speeds.

---

## 7. Sources

**Primary (manufacturer / government):**
- Kalmar Ottawa 4x2 DOT/EPA Specification Sheet (2022) — https://www.kalmarottawa.com/49bda7/globalassets/media/278871/278871_Ottawa-4X2-DOT-EPA-Specification-Sheet-English-2022.pdf
- Kalmar Ottawa T2 4x2 product page — https://www.kalmarottawa.com/terminal-tractors/ottawa-T2-4x2/
- Kalmar Ottawa Electric Terminal Tractor brochure (T2 EV) — https://www.kalmarottawa.com/4994d3/globalassets/media/571466/571466_Kalmar-Ottawa-Electric-Terminal-Tractor-brochure.pdf.pdf
- Kalmar Heavy Terminal Tractor TR618i 4x4 spec sheet (USA) — https://www.kalmarottawa.com/49c4ac/globalassets/media/242002/242002_Kalmar-Heavy-Terminal-Tractor-TR618i-4x4-Spec-sheets-USA-1.pdf
- Kalmar Global Ottawa T2 page — https://www.kalmarglobal.com/equipment/terminal-tractors/ottawa-t2-yard-truck-terminal-tractor/
- Terberg Taylor Americas YT-series — https://terbergtaylor.com/terminal-tractors/yt-series
- DOE Heavy Vehicle Aerodynamics report (Class 8 CD ≈ 0.60) — https://digital.library.unt.edu/ark:/67531/metadc723141/m2/1/high_res_d/770966.pdf
- DOE/OSTI heavy-truck aero/axle report — https://www.osti.gov/servlets/purl/1073121
- NHTSA/UMTRI heavy-truck tire cornering data: Ervin, Winkler et al., DOT-HS-802-141 (Dec 1976) [cited via SMAC/msmac]

**Dealer spec sheets (primary-equivalent):**
- Syverson Truck — Ottawa T2 — https://www.syversontruck.com/pdf/Ottawa-T2.pdf
- Rush Truck Centres — Kalmar Ottawa T2 electric — https://www.rushtruckcentres.ca/fckimages/pdf/kalmar-ottawa-t2-electric.pdf

**Formula / secondary references:**
- McHenry Software — Yaw Inertia technical note (cites SAE 970951, MacInnis et al. 1997) — https://www.mchenrysoftware.com/forum/Yaw%20Inertia.pdf
- McHenry Software — Truck/Trailer Tire Properties (axle-summation rule) — https://www.mchenrysoftware.com/medit32/readme/msmac/truck.trailertireproperties.htm
- McHenry Software — Tire cornering stiffness calculation example — https://mchenrysoftware.com/medit32/readme/msmac/examplestirecorneringstiffnesscalculation1.htm
- Estimating vehicle inertia properties with the NHTSA database — https://kktse.github.io/jekyll/update/2020/10/25/estimating-vehicle-inertia-properties-with-nhtsa-database.html
- TruckScience — axle-weight moment method — https://truckscience.com/en-us/calculate-axle-weights/

**Class 8 comparison:**
- Tesla Semi specs — https://www.teslasemi.com/specs
- Freightliner Cascadia (Wikipedia) — https://en.wikipedia.org/wiki/Freightliner_Cascadia
- Ritchie Specs — Freightliner Cascadia 125 truck tractor — https://www.ritchiespecs.com/model/freightliner-cascadia-125-truck-tractor
- Terminal tractor (Wikipedia) — https://en.wikipedia.org/wiki/Terminal_tractor
