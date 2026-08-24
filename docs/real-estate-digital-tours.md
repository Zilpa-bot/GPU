# Real Estate Digital Tours — Deep Research & Feature Build Plan

**Version:** 1.0 · **Date:** 2026-08-24 · **Scope:** Interior + exterior digital tour features, from table-stakes to frontier, each with a build plan, examples and links.

---

## How to read this document

Every feature is written as a **plan**, using the same seven fields:

| Field | Meaning |
|---|---|
| **What** | One-paragraph definition of the feature |
| **Why** | The business reason it exists (conversion, trust, cost, compliance) |
| **Plan** | Numbered build/rollout steps — the thing you actually execute |
| **Stack** | Concrete tools, APIs, formats, hardware |
| **Effort / Cost** | Rough build effort and per-property running cost |
| **Watch-outs** | Where teams get burned |
| **Links** | Primary sources and product pages |

Features are grouped into **tiers**:

- **Tier 1 — Baseline**: if you don't have these you are not competitive. Weeks to ship.
- **Tier 2 — Professional**: what a serious listing/leasing product ships in 2026. 1–2 quarters.
- **Tier 3 — Advanced/Frontier**: differentiators, radiance fields, XR, AI agents, geospatial. 2+ quarters, higher risk.

A **statistics caveat** applies throughout: most engagement numbers in this space (e.g. "87% more views", "403% more inquiries", "days-on-market down 44%") come from vendor blogs and NAR-adjacent marketing material, not peer-reviewed studies. Treat them as directional, and instrument your own funnel (see §C1) before you defend a budget with them.

---

## Table of contents

- [Part 0 — Market and technology landscape](#part-0)
- [Part A — Interior features](#part-a) — A1–A20
- [Part B — Exterior features](#part-b) — B1–B15
- [Part C — Cross-cutting: analytics, distribution, compliance, delivery](#part-c) — C1–C6
- [Part D — Capture hardware and pipeline reference](#part-d)
- [Part E — Build vs buy, reference architecture, phased roadmap](#part-e)
- [Part F — Source list](#part-f)

---

<a name="part-0"></a>
## Part 0 — Market and technology landscape (what changed by 2026)

Four structural shifts define the current moment:

**1. Radiance fields went mainstream.** 3D Gaussian Splatting (3DGS), introduced at SIGGRAPH 2023 by INRIA, moved from research to shipped product in 2025: Zillow launched **SkyTour** (drone footage → interactive splat flight around a home's exterior) in July 2025, and CoStar/Matterport answered with 3D exteriors. 3DVista added native 3DGS in its 2025.0 release. The practical consequence: *continuous free-viewpoint navigation* is now an option, not just teleport-between-panoramas.

**2. The portal/platform relationship broke.** CoStar closed its **$1.6B acquisition of Matterport** in early 2025; in **October 2025 Zillow stopped natively embedding Matterport tours** when the API agreement lapsed. Distribution is now a first-class architectural concern — do not build a tour product that assumes one embed target survives forever.

**3. Capture got cheap and phone-first.** A $549 Insta360 X5 or an iPhone Pro with LiDAR now produces usable input for tours, floor plans (Apple **RoomPlan**) and splats. LiDAR-grade rigs (Matterport Pro3, iGUIDE) still win where *measurement* is the deliverable — iGUIDE claims ≤0.5% distance and ~1% square-footage uncertainty, inside ANSI Z765 / RESO RMS tolerances.

**4. AI moved from post-production into the tour itself.** Virtual staging collapsed from ~$30/image and 48h to cents and seconds; Matterport ships **Defurnish** (AI furniture removal) and AI property descriptions; and conversational agents now live *inside* the tour to qualify leads and book showings. Regulation followed: **California AB 723, effective January 2026, makes non-disclosure of digitally altered real estate photos a misdemeanor.**

**Tour format taxonomy** — pick deliberately, most products ship 2–3 of these:

| Format | Navigation | Spatial data | Capture cost | Best for |
|---|---|---|---|---|
| Photo gallery + video | Linear | None | $ | Every listing |
| 360° panorama tour (hotspot) | Teleport | None | $ | Volume/rental |
| 360° + LiDAR digital twin | Teleport + dollhouse | Mesh + measurements | $$$ | Resale, CRE, insurance |
| Gaussian splat / radiance field | Free-fly, continuous | Implicit (poor metrics) | $$ | Exterior, hero experience |
| Real-time engine (UE5) | Free-walk | Authored (BIM) | $$$$ | Pre-construction, configurators |

---

<a name="part-a"></a>
# Part A — Interior features

## Tier 1 — Baseline interior

### A1. 360° panorama tour with hotspot navigation

**What.** A set of equirectangular spherical images (one per room / per capture point), linked by clickable hotspots so the viewer teleports room to room. The single most common "virtual tour" on earth. Output is a URL that embeds in an iframe.

**Why.** Cheapest path from "photos" to "space". Sets the baseline expectation for any listing above the lowest price tier, and it is the only tour format that reliably syndicates through MLS → portals as a plain link.

**Plan.**
1. Define a capture SOP: tripod at ~1.5 m, one pano per room plus one per hallway junction, doorways always visible from two adjacent points, HDR bracketing on, photographer excluded via remote trigger.
2. Ingest: upload equirectangular JPEGs (min 6K×3K; 8K for hero rooms), auto-level horizon, auto-detect north.
3. Author the graph: nodes = panos, edges = hotspots. Auto-suggest edges via image similarity/EXIF position, human-confirm.
4. Add per-node metadata: room name, ceiling height, orientation, default view yaw/pitch.
5. Render: tile each pano into a multi-res cube map (see §C6) so mobile loads a low-res level in <1s.
6. Publish two URLs — branded and unbranded (see §C3).

**Stack.** Capture: Ricoh Theta X / Z1, Insta360 X5. Viewer (buy): Kuula, CloudPano, 3DVista, Panoee, RICOH360 Tours. Viewer (build): [Pannellum](https://pannellum.org/) (~21KB, zero-dep), [Marzipano](https://www.marzipano.net/) (Google, tiling tool included), [Photo Sphere Viewer](https://photo-sphere-viewer.js.org/) (three.js, best marker system).

**Effort / Cost.** Buy: hours, $0–36/mo (Kuula Pro ≈$20/mo, CloudPano Pro Plus ≈$33/mo annual, 3DVista $499 one-time perpetual). Build: 3–5 engineer-weeks for a solid authoring UI + viewer.

**Watch-outs.** Horizon tilt and nadir (tripod) artifacts read as amateur instantly. Don't ship a viewer without a mini-map — users get lost after ~6 nodes.

**Links.** [Pannellum](https://pannellum.org/) · [Photo Sphere Viewer](https://photo-sphere-viewer.js.org/) · [Open-source 360 libraries round-up 2026](https://portalzine.de/open-source-virtual-tour-360-panorama-libraries-in-javascript-2026/) · [Kuula: adding tours to MLS](https://blog.kuula.co/mls)

---

### A2. Guided walkthrough video + vertical short-form cutdowns

**What.** A continuous handheld/gimbal walkthrough (60–120s landscape) plus 3–5 vertical 15–30s cutdowns for Reels/TikTok/Shorts, with captions and licensed music. Increasingly generated automatically from raw footage.

**Why.** Video is the top-of-funnel that *drives traffic to* the tour; the tour is the mid-funnel that qualifies. NAR-cited figures put listings with video at materially higher inquiry rates. Cutdowns are where the audience actually is.

**Plan.**
1. Shoot a single unbroken "buyer's path": approach → entry → living → kitchen → primary suite → outdoor.
2. Auto-ingest to an editor that does scene detection, stabilization, speed ramps at transitions.
3. Generate AI voiceover from the listing's structured data (beds/baths/sqft/features) — never from a hallucinated description.
4. Auto-caption (burned-in), brand-safe intro/outro, and produce 9:16, 1:1, 16:9 renders.
5. Multi-language variants for your market's top 3 non-English languages.

**Stack.** Capture: gimbal (DJI Osmo), or reuse 360 footage reframed to 16:9. Automation: Sai, HeyGen (177+ languages), Synthesia, Colossyan, or a ffmpeg + TTS pipeline you own.

**Effort / Cost.** Manual edit historically $200–500/property; AI pipelines take it to near-zero marginal cost.

**Watch-outs.** AI avatars presenting a home reads as spam to many buyers — prefer voiceover-over-footage to talking-head. Music licensing on syndicated video is a real liability.

**Links.** [HeyGen: AI video tools for real estate](https://www.heygen.com/blog/best-ai-video-tools-real-estate) · [Multilingual listing video makers](https://www.heygen.com/blog/best-ai-multilingual-property-listing-video-maker)

---

### A3. 2D floor plan with room dimensions

**What.** A scaled schematic plan per level, with room labels, dimensions, doors/windows, and total finished area — delivered as an image and as vector/JSON.

**Why.** Consistently one of the top-3 features buyers say they want, and the only artifact that answers "will my furniture fit / can I move a wall". Also the substrate for tour navigation (A7), and for measurement disputes.

**Plan.**
1. Choose the source of truth: laser/LiDAR capture (iGUIDE, Matterport Pro3), phone LiDAR (RoomPlan), or manual drafting from photos.
2. Fix a measurement standard **before** first delivery — ANSI Z765 for US detached residential, RESO RMS for MLS data exchange, local equivalent elsewhere (RICS IPMS internationally).
3. Draft and QA: label every room, mark excluded areas (below-grade, sub-7ft ceilings) explicitly.
4. Publish three artifacts: raster PNG for portals, vector SVG/PDF for print, structured JSON (rooms, polygons, areas) for your own app.
5. Stamp every plan with the standard used, the tolerance, and a disclaimer.

**Stack.** iGUIDE (LiDAR camera + drafting service; claims ≤0.5% distance / ~1% area uncertainty), Matterport auto-generated schematic plans, Apple [RoomPlan](https://developer.apple.com/documentation/roomplan) (parametric room model on iPhone/iPad LiDAR, <2 min), CubiCasa, magicplan.

**Effort / Cost.** $15–60/property outsourced drafting; near-zero if generated automatically from a LiDAR capture you already took.

**Watch-outs.** Square footage is a litigation surface. Never publish an area number without the standard and tolerance beside it. Phone-LiDAR plans typically land within 1–2% — fine for marketing, not for appraisal.

**Links.** [iGUIDE measurement & drafting standards](https://help.youriguide.com/hc/en-us/articles/27645625048210-iGUIDE-Measurements-and-Drafting-Standards) · [Apple RoomPlan](https://developer.apple.com/documentation/roomplan) · [iGUIDE: floor plan vs 3D tour](https://goiguide.com/blogs/floor-plan-vs-3d-virtual-tour)

---

### A4. HDR photo set with AI enhancement pipeline

**What.** The 20–40 stills that still carry most listing traffic, run through a deterministic enhancement pipeline: exposure blending, lens/perspective correction, window pull, colour cast removal, decluttering, and lawn/sky work for the exteriors.

**Why.** Photos are the thumbnail; the tour is what happens after the click. A weak hero image means the tour is never opened.

**Plan.**
1. Standardize capture: bracketed exposures, tripod, vertical lines vertical, lights on, blinds set consistently.
2. Build an enhancement queue with a fixed order of operations and a per-brokerage preset.
3. Enforce a **realism policy**: allowed = exposure/colour/lens/sky/lawn/declutter; disclosed = staging, item removal, renovation; forbidden = removing defects (cracks, stains, damage).
4. Attach machine-readable edit provenance (what was altered) to every image — this is what makes disclosure automatic rather than manual.
5. Feed the same corrected set to the tour, the video, and the print sheet so nothing looks colour-mismatched.

**Stack.** Lightroom/Photoshop batch, PhotoUp, Twilight, AI Home Design, EditThisPic, or a hosted diffusion pipeline you control.

**Watch-outs.** Over-saturated skies and impossible window views are the fastest way to lose buyer trust — and, in California, to commit a misdemeanour if undisclosed (AB 723).

**Links.** [Real estate photo editing trends 2026](https://www.photoup.net/learn/new-real-estate-photo-editing-trends) · [Twilight.pics](https://twilight.pics/)

---

## Tier 2 — Professional interior

### A5. True 3D digital twin with dollhouse view

**What.** A registered mesh + textured point cloud of the whole interior, giving three linked view modes: **inside** (first-person), **dollhouse** (the whole home as an open model you can rotate), and **floor plan** (top-down). This is the Matterport-defined format.

**Why.** Dollhouse is the single feature buyers remember; it answers *layout adjacency* — which no set of panoramas can. It is also the gateway to measurement, BIM export and insurance/appraisal use cases.

**Plan.**
1. Capture with a spatially-aware rig: Matterport Pro3 (LiDAR, ~20mm accuracy at 10m), Leica BLK360, or NavVis for large floorplates; scan points every 6–8 ft with overlap.
2. Process to a mesh; QA registration drift on long corridors (the classic failure) and re-scan loops that don't close.
3. Auto-generate the dollhouse and schematic plan from the same capture.
4. Hand-fix: trim exterior spill, label rooms, set the default start point (front door, not the closet).
5. Publish with three-mode navigation and a persistent mini-map.

**Stack.** Buy: Matterport (Starter ~$65/mo up to Business ~$309/mo; active-space caps matter more than price), iGUIDE, Cupix, Giraffe360. Build: Reality Capture / RealityScan, COLMAP + Open3D, Potree/three.js for delivery.

**Effort / Cost.** 45–120 min on-site for a 2,500 sqft home; $150–400 per property fully outsourced.

**Watch-outs.** Matterport space caps punish high-volume photographers (25 active spaces on the mid tier). Glass, mirrors and dark rooms wreck depth capture — plan lighting. And since Oct 2025, a Matterport tour no longer embeds natively on Zillow: keep a portable fallback (A1/A13).

**Links.** [Matterport digital twin viewer update](https://matterport.com/blog/refreshing-the-way-to-explore-your-digital-twins) · [Matterport pricing breakdown 2026](https://www.thefuture3d.com/blog/matterport-pricing-guide-2026/) · [Zillow removes Matterport tours (Inman, Oct 2025)](https://www.inman.com/2025/10/20/zillow-removes-matterport-3d-home-tours-from-its-sites/)

---

### A6. In-tour measurement mode

**What.** Click two points inside the tour and get a distance; measure a wall, a doorway, a window opening, a ceiling height, or an appliance alcove, with a saved-measurements list.

**Why.** Converts a marketing asset into a *decision* asset: "does my sofa/fridge/wheelchair fit". Heavily used by out-of-market and relocating buyers, renovators, and by insurance/restoration users.

**Plan.**
1. Require a depth-bearing capture (LiDAR or dense photogrammetry) — do not offer measurement on panorama-only tours.
2. Implement snapping to detected planes (floor, wall, ceiling) so a two-click measure doesn't drift in Z.
3. Show tolerance in the UI (e.g. "±1%") and a persistent disclaimer; log every measurement taken.
4. Add a "save + export" pane producing a PDF measurement sheet.
5. Gate it behind email capture if you want it as a lead magnet (common pattern).

**Stack.** Matterport measurement mode (99% accuracy claim on Pro2/Pro3 captures), iGUIDE, Cupix; build-your-own on a point cloud with three.js raycasting + RANSAC plane fit.

**Watch-outs.** Measurement accuracy claims are a legal exposure. Never let the number appear without the tolerance. Phone-captured tours should say "approximate" everywhere.

**Links.** [Matterport Cortex AI](https://matterport.com/cortex-ai) · [iGUIDE vs manual measurement](https://goiguide.com/compare/iguide-vs-manual-measurement)

---

### A7. Floor-plan-linked navigation and mini-map

**What.** A persistent floor-plan overlay showing where the viewer is standing, which way they're facing (view cone), and every other capture point — click the plan to jump.

**Why.** The single highest-leverage usability fix in tour design. Without it, drop-off spikes after the third teleport because people lose orientation. Also makes multi-level homes comprehensible.

**Plan.**
1. Register every pano node to plan coordinates (automatic if you captured with LiDAR; otherwise a manual pin-drop step in the authoring tool).
2. Render an SVG mini-map with node dots, a live yaw cone, and a level switcher.
3. Sync bidirectionally: moving in the tour moves the dot; clicking the plan moves the tour.
4. Collapse to a corner button on mobile, expand on tap.
5. Log plan-clicks as an analytics event — it's a strong intent signal per room.

**Stack.** Any viewer with a plan layer: Matterport, iGUIDE, 3DVista, CloudPano; or SVG + your own state store on top of Pannellum/PSV.

**Effort.** ~1–2 engineer-weeks on a custom viewer. Highest ROI per week of work in this whole document.

---

### A8. Rich hotspots / info tags

**What.** Anchored 3D annotations inside the space: text, photos, video, PDF spec sheets, external links, live data, e-commerce links. Matterport calls them Mattertags; every platform has an equivalent.

**Why.** This is where a tour stops being pretty and starts answering questions — appliance model numbers, "new roof 2024", HOA docs, warranty info, or shoppable furniture links. In CRE it carries the whole spec pack.

**Plan.**
1. Define a tag taxonomy: `feature`, `upgrade`, `spec`, `disclosure`, `media`, `shop`. Colour-code by type.
2. Build an authoring mode: click a surface → tag anchors to that 3D point with a leader line.
3. Add a tag index panel listing all tags so they're discoverable without hunting.
4. Template tags per property type so agents fill in blanks instead of inventing.
5. Track tag opens per tag — the highest-opened tag tells you what the listing copy is missing.

**Watch-outs.** More than ~12 tags per home and everyone ignores them. Keep tags out of the default viewport of the hero shot.

**Links.** [Matterport tags/media in digital twins](https://matterport.com/blog/refreshing-the-way-to-explore-your-digital-twins)

---

### A9. AI virtual staging and de-furnishing

**What.** Two inverse operations: fill an empty room with photorealistic furniture in a chosen style, or strip an occupied room back to empty. Applied to stills, and increasingly to whole panoramas and 3D twins.

**Why.** Empty rooms photograph badly and read smaller; cluttered rooms read as "someone else's home". AI collapsed the cost from ~$30/image and 48 hours to under a dollar and ~30 seconds, so the question is no longer *whether* but *how consistently and how disclosed*.

**Plan.**
1. Pick a lane: AI-first ($0.23–$15/image, instant, volume) vs designer-led ($15–75/image, bespoke, revision control). Most operations need both, routed by listing price band.
2. Enforce **geometric fidelity**: staged furniture must respect real scale and not conceal defects or alter architecture. Validate against the floor plan.
3. Keep style consistent across every image of the same home — mismatched styles are the tell-tale of cheap AI staging.
4. **Disclosure pipeline (non-negotiable):** watermark or caption every altered image ("Virtually staged"), plus a line in listing remarks; store the original alongside the altered version.
5. For twins: apply staging as a *toggleable layer* ("furnished / unfurnished") rather than baking it in — this is both more useful and more honest.

**Stack.** Matterport **Defurnish** (AI furniture removal, part of Project Genesis), Collov, AI Home Design, Roomstage, Styldod, REimagineHome, virtualstagingai.

**Watch-outs.** **California AB 723 (effective January 2026)** makes non-disclosure of digitally altered real estate photos a misdemeanour; NAR and most state commissions require clear disclosure. Never virtually repair a defect.

**Links.** [Matterport AI Defurnish & descriptions (HousingWire)](https://www.housingwire.com/articles/matterport-ai-defurnish-photos-property-descriptions/) · [MLS virtual staging rules & disclosure (2026)](https://www.roomstage.ai/mls-virtual-staging-rules) · [Virtual staging tool comparison 2026](https://collov.ai/blog/choosing-ai-virtual-staging-for-real-estate-2026-comparison)

---

### A10. Guided / auto-play narrated tour path

**What.** A director's cut through the twin: a scripted camera path with timed narration, captions and highlight callouts, that plays on load for passive viewers and can be exited into free navigation at any moment.

**Why.** ~Half of tour visitors never actively navigate. A guided path converts a passive scroll into a completed walkthrough, and it makes the tour usable as a *presentation* on a screen share or in a lobby kiosk.

**Plan.**
1. Author a path as an ordered list of (node, yaw, pitch, dwell, caption) keyframes; interpolate camera moves.
2. Generate narration from structured listing data via TTS; produce per-language tracks.
3. Add captions (also serves accessibility — see A12) and a chapter list.
4. Always show a persistent "explore on your own" affordance; never trap the user.
5. Measure completion rate of the guided path as a primary KPI.

**Stack.** 3DVista guided tours, Matterport highlight reels, or a keyframe player on your own viewer.

---

### A11. Multi-unit / floor-plan-type tour switching (multifamily & new build)

**What.** One tour experience covering a *building*: pick a floor plan type (A1, B2, penthouse), then a specific unit and floor, and see that unit's tour, availability, price, and the view from that elevation.

**Why.** Multifamily and pre-sale condo don't sell a single space, they sell an inventory. Without unit-level switching, leasing teams end up maintaining 40 separate tour links, which nobody does correctly.

**Plan.**
1. Model the data: Building → Floor → Unit → FloorPlanType → Tour asset. One tour per *type*, not per unit; per-unit deltas are view direction, floor height, finish package.
2. Build a stacking-plan selector UI (interactive building elevation) as the entry point.
3. Bind live availability/pricing from the PMS/CRM so an unavailable unit is visibly gone.
4. Substitute the view-from-window per floor (see B10) instead of re-shooting each unit.
5. Include amenity spaces (gym, lounge, rooftop) as a separate tour graph linked from the lobby node.

**Stack.** Realync, Peek/LCP360, Matterport + custom front-end, Engrain SightMap for stacking plans; RealPage/Yardi/Entrata for availability.

**Links.** [Realync 3D/360 tours for apartments](https://www.realync.com/3d-virtual-tours/)

---

### A12. Accessibility layer (WCAG 2.1 AA)

**What.** Keyboard-navigable tour, screen-reader semantics for every node and hotspot, captions and audio description on tour video, sufficient contrast in the UI chrome, and a text-equivalent walkthrough of the property.

**Why.** Two overlapping legal regimes: ADA (real estate sites treated as places of public accommodation, with US courts benchmarking **WCAG 2.1 AA**) and the **Fair Housing Act** (equal access to housing information regardless of disability). An inaccessible tour can be a violation of both. Realync publicly certified WCAG 2.1 in multifamily precisely because of this.

**Plan.**
1. Keyboard: arrow keys to look, Tab to cycle hotspots/nodes, Enter to move, Esc to exit immersive mode. Visible focus ring throughout.
2. Semantics: each node gets an accessible name ("Kitchen, north-facing"); each hotspot is a real `<button>` with a label; the mini-map exposes a list view.
3. Provide a **text tour**: room-by-room description with dimensions, generated from the floor plan JSON + AI, human-reviewed.
4. Captions + audio description track on any narrated content.
5. Audit with axe/Lighthouse plus one real screen-reader pass (NVDA + VoiceOver) per release; document conformance.

**Watch-outs.** Overlay widgets ("accessibility plugins") do not fix a canvas-based tour and have themselves attracted litigation. Fix the viewer, don't bolt on a widget.

**Links.** [ADA compliance for real estate websites 2026 (accessiBe)](https://accessibe.com/blog/knowledgebase/ada-compliance-for-real-estate) · [Realync WCAG 2.1 announcement](https://www.prweb.com/releases/realync-announces-wcag-2-1-compliance-enabling-web-accessibility-in-multifamily-831479928.html) · [Fair housing + ADA in digital apartment marketing](https://www.marketapts.com/blog/staying-compliant-housing-ada-laws-digital-apartment-marketing/)

---

## Tier 3 — Advanced interior

### A13. Gaussian splat / radiance-field interior (free-viewpoint)

**What.** Instead of discrete panoramas, the room is represented as millions of anisotropic 3D Gaussians trained from a video walkthrough. The viewer moves *continuously* — walking, leaning, ducking under a beam — at 100+ FPS in a browser, with true view-dependent reflections and specular highlights that a textured mesh cannot reproduce.

**Why.** It closes the "teleport gap". Everything else in a tour is a compromise on locomotion; 3DGS removes it. Industry consensus going into 2026 is that radiance fields are becoming a fundamental imaging medium, and the major portals shipped support in 2025.

**Plan.**
1. **Capture**: 3–6 minutes of smooth video per floor at 4K60, shutter fast enough to avoid motion blur, even lighting, loop closure (end where you started). A phone is sufficient; a 360 camera gives better coverage per pass.
2. **Pose**: run structure-from-motion (COLMAP or GLOMAP) to recover camera intrinsics/extrinsics; this is where most failures happen — featureless white walls need added texture cues.
3. **Train**: `splatfacto` in [Nerfstudio](https://docs.nerf.studio/nerfology/methods/splat.html) on the [gsplat](https://github.com/nerfstudio-project/gsplat) backend (≈4× less memory, ~10% faster than the reference implementation), or Postshot/Brush for a desktop GUI workflow.
4. **Clean**: crop the scene bounds, delete floater Gaussians, mask out people/reflections of the operator.
5. **Compress**: apply Self-Organizing Gaussians (SOG) compression — roughly **8 MB per 500K Gaussians** with near-unchanged fidelity, down from >1 GB a year prior. Ship `.spz`/`.sog` rather than raw `.ply`.
6. **Deliver**: WebGL/WebGPU viewer with a constrained camera (rails or collision volume) so users can't fly into a wall and see the scene fall apart.
7. **Fallback**: always ship an A1 panorama tour for devices that can't render the splat.

**Stack.** Training: Nerfstudio + gsplat, Jawset Postshot, Brush, Luma AI. Viewers: PlayCanvas SuperSplat, Babylon.js, three.js splat renderers, Spline. Platforms with native 3DGS: 3DVista 2025.0+, Zillow SkyTour (exterior), splatlabs, Real Horizons.

**Effort / Cost.** GPU training 20–90 min/scene on an A10/4090 class card; ~$1–5 of compute per property. Engineering: 4–8 weeks to productionize capture→train→compress→serve.

**Watch-outs.** Splats have **no reliable metric geometry** — do not offer measurement on top of one without a co-registered LiDAR pass. Transparent and mirrored surfaces produce ghosting. Mobile memory limits are the real ceiling: budget <150 MB decoded.

**Links.** [Nerfstudio splatfacto docs](https://docs.nerf.studio/nerfology/methods/splat.html) · [gsplat (arXiv)](https://arxiv.org/pdf/2409.06765) · [Nerfstudio adds SOG compression](https://radiancefields.com/nerfstudio-adds-compression-to-gsplat) · [3DGS services & web viewers guide 2026](https://www.utsubo.com/blog/gaussian-splatting-guide) · [Gaussian splatting for real estate tours](https://realhorizons.ai/blog/gaussian-splatting-for-real-estate/)

---

### A14. VR / WebXR immersive mode

**What.** A button in the browser tour that says "View in VR". On a Quest 3 or Vision Pro the tour re-renders at 1:1 human scale, stereoscopic, with head tracking — no app install, via the [WebXR Device API](https://www.w3.org/TR/webxr/).

**Why.** Scale perception is the one thing a flat screen cannot deliver: ceiling height, corridor width, whether a room feels tight. For relocation buyers, off-plan sales, and luxury it converts remote interest into offers. Reported effects (vendor-sourced) include large drops in days-on-market; treat as directional.

**Plan.**
1. Add a WebXR session button; feature-detect `navigator.xr.isSessionSupported('immersive-vr')` and hide gracefully otherwise.
2. Render stereo panoramas (capture with a stereo 360 rig, or synthesize stereo from the splat/mesh) — mono panoramas in VR feel flat and cause discomfort.
3. Implement comfort: teleport locomotion by default, vignette on smooth movement, fixed horizon, 72–90 FPS target with aggressive LOD.
4. Add a "short code" entry path — the user types a 6-digit code in the headset browser instead of retyping a URL.
5. For sales galleries, add a passthrough/mixed-reality mode on Vision Pro: anchor a 1:1 rendering of the future unit onto the real gallery floor.
6. Track XR sessions separately in analytics — they behave nothing like desktop sessions.

**Stack.** WebXR + three.js/Babylon.js/A-Frame/PlayCanvas; Meta Quest Browser, Apple Vision Pro Safari; native visionOS app for the premium tier.

**Watch-outs.** Headset penetration is still low — WebXR is a *bonus channel*, never the primary. Anything above ~90 min of headset content is unusable. Test on-device: desktop WebXR emulators lie about performance.

**Links.** [W3C WebXR Device API](https://www.w3.org/TR/webxr/) · [WebXR browser support 2026](https://www.testmuai.com/learning-hub/webxr-compatible-browsers/) · [Apple Vision Pro real estate developer guide](https://r2u.io/en/blog/apple-vision-pro-real-estate-guide/) · [VR tours & immersive experiences 2026](https://www.360tours.studio/blog/future-of-virtual-reality-tours)

---

### A15. In-tour AI concierge (chat + voice)

**What.** A conversational agent living inside the tour that knows *this* property: it answers "how old is the roof", "does the HOA allow dogs", "what's the commute to downtown", walks the user to the relevant room, qualifies them, and books a showing — by text or voice.

**Why.** The tour is the moment of peak intent, and it is exactly the moment nobody is available to answer. Vendors report large lead-response and time-to-lease improvements; the credible mechanism is simply that response latency goes to zero and every visitor gets qualified.

**Plan.**
1. **Ground it**: build a per-property knowledge base (listing fields, disclosures, HOA docs, floor plan JSON, tag content, neighbourhood data). Retrieval-augmented, with **no free-form generation about facts that aren't in the KB**.
2. **Wire it to the space**: give the agent a tool to move the camera ("show me the primary bath" → navigates the tour). This is the feature that makes it feel native rather than bolted-on.
3. **Qualification flow**: adaptive questions — buying/selling, timeline, pre-approval status, neighbourhoods, motivation — rather than a static form. Hand off to a human on high-intent signals.
4. **Compliance guardrails (critical)**: hard-block any answer touching protected classes, "good schools", neighbourhood demographics, "family-friendly", or steering language. Fair Housing violations by chatbot are still violations. Maintain a refusal-and-redirect script.
5. **Escalation + logging**: full transcript into the CRM, with the tour analytics (rooms viewed, dwell) attached to the lead record.
6. Voice: add a low-latency speech pipeline for hands-free use on mobile and in headsets.

**Stack.** LLM + RAG on your own data; commercial options include Perspective AI, Crescendo, Structurely, Ylopo; portal/PM stacks (Knock, Elise) for multifamily.

**Watch-outs.** Hallucinated property facts are a disclosure liability. Log everything. Rate-limit and CAPTCHA — these endpoints get scraped.

**Links.** [Conversational AI for real estate — 5 applications](https://www.crescendo.ai/blog/conversational-ai-for-real-estate) · [AI voice agents for real estate 2026, compared](https://getperspective.ai/blog/ai-voice-agents-for-real-estate-in-2026-7-options-compared-by-conversation-depth)

---

### A16. Live agent-led tour + co-browsing

**What.** A scheduled or on-demand session where an agent walks the prospect through the tour in real time — either live on-site from a phone camera, or screen-sharing/co-driving the 3D tour from a desk, with both cursors visible.

**Why.** It preserves the relationship that self-service removes. For high-value or remote transactions the live tour is the closing tool; the recorded version becomes a follow-up asset.

**Plan.**
1. Implement a shared session: a "guide" role drives the camera, "viewers" follow, with an optional "take control" handoff.
2. Transport over WebRTC (video + data channel for camera state); no app install, mobile-web first.
3. Record every session automatically; auto-generate a highlight clip and email it as follow-up.
4. Add scheduling with round-robin routing and calendar sync.
5. Feed session artifacts (duration, rooms lingered on, questions asked) into the CRM lead score.

**Stack.** Realync (multifamily; live + recorded + 360, Grace Hill), Zoom/Twilio/LiveKit for a build, Matterport SDK for a co-driven twin.

**Links.** [Realync live video tours](https://www.realync.com/platform-live-video/)

---

### A17. Self-guided physical tour orchestration

**What.** The bridge from digital to physical: a prospect verifies identity online, books a slot, and receives a time-boxed credential that opens the building and the specific unit — with in-app wayfinding and a digital tour guide running on their phone while they walk.

**Why.** 24/7 availability without staffing; >66% of consumers prefer self-service options and self-touring measurably compresses days-on-market. Rently alone reports 20M+ self-tours completed.

**Plan.**
1. **Identity + fraud**: government ID scan, selfie match, card authorization hold, blocklist check. This step is the whole risk model.
2. **Access**: issue a credential valid only for the booked window and only for that unit — smart lock at the unit door plus community access at the perimeter.
3. **On-tour guidance**: push the digital tour to their phone with room-by-room prompts, a feature checklist, and a "talk to someone" button.
4. **Telemetry**: door-open events, dwell per unit, exit confirmation; auto-alert if the unit isn't re-secured.
5. **Follow-up**: application link fires within minutes of exit while intent is peaking.

**Stack.** Rently (Smart Bolt Elite), SmartRent, Gatewise, Tour24, ButterflyMX; integrate with the PMS for availability and pricing.

**Watch-outs.** Fair-housing risk hides in eligibility screening for tour access — apply identical criteria to everyone and log the decisions. Also plan for the vacant-unit incident: contents, staging, and insurance.

**Links.** [Rently: choosing self-touring technology](https://use.rently.com/blog/choosing-the-right-self-touring-technology/) · [SmartRent locks & lock boxes](https://smartrent.com/hardware/locks-lock-boxes/) · [Self-guided tours need smart unit locks](https://gatewise.com/blog/self-guided-tours-smart-unit-locks-security-leasing-playbook)

---

### A18. Interior configurator (finishes, packages, layout options)

**What.** A real-time interactive interior where the viewer swaps finish packages (cabinetry, flooring, countertops, paint), lighting scenarios (day/evening), furniture packages, and structural options (den vs third bedroom) — with price deltas attached and the selection persisting into a reservation.

**Why.** The core sales tool for pre-construction and build-to-order. It moves the buyer from evaluating a home to *designing theirs*, and it captures option revenue at the moment of maximum enthusiasm.

**Plan.**
1. Model once in the design tool of record (Revit/Archicad → Datasmith) so the configurator stays in sync with construction documents.
2. Author options as swappable material/mesh variants; Twinmotion's **Configurations** (2026.1) does this natively for presentation, Unreal for full interactivity.
3. Choose delivery: **pixel streaming** (server-rendered UE5, any browser/phone, no install, but $$$ per concurrent session) vs **baked WebGL** (cheap, scales, lower fidelity). Most teams run pixel streaming for the sales gallery and WebGL for the public site.
4. Separate UI from render: React front-end holds state and pricing; the engine only renders (proven pattern in UE 5.7 Pixel Streaming 2 integrations).
5. Persist the configuration to a shareable URL + PDF spec sheet, and push it into CRM/reservation flow with deposit capture.
6. Feed selections back to procurement — the configurator becomes a demand signal, not just a demo.

**Stack.** Unreal Engine 5 + Pixel Streaming 2, Twinmotion 2026.1 Configurations, Unity, three.js/PlayCanvas for the light build; Datasmith for CAD/BIM import.

**Effort / Cost.** 8–20 weeks; GPU streaming ~$0.5–2/concurrent-hour. This is the most expensive feature in this document.

**Watch-outs.** Pixel streaming costs scale with *concurrency* — cap sessions and queue. Render quality that exceeds what will actually be built creates delivery disputes; put a materials disclaimer in the flow.

**Links.** [Twinmotion 2026.1 release (CG Channel)](https://www.cgchannel.com/2026/04/epic-games-releases-twinmotion-2026-1/) · [Unreal Engine for architecture](https://www.unrealengine.com/uses/architecture) · [UE5.7 Pixel Streaming 2 + React UI integration](https://forums.unrealengine.com/t/ue5-7-pixel-streaming-2-react-frontend-ui-integration-plugin-dev/2706135)

---

### A19. AR: scan-your-furniture and fit-it-in-the-listing

**What.** Two directions. (a) The buyer scans their current apartment with RoomPlan/ARKit and drops their real furniture into the listing's 3D model to check fit. (b) On-site, the buyer points their phone at an empty room and sees staged furniture, or a renovation option, anchored in place.

**Why.** Removes the biggest unresolved question in a remote purchase — "does my life fit in this". It is also the most shareable feature you can ship; buyers screenshot it.

**Plan.**
1. Use Apple [RoomPlan](https://developer.apple.com/documentation/roomplan) to capture the user's current space as a parametric model (walls, doors, windows, furniture bounding boxes) in <2 minutes.
2. Normalize both spaces to a common scale/coordinate convention; import the listing's floor plan JSON (A3).
3. Provide a 2D "fit view" first (top-down, drag furniture) — it's more useful than AR for the actual decision — then AR for the emotional payoff.
4. On-site AR: ARKit/ARCore plane detection, occlusion via LiDAR depth, with a catalogue of staging packages.
5. Export a shopping list / dimension sheet from the arrangement.

**Stack.** RoomPlan + ARKit (iOS), ARCore + Scene Semantics (Android), Polycam/Canvas/magicplan for scanning, AR Plan 3D on Android; model delivery in USDZ/glTF.

**Watch-outs.** Accuracy on LiDAR devices lands ~1–2%; on non-LiDAR phones it's much worse — degrade the feature, don't fake it.

**Links.** [Apple RoomPlan](https://developer.apple.com/documentation/roomplan) · [RoomPlan use cases 2026](https://volpis.com/blog/top-use-cases-for-apps-utilizing-apple-roomplan/) · [AR room-planning technologies](https://www.netguru.com/blog/augmented-reality-room-planning-technologies)

---

### A20. Renovation visualizer with cost estimates

**What.** Layered "before / after" states inside the tour: open the kitchen wall, refinish floors, convert the basement, add an ADU — each with a scope description and a cost band sourced from local trade data.

**Why.** Sells the *potential* of dated stock, which is where the margin is. It also converts a listing tour into a lead-gen surface for renovation lenders, contractors and iBuyer-style offers.

**Plan.**
1. Constrain the option set to templated, high-frequency renovations per property archetype rather than free-form generation.
2. Generate the "after" state: AI image-to-image on the panorama for speed, or a modelled variant of the mesh for accuracy on structural changes.
3. Attach a cost band per option from a maintained regional cost table (not from the model), with an explicit "estimate, not a quote" disclaimer.
4. Add a slider/toggle UI so before-after is always one gesture apart, and never default to the renovated state.
5. Route interest to financing and contractor partners — this is the monetization.

**Stack.** Collov, REimagineHome, AI Renovation, Roomstage; or SDXL/Flux-class inpainting with ControlNet depth conditioning on your own infra.

**Watch-outs.** Renovation renders are the highest-risk category for misleading-advertising claims — the disclosure requirements of A9 apply doubly, and structural feasibility must be caveated.

**Links.** [AI virtual staging & renovation tools 2026](https://airenovation.io/blog/best-ai-virtual-staging-tools-real-estate-2026)

---

<a name="part-b"></a>
# Part B — Exterior features

Exterior is where most tour products are weakest: they treat the outside as one hero photo. In 2026 the exterior is a full second product surface — the lot, the envelope, the light, the neighbourhood, and the risk.

## Tier 1 — Baseline exterior

### B1. Aerial stills + drone video

**What.** 4–8 aerial stills (front elevation, rear, both obliques, roof, lot context, neighbourhood context) plus a 20–40s flight: reveal-from-behind-trees, orbit, and pull-back-to-context.

**Why.** Answers the questions ground photos structurally cannot: how big is the lot, what backs onto it, how close are the neighbours, what's the roof like, is there a view.

**Plan.**
1. **Legal first**: in the US, commercial flight requires an FAA **Part 107** remote pilot certificate; check controlled airspace (LAANC authorization), altitude ceiling, and no-fly zones before booking.
2. Standardize the shot list above so every listing is comparable.
3. Shoot at the right light — 1 hour before sunset for warmth, overcast for roof/lot documentation.
4. Fly the same orbit twice (fast + slow) so you have both a hero clip and stabilized frames.
5. Deliver stills into the gallery and the flight as the tour's opening sequence.

**Stack.** DJI Mavic/Air class; LAANC via Aloft/Airmap; editing per A2.

**Watch-outs.** Privacy: neighbours' windows, pools and yards should be excluded or blurred. Some HOAs and jurisdictions restrict launch from common property.

**Links.** [FAA Part 107 / commercial drone operators](https://www.faa.gov/uas/commercial_operators)

---

### B2. Twilight conversion and sky replacement

**What.** Turning a daytime exterior into a dusk scene — deep blue gradient sky, warm interior window glow, landscape lighting — or simply replacing a flat grey sky with a credible one.

**Why.** The twilight exterior is consistently the highest-performing single image in listing marketing. AI made it a $1 operation rather than a second shoot.

**Plan.**
1. Shoot for the edit: correct exposure on the structure, windows visible, no blown highlights.
2. Apply a house style — 2–3 approved sky presets, not a random dramatic sky per listing.
3. Keep light direction physically consistent between the sky, the shadows on the building and the ground.
4. **Disclose** any material alteration per A9's pipeline; sky/exposure work is generally accepted, "adding a view that isn't there" is not.
5. Produce both the day and dusk versions — portals and print want different ones.

**Stack.** Twilight.pics, PhotoUp, AI Home Design, Luminar Neo, EditThisPic.

**Links.** [Twilight photography: shoot or fake it in 2026](https://aihomedesign.com/blog/real-estate-photography/twilight-real-estate-photography/) · [Sky replacement: when and how](https://twilight.pics/blog/sky-replacement-real-estate-photos)

---

### B3. Parcel boundary and site plan overlay

**What.** The legal lot outline drawn over an aerial image or 3D view, with dimensions, setbacks, easements, and the position of structures, driveway, well/septic, and outbuildings.

**Why.** "Where does the property end" is one of the first questions on any land, rural or large-lot listing, and it is nearly impossible to answer from photos. It also pre-empts a whole category of post-offer disputes.

**Plan.**
1. Source parcel geometry from county GIS / assessor data or a national parcel provider (Regrid, ATTOM, LightBox).
2. Reproject and overlay on the orthophoto or on the 3D exterior scene; label dimensions per side and total acreage.
3. Layer optional context: zoning, flood zone, easements, topography contours.
4. Add a clear "for illustration; not a survey" disclaimer — this is mandatory.
5. Export a shareable one-page site plan PDF.

**Stack.** Regrid/ATTOM parcel APIs, county GIS WMS/WFS feeds, Mapbox GL or Leaflet, Cesium for 3D drape.

**Watch-outs.** Public parcel geometry can be off by metres. Never let it be mistaken for a survey.

---

## Tier 2 — Professional exterior

### B4. Aerial 360 panoramas and exterior tour graph

**What.** Spherical panoramas captured *from the air* at 3–5 altitudes and positions (over the roof, over the backyard, at treetop height at the front, and at the highest point for view context), linked into the same tour graph as the interior — so the user can literally fly from the front door up over the roof and back into the kitchen.

**Why.** It gives view context that neither a ground pano nor a fixed aerial photo can, and it is the cheapest way to make an exterior *navigable* rather than static. For view properties it is the value proposition.

**Plan.**
1. Capture: drone-mounted 360 camera, or the drone's own sphere/pano mode; shoot at 3 altitudes (≈10 m, 30 m, 80 m).
2. Remove the drone/nadir artifact and patch the nadir with the ground orthophoto.
3. Author transitions: ground pano → low aerial → high aerial, with animated zoom-out rather than a hard cut.
4. Add hotspots in the aerial views for lot features, neighbourhood POIs and "view toward X".
5. Link the aerial graph to the interior graph at the entry node so it is one continuous experience.

**Stack.** CloudPano (drone + 360 integration), TeliportMe, threesixty.tours, SkyeBrowse; any A1 viewer will host the panos.

**Links.** [CloudPano drone + 360 integration](https://www.cloudpano.com/property-managers/drone-and-360-virtual-tour-integration) · [SkyeBrowse drone 360 tours](https://www.skyebrowse.com/news/posts/drone-360-tours)

---

### B5. Exterior photogrammetry model (drone orbit → 3D mesh)

**What.** A textured 3D mesh of the building and immediate site, reconstructed from a structured drone orbit — spin it, look at any elevation, inspect the roof, measure a facade.

**Why.** The exterior equivalent of the dollhouse. Essential for CRE, land, multi-building assets, insurance documentation and any renovation conversation about the envelope.

**Plan.**
1. Fly a **double-grid + orbit** mission: nadir grid at 80% overlap, two oblique orbits at 45° and 60°, 2–3 altitudes. Automate it (Litchi, DJI Pilot mission planning) so it's repeatable.
2. Add ground control or RTK if you need survey-grade scale; otherwise scale from a known dimension.
3. Reconstruct: RealityScan/RealityCapture, Agisoft Metashape, Pix4D, or cloud (DroneDeploy, SkyeBrowse videogrammetry — minutes from video).
4. Decimate + texture-bake to a web budget (<50 MB), export glTF/3D Tiles.
5. Publish with orbit controls, elevation snap buttons (N/E/S/W), and measurement on the mesh.

**Stack.** RealityScan, Metashape, Pix4D, DroneDeploy, SkyeBrowse; delivery via three.js/Cesium/Potree.

**Effort / Cost.** 20–40 min of flight; 30–120 min of processing; $50–200/property outsourced.

**Watch-outs.** Thin structures (railings, wires, fences) and reflective glass reconstruct badly — this is precisely where 3DGS (B8) wins.

**Links.** [SkyeBrowse for real estate](https://www.skyebrowse.com/news/posts/real-estate-marketing)

---

### B6. Outdoor rooms, landscape and amenity mapping

**What.** Treating the outside as rooms: patio, pool, deck, garden zones, garage/workshop, ADU, dock — each with panoramas or 3D, dimensions, materials, age, and maintenance notes, all pinned on the site plan.

**Why.** Outdoor living is often 30–50% of the perceived value in warm markets and is systematically under-documented. It's also the highest-yield place to put tags (A8): pool equipment age, irrigation zones, fence type.

**Plan.**
1. Extend the floor-plan data model outward: `OutdoorSpace` entities with polygon, type, surface, orientation and dimensions.
2. Capture one 360 per outdoor room, plus detail stills of equipment.
3. Pin each to the site plan (B3) so the mini-map covers indoor *and* outdoor.
4. Record orientation per outdoor space — it feeds the sun analysis (B10).
5. For multifamily/CRE, run the same model over amenities (pool deck, gym, dog run, rooftop, parking) as a separate tour graph.

---

### B7. Neighbourhood and location tour

**What.** A guided exploration of *where* the home is: street-level views, distance/time to named POIs, schools, transit, shopping, parks — as an interactive map layer plus optional 360 nodes at key locations (the corner, the park entrance, the station).

**Why.** People buy neighbourhoods. It is also the most common reason a remote buyer refuses to make an offer sight-unseen.

**Plan.**
1. Build a POI layer from a maps API with **isochrones** (walk/drive/transit time bands) rather than raw distances.
2. Add optional 360 captures at 3–5 neighbourhood nodes for premium listings.
3. Present as an objective data layer, not editorial description.
4. **Fair-housing guardrail (critical)**: never characterize a neighbourhood in terms of demographics, "safety", "good/bad schools", or "family-friendly". Present raw, sourced, non-editorial data (ratings from a named third party, attributed) and let the user judge. Steering by content is a violation regardless of intent.
5. Localize the POI categories to the buyer segment (schools vs nightlife vs healthcare) *only* when the user selects them — never infer.

**Stack.** Google Maps Platform / Mapbox isochrones, Walk Score API, transit GTFS feeds, GreatSchools (attributed).

**Watch-outs.** This is the single highest fair-housing-risk feature in the document. Legal review before launch, and log the exact copy shipped.

---

## Tier 3 — Advanced exterior

### B8. Gaussian splat exterior / free-flight property tour

**What.** The whole property — building, grounds, streetscape, tree canopy, setting — captured as a photorealistic radiance field the user flies around continuously in a browser. Zillow's **SkyTour** is the mass-market instance: drone footage converted with Gaussian splatting into an interactive drone-style flight on the Zillow app and site for Showcase listings.

**Why.** Exteriors are where 3DGS is decisively better than mesh photogrammetry: vegetation, thin structures, glass and sky all reconstruct convincingly. It also captures *context* — you can zoom out to read the lot, or dive to street level, in one continuous gesture.

**Plan.**
1. **Capture** a slow double orbit at two altitudes plus a low pass, 4K60, consistent exposure (lock it), no people/cars moving through frame. 4–8 minutes of footage.
2. Pose with COLMAP/GLOMAP; scenes with large sky regions need masking.
3. Train with gsplat/Postshot; segment and delete the sky, then composite a stable sky dome behind the splat.
4. **Crucially, constrain the camera**: define an allowed flight volume (a dome above the property) so users cannot fly into under-observed regions where the reconstruction is empty.
5. Compress (SOG/`.spz`) to a mobile-viable budget; provide a lower-Gaussian LOD for phones.
6. Add hotspots inside the splat that dive into the interior tour (A1/A13) at the front door.
7. Ship a video fallback (an orbit render) for unsupported devices.

**Stack.** Same as A13. Platform precedent: Zillow SkyTour, Matterport 3D Exteriors on Apartments.com, DJI's 2025 splat support, splatlabs, Real Horizons, See3D.

**Watch-outs.** Neighbouring properties end up in the reconstruction — have a privacy policy and a takedown path. Seasonal appearance is baked in; a splat captured in winter will misrepresent a summer listing.

**Links.** [Zillow launches SkyTour (GeekWire, 2025)](https://www.geekwire.com/2025/zillow-uses-drone-imagery-for-new-exterior-3d-tour-feature/) · [Designing SkyTour (Zillow engineering)](https://www.zillow.com/news/designing-an-immersive-3d-experience-for-home-exteriors-skytour/) · [Gaussian splatting vs traditional virtual tours](https://www.splatlabs.ai/blog/virtual-tours-real-estate-gaussian-splatting)

---

### B9. Geospatial context streaming (city-scale 3D around the property)

**What.** Dropping the property's own model into a streamed photorealistic 3D model of the actual city, so the viewer can pull back from the front door to the block, the district and the skyline without a seam — and see the real surrounding buildings, terrain and streets.

**Why.** For urban condos, CRE and pre-construction, *context is the product*: what will I see, what's next door, what's being built across the street. It also makes exterior tours feel infinite rather than a floating object in grey space.

**Plan.**
1. Stream a photorealistic base: Google Maps Platform **Photorealistic 3D Tiles** (2,500+ cities), consumed through CesiumJS or Cesium ion (where the tiles are included).
2. Georeference your own asset (splat, mesh, or BIM model) to real-world coordinates and clamp it to terrain.
3. Author camera bookmarks: front door → block → district → skyline, each a smooth flight.
4. Layer parcel (B3), sun (B10), risk (B11) and POI (B7) data on the same globe so they share one coordinate system.
5. Budget tile streaming cost and cache aggressively — this is a metered API.

**Stack.** [Google Photorealistic 3D Tiles](https://developers.google.com/maps/documentation/tile/3d-tiles), [CesiumJS](https://cesium.com/learn/cesiumjs-learn/cesiumjs-photorealistic-3d-tiles/), Cesium for Unreal (for the UE5 path — the pattern Palatial uses for architectural presentation), Geopipe for procedural city context, deck.gl for data layers.

**Watch-outs.** Google's 3D Tiles have terms restricting some derivative uses — read the Maps Platform terms before building a product around them. Coverage and freshness vary sharply by city.

**Links.** [Google Photorealistic 3D Tiles docs](https://developers.google.com/maps/documentation/tile/3d-tiles) · [CesiumJS + Photorealistic 3D Tiles tutorial](https://cesium.com/learn/cesiumjs-learn/cesiumjs-photorealistic-3d-tiles/) · [Palatial: Cesium + Google tiles for architecture](https://cesium.com/blog/2023/06/27/palatial-uses-google-photorealistic-3d-tiles-and-cesium-for-unreal-for-architectural-communication/)

---

### B10. Sun path, shadow study and view analysis

**What.** An interactive time-of-day/time-of-year scrubber showing exactly where sunlight falls on the lot, the patio and into each window — plus, for apartments, the simulated view out of a specific unit at a specific floor and orientation.

**Why.** "Which rooms get morning light", "is the garden in shade by 3pm in October", "will the new tower block my view" are decision-grade questions that no photo answers, and that buyers currently resolve by visiting. It is also directly monetizable in solar, gardening and renovation contexts.

**Plan.**
1. Get accurate geometry: your property model (B5/B8/BIM) plus surrounding buildings and terrain (B9) — shadows are cast *by neighbours*, so context is mandatory.
2. Compute sun position from lat/long/date/time; render real-time shadows in the 3D scene with a date + hour scrubber and a season preset row.
3. Per-window analysis: annotate each window's orientation (from the floor plan) and compute direct-sun hours per room per season; surface as a small table per room.
4. Roof/solar: query the **Google Solar API** `buildingInsights` and `dataLayers` for roof segment azimuth/pitch, annual flux, and shading from trees and neighbouring structures — present as "solar potential" with an estimated system size.
5. View simulation: render the outward view from each unit's elevation using the city context, with a "what if the approved development is built" toggle where planning data exists.
6. Cache results per property — these are expensive and static.

**Stack.** [Google Solar API](https://developers.google.com/maps/documentation/solar/concepts), [Shadowmap API](https://shadowmap.org/api) (global 3D solar/shadow data layer), SunCalc for cheap sun position math, Cesium/three.js for shadow rendering, Ladybug/Radiance for rigorous daylight analysis on the professional tier.

**Watch-outs.** Solar API coverage and imagery age vary; label results as estimates. Vegetation shading changes seasonally and models often use one canopy state.

**Links.** [Google Solar API concepts](https://developers.google.com/maps/documentation/solar/concepts) · [Shadowmap](https://shadowmap.org/) · [Shadowmap API](https://shadowmap.org/api)

---

### B11. Climate and environmental risk layers

**What.** Property-level risk scores and visualizations for flood, wildfire, hurricane wind, extreme heat — plus optional noise, air quality and seismic — shown as map layers and as a plain-language panel attached to the tour.

**Why.** It is increasingly the difference between an insurable and an uninsurable purchase, and buyers are asking. First Street scores properties 1–10 across flood, fire, wind and heat, and its data is embedded in NAR's RPR and (formerly) major portals.

**Plan.**
1. License a property-level risk dataset (First Street is the reference implementation in the US; ClimateCheck, Verisk, Munich Re for alternates; national flood agencies elsewhere).
2. Render as a layer on the same globe/map as B3/B9, with a per-hazard score card and a 30-year projection.
3. Pair every risk with a **mitigation** panel (elevation certificate, defensible space, roof class, storm shutters) so the feature is constructive rather than purely negative.
4. Add insurance-context copy and, if you have partners, a quote hand-off.
5. Make the layer opt-in and clearly attributed to the data provider.

**Watch-outs.** This is commercially sensitive: **Zillow removed climate risk scores in December 2025 after agent complaints about lost sales.** Expect internal resistance; the counter-argument is that buyers now find the data elsewhere and trust the source that showed it to them. Also: never let risk data be used in a way that correlates with protected classes in marketing targeting.

**Links.** [First Street flood model methodology](https://firststreet.org/methodology/flood) · [First Street in NAR RPR](https://blog.narrpr.com/support/what-is-first-street/) · [Zillow drops climate risk scores (TechCrunch, Dec 2025)](https://techcrunch.com/2025/12/01/zillow-drops-climate-risk-scores-after-agents-complained-of-lost-sales)

---

### B12. Virtual landscaping, facade renovation and ADU visualization

**What.** Exterior equivalents of A9/A20: repaint the siding, change the roof, replace the driveway, mature the planting, add a pool, add a garage conversion or a detached ADU — as before/after layers on the exterior photo or 3D scene.

**Why.** Curb appeal is the highest-elasticity part of perceived value, and "what could this lot become" is the entire pitch for underbuilt lots in high-value markets (where ADU legislation has made it concrete).

**Plan.**
1. Template the option set by archetype: siding/paint, roof material, front door + hardware, driveway, planting maturity, fence, pool, ADU footprint.
2. Generate options with depth/segmentation-conditioned image models so the building's geometry survives the edit.
3. For ADUs and structural adds, place a *modelled* volume into the 3D scene rather than a 2D repaint — it needs to respect setbacks (B3) and shadows (B10).
4. Attach zoning feasibility flags where you have the data ("ADU permitted in this zone, subject to X").
5. Disclose every rendering; keep the unmodified image adjacent, always.

**Stack.** EditThisPic exterior editor, AI Home Design, Collov, REimagineHome; Blender/UE for modelled additions.

**Links.** [AI home exterior editor (2026)](https://editthispic.com/edit/exterior)

---

### B13. Roof, envelope and condition documentation layer

**What.** A technical exterior capture aimed at *condition* rather than marketing: high-resolution roof orthomosaic, facade tiles, thermal imagery for insulation/moisture, and a defect annotation layer — attached to the same 3D model as the marketing tour.

**Why.** Serves inspection, insurance underwriting, claims, lender review and CRE due diligence off the same flight that produced the marketing assets. This is the highest-margin adjacent product a tour operator can sell.

**Plan.**
1. Add a nadir grid at higher overlap and lower altitude to the B5 mission; add a thermal payload where relevant (best flown at dusk).
2. Produce an orthomosaic with GSD ≤1 cm/px and a per-facade tile set.
3. Annotate defects (missing shingles, flashing, cracks, ponding) as geo-anchored issues with severity and photo evidence.
4. Export an inspection PDF plus a machine-readable defect list; version it so the next capture is a *comparison*.
5. Keep this layer **private by default** — it must not leak into the marketing tour.

**Stack.** DroneDeploy, Pix4Dinspect, Scopito, IMGING; thermal via DJI Mavic 3T/M30T.

**Watch-outs.** In most jurisdictions this is not a substitute for a licensed inspection — say so. Sharing defect data with a buyer creates disclosure obligations for the seller.

---

### B14. Pre-construction exterior: massing, phasing and time-of-day

**What.** For off-plan sales: the future building placed in real geospatial context, with construction phasing timeline, seasonal and day/night lighting, landscape maturity over 5 years, and view-from-unit simulation at each floor.

**Why.** Off-plan buyers are purchasing a promise; the exterior context (what will I look at, what will the courtyard feel like in winter, when does phase 2 block my view) is the substance of the promise and the source of most later disputes.

**Plan.**
1. Import the architectural model (Revit → Datasmith → UE5/Twinmotion), georeferenced with Cesium for Unreal onto the real city (B9).
2. Author phases as scene states with a timeline scrubber; include construction staging so buyers know what the next 24 months look like from their window.
3. Author lighting states (dawn/noon/dusk/night, summer/winter) tied to the real sun path (B10).
4. Add vegetation growth states — 5-year landscape maturity is a standard sales asset and a standard source of complaint when omitted.
5. Deliver via pixel streaming for the sales gallery, pre-rendered flythroughs for the web, and VR (A14) for the gallery headset.
6. Watermark everything "artist's impression" with a materials/change disclaimer.

**Stack.** Unreal Engine 5 + Cesium for Unreal + Datasmith; Twinmotion for the faster/cheaper path; Palatial for a hosted version of this pattern.

**Links.** [Cesium for Unreal + Google tiles case study](https://cesium.com/blog/2023/06/27/palatial-uses-google-photorealistic-3d-tiles-and-cesium-for-unreal-for-architectural-communication/) · [Twinmotion](https://www.twinmotion.com/)

---

### B15. Construction progress capture (as-built digital twin over time)

**What.** Weekly 360 walkthroughs of an in-progress building, auto-registered to the floor plan and BIM, producing a time-sliderable record: "show me this wall on March 3rd, before the drywall".

**Why.** For developers, it is progress verification, draw-request evidence, and dispute resolution; for buyers of off-plan units it is a trust-building progress feed; at handover it becomes the operations twin. OpenSpace reports ~69B sq ft captured across 100K+ projects.

**Plan.**
1. Capture: hard-hat-mounted 360 camera, walk the floor, no setup — the AI maps frames to the plan automatically.
2. Register captures to floor plan and BIM; run automated comparison to detect progress and deviations.
3. Expose two views: an internal one (progress %, deviations, issues) and a buyer-facing one (curated, monthly, per building).
4. At completion, convert the final capture into the marketing twin (A5) and the operations twin (B16 below).
5. Retain everything — as-built imagery is the cheapest insurance against a construction dispute.

**Stack.** OpenSpace (Capture/Field/BIM+/Track), StructionSite, HoloBuilder, Cupix; 360 camera per Part D.

**Links.** [OpenSpace Capture](https://www.openspace.ai/products/capture/) · [Reality capture with BIM](https://www.openspace.ai/blog/how-does-reality-capture-work-with-bim/)

---

<a name="part-c"></a>
# Part C — Cross-cutting features

These are not "nice to have". A tour without C1–C4 is a demo, not a product.

### C1. Analytics: heatmaps, dwell time, drop-off, intent scoring

**What.** Per-session instrumentation of the tour: which rooms were visited, in what order, dwell time per node, where the camera lingered (view heatmap), hotspot opens, measurement usage, floor-plan clicks, replays, device/XR mode, and drop-off point.

**Why.** It converts the tour from a marketing cost into a *signal source*. Knowing a lead spent 4 minutes in the primary suite and opened the HOA doc tells the agent more than any email open rate; it also tells the seller what to fix.

**Plan.**
1. Define an event schema up front: `tour_open`, `node_enter/exit`, `view_dwell` (sampled yaw/pitch), `hotspot_open`, `measure_used`, `plan_click`, `xr_enter`, `lead_submit`, `tour_complete`. Include `property_id`, `session_id`, `channel`, `referrer`.
2. Aggregate into per-property reports: room ranking by dwell, drop-off funnel by node index, heatmap image per room.
3. Compute an **intent score** per session (weighted dwell + return visits + tag opens + measurement + gated-content unlock) and push it to the CRM as a lead attribute.
4. Give sellers a weekly report — this is the retention feature for listing agents.
5. Instrument your *own* funnel to replace vendor statistics with your numbers: tour-open rate per listing view, completion rate, tour→inquiry conversion, and days-on-market with/without tour, matched on price band and geography.

**Stack.** Your own event pipeline (Segment/RudderStack → warehouse) beats platform-native analytics; Matterport/CloudPano/3DVista all expose some analytics but rarely at the session grain you need.

**Watch-outs.** Behavioural data on housing prospects is regulated-adjacent; never use it to differentiate service by inferred protected class, and get consent for tracking where GDPR/CPRA apply.

**Links.** [Virtual tour analytics and CRM integration overview](https://panoee.com/virtual-tour-real-estate/)

---

### C2. Lead capture, gating and CRM integration

**What.** In-tour forms, gated premium layers (dollhouse, measurements, floor plan download, price history), "schedule a showing" and "apply now" CTAs, all writing to the CRM with the analytics payload attached.

**Plan.**
1. Choose the gate carefully: gate *depth* (measurements, downloads, unit availability), never the basic tour — gating the tour itself kills top-of-funnel.
2. Trigger the ask at engagement thresholds (e.g. after 90 seconds or the third room), not on load.
3. Two-field forms (name + email/phone) with progressive profiling later.
4. Push to CRM with full session context; fire an instant response (AI concierge A15 or agent alert) — response latency dominates conversion.
5. Branch by channel: portal-sourced traffic often can't be gated at all under portal rules; your own site can.

**Stack.** CloudPano/Kuula/3DVista native lead forms; HubSpot/Follow Up Boss/Salesforce/Knock/Entrata for CRM; webhook-first design so you're not locked to one.

---

### C3. Distribution: MLS, IDX, portals, SEO

**What.** Getting the tour in front of people: MLS virtual tour fields, IDX-fed listing pages, portal syndication, social, email, and organic search.

**Plan.**
1. **Always produce two URLs**: *unbranded* (no logos, agent name, phone, CTAs, or external links) for the MLS/IDX/syndication field, and *branded* for your own site and social. Many MLSs provide exactly two fields for this reason, and rules vary by market — check the local MLS handbook, not a national summary.
2. Provide **embed code**, not just a link, wherever the destination supports it; embed is more reliable than URL-sniffing for playback on syndication targets.
3. Keep a portable format (A1 panorama tour, or a video render) as the fallback for destinations that reject your primary format — the Zillow/Matterport break of October 2025 is the cautionary tale.
4. **SEO**: give each tour a crawlable HTML landing page (not a bare iframe) with the address, structured data, transcript/text tour (from A12), and image alt text. Use schema.org `Residence`/`Accommodation` + `VideoObject` for the walkthrough, and `Place`/`geo` for location.
5. Optimize the share card (OG image = twilight hero, OG video = the 30s cutdown).
6. Track per-channel attribution in C1.

**Watch-outs.** Branded content in an unbranded field is the most common MLS violation and can pull the listing. Automate the check.

**Links.** [Kuula: how to add virtual tours to MLS listings 2026](https://blog.kuula.co/mls) · [MLS syndication guide (CloudPano)](https://www.cloudpano.com/blog/mls-guide-how-to-ensure-your-virtual-tour-appears-on-all-major-listing-sites) · [Branded vs unbranded tour requirements](https://gotofsbo.com/virtual-tour-requirements-branding.php) · [MLS-compatible unbranded tours guide](https://virtualtoureasy.com/articles/mls-compatible-virtual-tours-unbranded-listing-video-guide-2026/)

---

### C4. Compliance and trust layer

A single checklist to run before any tour ships:

| Area | Requirement | Where it bites |
|---|---|---|
| **Digital alteration disclosure** | Label every virtually staged/renovated/materially altered image; keep originals | CA **AB 723** (Jan 2026) makes non-disclosure a misdemeanour; NAR/state commissions require disclosure |
| **Accessibility** | WCAG 2.1 AA on the tour and its page; keyboard + screen reader + captions | ADA + Fair Housing Act (§A12) |
| **Fair housing** | No demographic/steering language in copy, AI concierge, neighbourhood layer, or ad targeting | §A15, §B7 — highest-risk surfaces |
| **Measurement** | State the standard (ANSI Z765 / RESO RMS / IPMS) and tolerance next to every area figure | §A3, §A6 |
| **Drone** | Part 107 certificate, airspace authorization, altitude limits, privacy of neighbours | §B1 |
| **Privacy** | Blur faces, licence plates, neighbours' interiors, personal documents, family photos, security devices | Every capture; run automated detection |
| **Security** | Don't publish alarm panels, safe locations, key hiding spots, or occupancy patterns | Occupied listings |
| **Data** | Consent for tracking, retention policy, DSAR path (GDPR/CPRA) | §C1 |
| **IP** | Music licensing, artwork in frame, model releases for people | §A2 |

Build these as **automated gates in the publish pipeline**, not as a training document — a manual checklist will be skipped.

---

### C5. Data model and interoperability

**What.** The schema underneath everything, so a property's assets are portable across viewers, portals and future formats.

**Plan.**
1. One `Property` root with: identifiers (APN, MLS #, address, geo), `Structure[]`, `Level[]`, `Room[]` (polygon, area, orientation, ceiling height), `OutdoorSpace[]`, `CapturePoint[]` (pose in a single coordinate frame), `Asset[]` (pano, mesh, splat, video, plan, photo — each with format, LOD, size, provenance), `Tag[]`, `Alteration[]` (what was AI-modified, by which tool, when).
2. Adopt standards where they exist: **RESO Data Dictionary / Web API** for listing data, **IFC** for BIM, **glTF/GLB** for meshes, **3D Tiles** for streamed geometry, **`.spz`/SOG** for splats, **USDZ** for iOS AR, equirectangular JPEG/AVIF for panos.
3. Keep every capture in one metric coordinate frame so plan, tour, splat, measurements and AR all agree.
4. Version assets — reshoots and re-renders are constant; the tour URL must be stable while the assets behind it change.
5. Store provenance for every AI transformation (this is what makes C4 disclosure automatic).

---

### C6. Performance and delivery budget

**What.** The engineering that decides whether the tour is used at all: most tours are opened on a phone, on cellular, with a 3-second patience window.

**Plan.**
1. Set hard budgets: **first meaningful pano <1.5s on 4G**, total initial payload <3 MB, decoded memory <150 MB on mobile, 60 FPS interaction.
2. Tile and pyramid every pano (multi-res cube maps); load level 0 immediately, refine progressively. Prefetch only the adjacent nodes in the graph.
3. Use AVIF/WebP for panos with JPEG fallback; Draco/Meshopt for meshes; SOG/`.spz` for splats; per-device LOD selection.
4. Serve from a CDN with long-lived immutable asset URLs; keep the manifest small and separately cacheable.
5. Feature-detect: WebGPU → WebGL2 → static panorama → video fallback. Never white-screen.
6. Measure real-user metrics per device class and treat tour-open abandonment as a P1 bug.

---

<a name="part-d"></a>
# Part D — Capture hardware and pipeline reference

### D1. Camera selection

| Gear | Approx. price | Produces | Best for | Notes |
|---|---|---|---|---|
| iPhone Pro (LiDAR) | — | Panos, RoomPlan floor plans, splat source video, AR | Volume rentals, self-capture, AR features | ~1–2% measurement accuracy; free floor plans via RoomPlan |
| Insta360 X5 | ~$549 | 8K video, 72MP stills, 360 panos | Best all-round for agents shooting several listings/week | 1/1.28" sensors handle mixed light; IP68 |
| Ricoh Theta X | ~$800 | 60MP 360 stills, HDR bracketing | Phone-free workflow, fast JPEG turnaround | Touchscreen, no phone tethering |
| Ricoh Theta Z1 | ~$1,000 | 360 stills, RAW/DNG | Best low-light of the non-LiDAR cameras | Dual 1" BSI sensors |
| Matterport Pro3 | ~$5,995 | LiDAR + HDR → true twin, point cloud, BIM export | Measurement-grade twins, CRE, insurance | ~20mm accuracy at 10m; deepest native platform integration |
| iGUIDE (Planix) | subscription/kit | LiDAR floor plans + tour | Where the *floor plan* is the deliverable | ≤0.5% distance, ~1% area uncertainty; ANSI Z765 / RESO RMS |
| Leica BLK360 / NavVis | $$$$ | Survey-grade point cloud | Large CRE floorplates, as-built | Overkill for residential |
| DJI Mavic 3 / Air 3 | $1–3k | Aerial stills, video, photogrammetry, splat source | All exterior features | Part 107 required commercially |
| DJI Mavic 3T / M30T | $$$$ | Thermal + visual | Envelope/condition layer (B13) | Dusk flights for thermal |

Note: Theta Z1/X/SC2, Insta360 X5 and X3 can all feed the Matterport platform; only the Pro3 captures native LiDAR spatial data within it.

### D2. Standard capture SOP (single-visit, all features)

A single 90-minute visit that feeds every feature in this document:

1. **Pre-flight** (off-site): confirm airspace + Part 107, pull parcel geometry, confirm occupancy and pets, send prep checklist to seller.
2. **Prep** (10 min): all lights on, blinds to a consistent position, toilet seats down, cars moved, personal photos/documents flagged for blurring, pets secured.
3. **Stills** (25 min): bracketed HDR per room + exteriors, tripod, vertical lines vertical.
4. **360 / twin capture** (25 min): scan points every 6–8 ft, doorways covered from both sides, operator out of frame.
5. **Splat video** (5 min interior + 5 min exterior): slow continuous walk/orbit, locked exposure, loop closure.
6. **Drone** (20 min): standard shot list (B1), double-grid + orbit mission (B5), aerial panos at 3 altitudes (B4), nadir roof grid (B13).
7. **Notes**: room names, ceiling heights, ages of systems, outdoor space types, orientation — this is the metadata that powers tags, text tours and sun analysis.
8. **Upload + QA**: automated checks for horizon tilt, blown highlights, missing rooms, faces/plates needing blur, and registration drift before anything reaches a customer.

### D3. Processing pipeline (reference)

```
capture ──▶ ingest & validate ──▶ enhance (HDR merge, lens, colour)
                                   │
                                   ├─▶ pano tiling ─────────▶ 360 tour (A1)
                                   ├─▶ SfM (COLMAP/GLOMAP) ─▶ 3DGS train (gsplat)
                                   │                          └─▶ SOG compress ─▶ splat tour (A13/B8)
                                   ├─▶ LiDAR registration ──▶ mesh + dollhouse (A5) ─▶ floor plan (A3) ─▶ measurements (A6)
                                   ├─▶ photogrammetry ──────▶ exterior mesh (B5) ─▶ ortho + defects (B13)
                                   └─▶ AI layer ────────────▶ staging/defurnish (A9), twilight (B2), descriptions
                                                              └─▶ alteration provenance ─▶ disclosure (C4)
                          all ──▶ asset registry (C5) ──▶ CDN (C6) ──▶ viewer ──▶ analytics (C1) ──▶ CRM (C2)
```

---

<a name="part-e"></a>
# Part E — Build vs buy, and a phased roadmap

### E1. Build vs buy, per feature

| Feature group | Recommendation | Reasoning |
|---|---|---|
| Panorama tour + viewer (A1, A7, A8) | **Build** (on Pannellum/PSV/Marzipano) | Cheap, gives you the analytics grain and the data model; vendor viewers cap you |
| LiDAR twin + dollhouse (A5, A6) | **Buy** | The capture hardware + processing pipeline is a company in itself |
| Floor plans (A3) | **Buy** service, **own** the JSON | Drafting is commoditized; the structured output is strategic |
| AI staging / twilight (A9, B2) | **Buy** per-image API | Fast-moving model market; don't own model weights |
| Gaussian splats (A13, B8) | **Build** on open source | Nerfstudio/gsplat are mature; the differentiation is capture SOP + compression + viewer |
| Analytics / CRM (C1, C2) | **Build** | This is your product's actual moat |
| Configurator (A18, B14) | **Buy/partner** unless it's your core business | UE5 pipelines need a dedicated team |
| Self-guided access (A17) | **Buy** | Hardware + fraud + insurance; not a software problem |
| Geospatial/sun/risk (B9–B11) | **Buy APIs, build the layer** | Data licensing, thin integration |

### E2. Phased roadmap

**Phase 0 — Foundations (weeks 1–4).** Data model (C5), asset registry, event schema (C1), CDN + performance budgets (C6), compliance gates in the publish pipeline (C4). Nothing user-facing. Skipping this phase is the most common and most expensive mistake in this domain.

**Phase 1 — Baseline product (weeks 5–12).** A1 panorama tour + A7 mini-map + A8 tags + A3 floor plan + A2 video/reels + A4 photo pipeline + B1 aerials + B2 twilight + C3 branded/unbranded distribution + C2 lead capture. *Exit criterion:* a listing can go from capture to two published URLs in under 24 hours, unattended.

**Phase 2 — Professional (weeks 13–26).** A5 twin + A6 measurement + A9 staging/defurnish with disclosure + A10 guided path + A12 accessibility + B3 parcel + B4 aerial 360 + B5 exterior mesh + B7 neighbourhood (with legal review) + C1 full analytics with intent scoring. *Exit criterion:* measurable lift in tour→inquiry conversion, on your own instrumentation.

**Phase 3 — Advanced (weeks 27–52).** A13 + B8 Gaussian splats (start with exterior — better ROI and lower risk than interior), A15 AI concierge, A16 live tours, B10 sun/shadow, B11 risk layers, A14 WebXR. *Exit criterion:* one differentiating feature buyers name unprompted.

**Phase 4 — Vertical depth (year 2).** A11 multifamily inventory, A17 self-guided touring, A18/B14 configurators and pre-construction, B13 condition layer, B15 construction progress, B9 city-scale context, A19/A20/B12 AR and renovation.

### E3. KPIs to hold each phase to

| Metric | Definition | Target direction |
|---|---|---|
| Tour open rate | tour opens ÷ listing page views | >35% |
| Completion rate | sessions reaching ≥80% of nodes | >25% |
| Median dwell | time in tour per session | >2 min |
| Tour → inquiry | leads ÷ tour sessions | Track lift vs no-tour control |
| Time to publish | capture → live URL | <24h, then <4h |
| p75 first-pano time | on 4G mobile | <1.5s |
| Cost per property | all-in capture + processing | Trending down per phase |
| Compliance defect rate | publishes failing an automated gate | →0 |

Run tour vs no-tour as a **matched comparison** (price band, geography, season, agent) rather than trusting industry averages.

---

<a name="part-f"></a>
# Part F — Source list

**Platforms & market**
- [Matterport — best virtual tour software 2026](https://matterport.com/blog/best-virtual-tour-software-for-real-estate)
- [Matterport — Cortex AI](https://matterport.com/cortex-ai) · [digital twin viewer update](https://matterport.com/blog/refreshing-the-way-to-explore-your-digital-twins) · [publishing tours anywhere](https://matterport.com/blog/matterport-customers-remain-free-to-publish-their-3d-tours-anywhere)
- [HousingWire — Matterport AI Defurnish & descriptions](https://www.housingwire.com/articles/matterport-ai-defurnish-photos-property-descriptions/) · [Zillow & CoStar spar over Matterport tours](https://www.housingwire.com/articles/zillow-matterport-3d-tours/)
- [Inman — Zillow removes Matterport 3D tours (Oct 2025)](https://www.inman.com/2025/10/20/zillow-removes-matterport-3d-home-tours-from-its-sites/)
- [Zillow — SkyTour design deep dive](https://www.zillow.com/news/designing-an-immersive-3d-experience-for-home-exteriors-skytour/) · [GeekWire — Zillow launches SkyTour](https://www.geekwire.com/2025/zillow-uses-drone-imagery-for-new-exterior-3d-tour-feature/)
- [HousingWire — best real estate virtual tour software 2026](https://www.housingwire.com/articles/virtual-tour-software/)
- [Panoee — virtual tour cost guide 2026](https://panoee.com/virtual-tour-cost/) · [THE FUTURE 3D — Matterport pricing 2026](https://www.thefuture3d.com/blog/matterport-pricing-guide-2026/)

**Radiance fields / 3DGS**
- [Nerfstudio splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) · [gsplat library (arXiv)](https://arxiv.org/pdf/2409.06765) · [SOG compression in gsplat](https://radiancefields.com/nerfstudio-adds-compression-to-gsplat)
- [3DGS guide: services, use cases, web viewers (2026)](https://www.utsubo.com/blog/gaussian-splatting-guide) · [Splat Labs — virtual tours & Gaussian splatting](https://www.splatlabs.ai/blog/virtual-tours-real-estate-gaussian-splatting) · [Real Horizons — 3DGS for real estate](https://realhorizons.ai/blog/gaussian-splatting-for-real-estate/)

**Geospatial, sun, risk**
- [Google Photorealistic 3D Tiles](https://developers.google.com/maps/documentation/tile/3d-tiles) · [CesiumJS tutorial](https://cesium.com/learn/cesiumjs-learn/cesiumjs-photorealistic-3d-tiles/) · [Palatial case study](https://cesium.com/blog/2023/06/27/palatial-uses-google-photorealistic-3d-tiles-and-cesium-for-unreal-for-architectural-communication/)
- [Google Solar API concepts](https://developers.google.com/maps/documentation/solar/concepts) · [Shadowmap](https://shadowmap.org/) · [Shadowmap API](https://shadowmap.org/api)
- [First Street flood methodology](https://firststreet.org/methodology/flood) · [First Street in NAR RPR](https://blog.narrpr.com/support/what-is-first-street/) · [TechCrunch — Zillow drops climate risk scores](https://techcrunch.com/2025/12/01/zillow-drops-climate-risk-scores-after-agents-complained-of-lost-sales)

**Capture, floor plans, AR**
- [iGUIDE measurement & drafting standards](https://help.youriguide.com/hc/en-us/articles/27645625048210-iGUIDE-Measurements-and-Drafting-Standards) · [iGUIDE floor plan vs 3D tour](https://goiguide.com/blogs/floor-plan-vs-3d-virtual-tour)
- [Matterport — best 360 cameras for real estate](https://matterport.com/blog/best-360-cameras-real-estate) · [Insta360 — best 360 camera for real estate 2026](https://www.insta360.com/blog/enterprise/360-camera-for-real-estate.html)
- [Apple RoomPlan](https://developer.apple.com/documentation/roomplan) · [RoomPlan use cases](https://volpis.com/blog/top-use-cases-for-apps-utilizing-apple-roomplan/) · [AR room planning technologies](https://www.netguru.com/blog/augmented-reality-room-planning-technologies)
- [SkyeBrowse drone 360 tours](https://www.skyebrowse.com/news/posts/drone-360-tours) · [CloudPano drone + 360](https://www.cloudpano.com/property-managers/drone-and-360-virtual-tour-integration) · [FAA Part 107](https://www.faa.gov/uas/commercial_operators)

**Viewers & engines**
- [Pannellum](https://pannellum.org/) · [Photo Sphere Viewer](https://photo-sphere-viewer.js.org/) · [Marzipano](https://www.marzipano.net/) · [Open-source 360 libraries 2026](https://portalzine.de/open-source-virtual-tour-360-panorama-libraries-in-javascript-2026/)
- [W3C WebXR Device API](https://www.w3.org/TR/webxr/) · [WebXR browser support 2026](https://www.testmuai.com/learning-hub/webxr-compatible-browsers/) · [Apple Vision Pro real estate guide](https://r2u.io/en/blog/apple-vision-pro-real-estate-guide/)
- [Unreal Engine for architecture](https://www.unrealengine.com/uses/architecture) · [Twinmotion](https://www.twinmotion.com/) · [Twinmotion 2026.1 release](https://www.cgchannel.com/2026/04/epic-games-releases-twinmotion-2026-1/) · [UE5.7 Pixel Streaming 2 + React](https://forums.unrealengine.com/t/ue5-7-pixel-streaming-2-react-frontend-ui-integration-plugin-dev/2706135)

**AI staging, video, concierge**
- [MLS virtual staging rules & disclosure 2026](https://www.roomstage.ai/mls-virtual-staging-rules) · [Virtual staging tool comparison](https://collov.ai/blog/choosing-ai-virtual-staging-for-real-estate-2026-comparison) · [AI staging & renovation tools 2026](https://airenovation.io/blog/best-ai-virtual-staging-tools-real-estate-2026)
- [Twilight photography 2026](https://aihomedesign.com/blog/real-estate-photography/twilight-real-estate-photography/) · [Sky replacement guide](https://twilight.pics/blog/sky-replacement-real-estate-photos) · [Photo editing trends 2026](https://www.photoup.net/learn/new-real-estate-photo-editing-trends) · [AI exterior editor](https://editthispic.com/edit/exterior)
- [HeyGen — AI video tools for real estate](https://www.heygen.com/blog/best-ai-video-tools-real-estate) · [multilingual listing video makers](https://www.heygen.com/blog/best-ai-multilingual-property-listing-video-maker)
- [Crescendo — conversational AI for real estate](https://www.crescendo.ai/blog/conversational-ai-for-real-estate) · [Perspective AI — voice agents compared](https://getperspective.ai/blog/ai-voice-agents-for-real-estate-in-2026-7-options-compared-by-conversation-depth)

**Leasing, self-tours, construction**
- [Realync live video tours](https://www.realync.com/platform-live-video/) · [Realync 360 tours for apartments](https://www.realync.com/3d-virtual-tours/) · [Realync WCAG 2.1 compliance](https://www.prweb.com/releases/realync-announces-wcag-2-1-compliance-enabling-web-accessibility-in-multifamily-831479928.html)
- [Rently — choosing self-touring tech](https://use.rently.com/blog/choosing-the-right-self-touring-technology/) · [SmartRent locks](https://smartrent.com/hardware/locks-lock-boxes/) · [Gatewise — self-guided tours & smart locks](https://gatewise.com/blog/self-guided-tours-smart-unit-locks-security-leasing-playbook)
- [OpenSpace Capture](https://www.openspace.ai/products/capture/) · [OpenSpace + BIM](https://www.openspace.ai/blog/how-does-reality-capture-work-with-bim/)

**Compliance & distribution**
- [accessiBe — ADA compliance for real estate 2026](https://accessibe.com/blog/knowledgebase/ada-compliance-for-real-estate) · [Fair housing + ADA in digital apartment marketing](https://www.marketapts.com/blog/staying-compliant-housing-ada-laws-digital-apartment-marketing/)
- [Kuula — virtual tours on MLS](https://blog.kuula.co/mls) · [CloudPano — MLS syndication](https://www.cloudpano.com/blog/mls-guide-how-to-ensure-your-virtual-tour-appears-on-all-major-listing-sites) · [Branded vs unbranded requirements](https://gotofsbo.com/virtual-tour-requirements-branding.php)

---

*Prepared as a research and planning document. Pricing, product capabilities and regulatory details change quickly in this space — re-verify vendor pricing, MLS rules and state disclosure law before committing to a build.*
