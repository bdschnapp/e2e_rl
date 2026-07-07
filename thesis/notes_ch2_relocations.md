# Passages removed from Ch2 (Literature Review) for relocation

The Ch2 critic pass (2026-07) found that §2.1 mixed genuine literature review with
problem-setup and model-derivation prose that belongs in other chapters. Those passages
were trimmed from Ch2 and are preserved here verbatim so they can be adapted into the
chapters noted. None of this is lost; it is staged for reuse.

Spelling note: convert to Canadian (`-our`/`-re` + `-ize`, "tire") when importing — see
memory `canadian-spelling-thesis`.

---

## → Ch1 (Introduction): Operational context / success criteria

Was §2.1.2 "Operational Context" (uncited — pure motivation, not review). Ch1's
Motivation already names the distribution-yard/loading-dock setting; the *second*
sentence (the success criteria) is the part worth folding into Ch1's Problem Definition.

> The practical setting for this work is the confined, low-speed movement of trailers in
> distribution yards and terminals, where reversing a trailer toward a dock is a routine
> operation and jackknifing is both a safety hazard and an operational failure. Precise
> trailer placement, hitch stability, and reliable reverse manoeuvring are therefore the
> capabilities a controller must deliver, and they define the metrics against which every
> method is ultimately judged.

Also removed (duplicated Ch1 scope): the §2.1 intro clause
> This thesis is concerned primarily with the control stage, and with the perception that
> feeds it, for a vehicle whose dynamics make that stage unusually demanding: a tractor
> towing a trailer through a passive hitch.

---

## → Ch3 (Modeling, Simulation, and Problem Formulation): vehicle-model derivation

Was the first half of §2.1.1 "Kinematics and the Origin of Instability". This is model
derivation, not a survey. Ch3 already derives the bicycle model and cites
`rajamani2011vehicle` / `pacejka2012tyre`; use the framing below to strengthen Ch3's
statement of the non-holonomic structure and the articulation degree of freedom if it is
not already explicit there.

> The tractor alone is well described by the single-track "bicycle" model, which captures
> the dominant lateral and yaw dynamics of a road vehicle while reducing each axle to a
> single equivalent wheel [rajamani2011vehicle]. In this setting, lateral tire forces are
> often modeled using the Pacejka tire formulation [pacejka2012tyre], or, for
> control-oriented analysis, by a linearized approximation with constant cornering
> stiffness. As with any vehicle with Ackermann steering, the tractor is subject to
> non-holonomic rolling constraints: its wheels impose a no-lateral-slip condition, so the
> vehicle cannot instantaneously translate sideways. The trailer extends the constrained
> system by adding an articulation angle between the tractor and trailer and a further
> rolling constraint at the trailer axle. The central difficulty therefore arises not from
> non-holonomy alone, but from the additional internal configuration variable introduced by
> the hitch. This extra degree of freedom creates the articulation dynamics that underlie
> the instability addressed throughout this thesis.

Note: the instability *findings* themselves (open-loop instability [fancher2007directional]
and non-minimum-phase reversing [altafini2001reversing]) were KEPT in Ch2 as genuine
surveyed prior work; only the model-derivation exposition moved here.
