# SSLI formant integration contract

Karaokeizer does not bundle Super Synth Lab Instrument (SSLI). The
`karaokeizer-prepare-ssli` command applies the reviewed scheduler patch to the
pinned upstream commit, builds a self-contained distribution, records a
hash-bound capability declaration, and restores the source checkout. Generate
the distribution's exact allowlist with `karaokeizer-manifest-ssli`.

## Required API

The embedded distribution must expose these functions on `SynthLab.formant`:

```javascript
scheduleNotes(notes, instrumentId, scheduleId)
cancelSchedule(scheduleId, instrumentId)
```

Each note contains `midi`, `velocity`, `startTime`, and `endTime`; times are
absolute seconds on SSLI's `AudioContext` clock. A batch is bounded to 10,000
notes and two hours, validated on both threads, and sorted only once.
The integration retains up to 40,000 lifecycle telemetry records, enough to
attest a maximum batch's scheduled, started, released, and cancelled states.
The implementation must enqueue the events in SSLI's formant AudioWorklet and
render them with the selected formant instrument. It must not translate them
into main-thread `setTimeout` calls or a generic oscillator substitute.

`cancelSchedule` must discard every pending event with the supplied identifier
and release any voice already started by that schedule. Karaokeizer also calls
`allNotesOff` when pausing, seeking, or changing tracks.

## Why the entire remaining track is scheduled

When playback starts, seeks, or changes rate, Karaokeizer maps all remaining
note events to the AudioContext clock in one operation. Browser animation frames
may stop in a background tab; the AudioWorklet clock does not. Rebuilding the
schedule at each transport discontinuity preserves timing without a fragile
two-second lookahead loop.

The manifest binds the pinned upstream commit, scheduler-patch hash, API version,
and built index hash. The player repeats the API check at runtime. Browser QA
uses that real patched distribution and requires scheduled/start/release/cancel
worklet events, render-quantum timing, and nonzero formant output energy.
