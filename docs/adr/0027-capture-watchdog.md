# 0027. Capture recovery: native detection, a Python watchdog

Date: 2026-10-22
Status: Accepted

## Context

A running Jarvis stopped reacting to "Hey Jarvis". Its log shows VAD speech
events until 15:19 and then nothing for more than 18 minutes, while the
process stayed alive (proactive jobs kept running). A fresh process opening
the same default microphone ("Microphone Array (AMD Audio Device)", WASAPI
through miniaudio) got audio at once, and restarting Jarvis fixed it. The
machine had most likely hung or suspended for about 13 minutes around then
(another process logged a 767 s stall), and the network dropped in the same
window.

Reading miniaudio 0.11.22 explains the silence. In shared-mode capture the
WASAPI read waits at most `MA_WASAPI_WAIT_TIMEOUT_MILLISECONDS` (5 s) for
the capture event. When the wait times out, the read returns an error, the
device's worker loop ends and the device is stopped. miniaudio reports that
only through `ma_device_config.notificationCallback`, which Jarvis did not
register. An invalidated endpoint (`AUDCLNT_E_DEVICE_INVALIDATED`) ends the
same way. Nothing restarted the device, so the engine ran on with an empty
ring buffer.

The same reading answers the default-device question. A device opened with
a null id gets `allowCaptureAutoStreamRouting`, and the backend's
`IMMNotificationClient::OnDefaultDeviceChanged` stops the stream, reroutes
it to the new default and starts it again, then fires a `rerouted`
notification. Plugging in a headset is therefore followed already; the
risk is a reroute or a detached device that never comes back, which looks
exactly like the stall above.

## Decision

**Native code detects and offers the mechanism.** `MiniaudioMicrophone`
registers a notification callback and forwards `stopped`, `rerouted`,
`interruption_began` and `interruption_ended` (not `started`, its own
doing, nor `unlocked`, which only exists on the Web Audio backend). A
`closing` flag drops the `stopped` notification that closing the device
fires. `AudioEngine` turns the notifications into `DeviceStopped`,
`DeviceRerouted`, `DeviceInterrupted` and `DeviceResumed` events, counts
every delivered sample in a lock-free `samplesReceived()`, and offers
`restartCapture()`. That call closes the source, resets the pipeline (VAD,
segmenter, wake word, pre-roll; the listening flag survives) and opens the
current default device again. It is serialised with `start()` and `stop()`
by a lifecycle mutex.

**Python decides.** `jarvis.audio.watchdog.CaptureWatchdog` runs next to
the event poller (`AudioEngine(recovery=CaptureRecovery(...))`) and reads
the sample counter every 250 ms:

- A counter that stays flat for 2.5 s is a stall. Silence still delivers a
  frame every device period (about 10 ms), so a quiet room never trips it.
- A `DeviceStopped` or `DeviceInterrupted` event is a stall unless frames
  resume within 0.5 s. miniaudio stops the device itself while rerouting.
- A stall logs `audio.capture.stalled` (reason, silence, counter, current
  default microphone) and restarts the capture at once. Further attempts
  back off: 1, 2, 4, 8, 16 s, then every 30 s for as long as the outage
  lasts. A restart that delivers no frames counts as a failed attempt.
- After three failed attempts the user gets one system notification,
  "Microphone stopped; trying again". When frames return,
  `audio.capture.recovered` is logged, and if the user was told, a second
  notification says the microphone works again.
- A check that comes more than 2.5 s after the previous one means the
  watchdog itself was not running (a suspended machine, a stalled process).
  The silence is then measured again from that moment, so a resume is not
  held against the microphone. A stream that really died shows up 2.5 s
  later.

The facade resets its own speech tracking on a restart. A capture cut off
by the gap times out at `max_utterance_seconds`, as a capture with no end
of speech always has.

## Alternatives considered

- **Restart inside the notification callback.** miniaudio forbids
  `ma_device_uninit` and `ma_device_start` from the device's own callbacks
  (they wait for the thread that is calling them). A stall without any
  notification needs a timer anyway.
- **A native watchdog thread.** It would detect a stall equally cheaply.
  But backoff, the one notification and log context are policy, and they
  use the notification path, structlog and settings, which all live in
  Python. A native thread would also need a fake clock seam to be tested
  the way the Python class is. The project's split is "Python conducts,
  C++ does the hot path"; a 250 ms timer is not a hot path.
- **Watch `samplesProcessed()`.** That clock advances on the worker thread.
  A worker slowed by a loaded machine (inference on a shared GPU) would
  look like a dead microphone. `samplesReceived()` is counted in the device
  callback.
- **Open the device by id instead of the default.** Restarting would then
  stick to a device the user may have unplugged. With a null id, a restart
  goes to whatever is the default now, and miniaudio follows default
  changes while running.
- **Listen for `WM_POWERBROADCAST` resume.** Not done: a resume either
  produces a WASAPI timeout (`DeviceStopped`) or a flat counter, and both
  are covered. A resume hook would only save the 2.5 s window.

## Consequences

- A capture that stops for any reason is reopened within about 3 s. The
  user hears about it only when it keeps failing.
- `AudioSource::start` takes a second callback for device notifications.
  `ReplaySource` ignores it.
- The HUD's MIC readout still shows the microphone name read at boot (and
  on a relaunch replay); the outage is visible in the log and, if it lasts,
  in a notification. Showing it live in the HUD needs a new overlay setter
  and is left for later.
- The real-device path (a WASAPI timeout, an unplugged USB microphone, a
  default-device switch) cannot run in automated tests. Catch2 covers the
  engine with a fake source that stops delivering, reports notifications
  and fails to reopen. The watchdog is tested with a fake clock.
