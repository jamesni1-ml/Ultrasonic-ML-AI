# Node-RED Integration — Ultrasonic UV Panel Monitor

Custom Node-RED node for classifying ultrasonic audio from the **Dodotronic Ultramic 384K EVO** using a trained MobileNetV2 TorchScript model on Raspberry Pi 4.

## Architecture

```
Ultramic 384K EVO → arecord → WAV files → [Node-RED] → Outputs
                                               │
                                ┌──────────────┴──────────────┐
                                │   ultrasonic-classifier      │
                                │   (custom Node-RED node)     │
                                │   Calls Python inference     │
                                └──────────────┬──────────────┘
                                               │
                          ┌────────────┬───────┼────────┬─────────────┐
                       Port 1       Port 1   Port 1   Port 2       Port 2
                       ABNORMAL     ABNORMAL  ABNORMAL NORMAL       NORMAL
                       + image      + image   + image  (status)     (status)
                          │            │        │         │            │
                       MQTT out    TCP out   BLE out   MQTT out    TCP out
```

## Node: `ultrasonic-classifier`

### Inputs

| Property | Type | Description |
|----------|------|-------------|
| `msg.payload` | string | Absolute path to a 1-second WAV file (384 kHz) |

### Outputs

| Port | Fires when | Includes image? | Description |
|------|-----------|-----------------|-------------|
| **Port 1** | Abnormal detected (above threshold) | Yes — base64 PNG | Spectrogram image + full classification result |
| **Port 2** | Normal / below threshold | No | Classification result only |

### Output payload

```json
{
  "prediction": "abnormal",
  "confidence": 0.92,
  "probabilities": {
    "normal": 0.05,
    "abnormal": 0.92,
    "abnormal2": 0.02,
    "pulsing_unstable": 0.01
  },
  "is_abnormal": true,
  "wav": "/tmp/ultrasonic-incoming/clip_042.wav",
  "timestamp": "2026-03-12T14:30:00.000Z",
  "image": "<base64 PNG data>",
  "image_path": "/tmp/ultrasonic-events/abnormal_1741789800123.png",
  "image_mime": "image/png"
}
```

Normal output is the same but without `image`, `image_path`, or `image_mime`.

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Model path | *(required)* | Path to TorchScript `.pt` model |
| Meta JSON | *(auto)* | Path to `meta.json` — auto-detected next to model if blank |
| Python | `python3` | Python executable path |
| Infer script | *(auto)* | Path to `ultrasonic_infer.py` — auto-detected if blank |
| Threshold | `0.80` | Confidence threshold (0–1) for abnormal detection |
| Event dir | `/tmp/ultrasonic-events` | Where to save spectrogram PNGs |

## Installation on Raspberry Pi

### Prerequisites

1. **Node-RED** installed on the Pi:
   ```bash
   bash <(curl -sL https://raw.githubusercontent.com/node-red/linux-installers/master/deb/update-nodejs-and-nodered)
   ```

2. **Python inference dependencies**:
   ```bash
   pip3 install -r requirements-pi.txt
   ```

3. **Trained model** files on the Pi:
   ```bash
   # Copy from your training machine
   scp ultrasonic_model.pt meta.json pi@raspberrypi:~/models/
   scp ultrasonic_infer.py pi@raspberrypi:~/
   ```

### Install the node

```bash
cd ~/.node-red
npm install /path/to/Ultrasonic-ML-AI/node-red
```

Or symlink for development:
```bash
cd ~/.node-red/node_modules
ln -s /path/to/Ultrasonic-ML-AI/node-red node-red-contrib-ultrasonic-classifier
```

Restart Node-RED:
```bash
node-red-restart
```

The **ultrasonic classifier** node will appear in the **analysis** category of the palette.

## Example Flows

Import any of these via **Menu → Import → Clipboard** in the Node-RED editor.

### MQTT Output
**File:** `examples/flow-mqtt.json`

Publishes results to an MQTT broker:
- `uv/panels/abnormal` — QoS 1, retained (with spectrogram image)
- `uv/panels/normal` — QoS 0 (status only)

**Requires:** MQTT broker (e.g. Mosquitto on the Pi: `sudo apt install mosquitto`)

### TCP Output
**File:** `examples/flow-tcp.json`

Sends JSON results over TCP sockets:
- Port 9000 — Abnormal detections (with image)
- Port 9001 — Normal readings

Edit the TCP out nodes to set your server address.

### Bluetooth (BLE) Output
**File:** `examples/flow-bluetooth.json`

Sends compact status messages over Bluetooth Low Energy:
- `{"t":"ABN","c":"abnormal","p":92,"ts":1741789800}` — abnormal alert
- `{"t":"OK","p":95,"ts":1741789800}` — normal status

**Note:** BLE has limited MTU (~20–512 bytes), so images are NOT sent over BLE. They stay on disk in the event directory. Only compact JSON status is transmitted.

**Requires:**
```bash
cd ~/.node-red
npm install node-red-contrib-generic-ble
```

### All Outputs Combined
**File:** `examples/flow-all-outputs.json`

Complete flow with MQTT + TCP + BLE all wired up. Includes a test inject node for manual testing.

## Recording Pipeline

To feed WAV files into the flow automatically, set up a recording cron job or systemd timer:

```bash
# Record 1-second clips continuously into the watched directory
while true; do
    FNAME="/tmp/ultrasonic-incoming/$(date +%s%N).wav"
    arecord -D hw:1,0 -f S16_LE -r 384000 -c 1 -d 1 "$FNAME"
done
```

Or use a systemd service — see the main project README for mic setup details.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Node not in palette | Restart Node-RED after `npm install` |
| "Python not found" | Set full path in node config: `/usr/bin/python3` |
| "Model path not configured" | Double-click the node and set the model `.pt` path |
| Slow inference | Normal on Pi 4 (~2–5s). Consider reducing `IMG_SIZE` to 128 in training |
| "file not found" error | Check that the WAV path in `msg.payload` is an absolute path |
| BLE node missing | Install: `cd ~/.node-red && npm install node-red-contrib-generic-ble` |
