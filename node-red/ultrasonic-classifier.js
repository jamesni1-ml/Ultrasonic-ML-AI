/**
 * Node-RED node: ultrasonic-classifier
 *
 * Classifies 1-second WAV segments from the Ultramic 384K EVO using
 * a TorchScript MobileNetV2 model. Two output ports:
 *   Port 1 — Abnormal detection (includes base64 spectrogram PNG)
 *   Port 2 — Normal / below-threshold (no image)
 */

const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

module.exports = function (RED) {

    function UltrasonicClassifierNode(config) {
        RED.nodes.createNode(this, config);
        const node = this;

        // Config from editor UI
        node.modelPath = config.modelPath || '';
        node.metaPath = config.metaPath || '';
        node.pythonPath = config.pythonPath || 'python3';
        node.inferScript = config.inferScript || '';
        node.threshold = parseFloat(config.threshold) || 0.80;
        node.outDir = config.outDir || '/tmp/ultrasonic-events';

        // Resolve paths
        const defaultInferScript = path.resolve(__dirname, '..', 'ultrasonic_infer.py');

        node.on('input', function (msg, send, done) {
            send = send || function () { node.send.apply(node, arguments); };

            const wavPath = msg.payload || msg.filename;
            if (!wavPath || typeof wavPath !== 'string') {
                node.status({ fill: 'red', shape: 'ring', text: 'no WAV path' });
                if (done) done(new Error('msg.payload must be a WAV file path'));
                return;
            }

            // Verify WAV file exists
            if (!fs.existsSync(wavPath)) {
                node.status({ fill: 'red', shape: 'ring', text: 'file not found' });
                if (done) done(new Error('WAV file not found: ' + wavPath));
                return;
            }

            const model = node.modelPath;
            const meta = node.metaPath;
            const script = node.inferScript || defaultInferScript;
            const threshold = node.threshold;
            const outDir = node.outDir;

            if (!model) {
                node.status({ fill: 'red', shape: 'ring', text: 'no model path' });
                if (done) done(new Error('Model path not configured'));
                return;
            }

            node.status({ fill: 'blue', shape: 'dot', text: 'classifying...' });

            const args = [
                script,
                '--model', model,
                '--wav', wavPath,
                '--threshold', String(threshold),
                '--out_dir', outDir
            ];

            if (meta) {
                args.push('--meta', meta);
            }

            execFile(node.pythonPath, args, {
                timeout: 30000,
                maxBuffer: 1024 * 1024
            }, function (err, stdout, stderr) {
                if (err) {
                    node.status({ fill: 'red', shape: 'ring', text: 'error' });
                    node.error('Inference failed: ' + (stderr || err.message), msg);
                    if (done) done(err);
                    return;
                }

                // Parse JSON output from last line of stdout
                let result;
                try {
                    const lines = stdout.trim().split('\n');
                    result = JSON.parse(lines[lines.length - 1]);
                } catch (parseErr) {
                    node.status({ fill: 'red', shape: 'ring', text: 'parse error' });
                    node.error('Failed to parse inference output: ' + stdout, msg);
                    if (done) done(parseErr);
                    return;
                }

                // Build output message
                const outMsg = {
                    payload: {
                        prediction: result.prediction,
                        confidence: result.confidence,
                        probabilities: result.probabilities,
                        is_abnormal: result.is_abnormal,
                        wav: result.wav,
                        timestamp: new Date().toISOString()
                    },
                    filename: wavPath
                };

                // If abnormal and spectrogram PNG was saved, attach as base64
                if (result.is_abnormal && result.spectrogram_png) {
                    try {
                        const pngBuffer = fs.readFileSync(result.spectrogram_png);
                        outMsg.payload.image = pngBuffer.toString('base64');
                        outMsg.payload.image_path = result.spectrogram_png;
                        outMsg.payload.image_mime = 'image/png';
                    } catch (imgErr) {
                        node.warn('Could not read spectrogram PNG: ' + imgErr.message);
                    }
                }

                if (result.is_abnormal) {
                    // Port 1: Abnormal (with image)
                    node.status({
                        fill: 'red', shape: 'dot',
                        text: result.prediction + ' (' + (result.confidence * 100).toFixed(1) + '%)'
                    });
                    send([outMsg, null]);
                } else {
                    // Port 2: Normal (no image)
                    node.status({
                        fill: 'green', shape: 'dot',
                        text: 'normal (' + (result.confidence * 100).toFixed(1) + '%)'
                    });
                    send([null, outMsg]);
                }

                if (done) done();
            });
        });

        node.on('close', function () {
            node.status({});
        });
    }

    RED.nodes.registerType('ultrasonic-classifier', UltrasonicClassifierNode);
};
