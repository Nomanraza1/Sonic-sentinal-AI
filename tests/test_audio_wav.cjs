const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const context = vm.createContext({Blob});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/audio-wav.js'), 'utf8'), context);

test('browser encoder writes a standard mono PCM header at 16, 44.1 and 48 kHz', async () => {
  for (const rate of [16000, 44100, 48000]) {
    const samples = new Float32Array(rate * 3);
    const wav = Buffer.from(await context.SonicAudio.encodeWav(samples, rate).arrayBuffer());
    assert.equal(wav.toString('ascii', 0, 4), 'RIFF');
    assert.equal(wav.readUInt32LE(4), wav.length - 8);
    assert.equal(wav.toString('ascii', 8, 16), 'WAVEfmt ');
    assert.equal(wav.readUInt32LE(16), 16);
    assert.equal(wav.readUInt16LE(20), 1);
    assert.equal(wav.readUInt16LE(22), 1);
    assert.equal(wav.readUInt32LE(24), rate);
    assert.equal(wav.readUInt32LE(28), rate * 2);
    assert.equal(wav.readUInt16LE(32), 2);
    assert.equal(wav.readUInt16LE(34), 16);
    assert.equal(wav.toString('ascii', 36, 40), 'data');
    assert.equal(wav.readUInt32LE(40), samples.length * 2);
  }
});

test('samples begin after the header and preserve polarity with clipping', async () => {
  const wav = Buffer.from(await context.SonicAudio.encodeWav([-2, -1, 0, 1, 2], 48000).arrayBuffer());
  assert.deepEqual(Array.from({length:5}, (_, i) => wav.readInt16LE(44+i*2)), [-32768,-32768,0,32767,32767]);
});
