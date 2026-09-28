"""Convert the checked-in TensorFlow.js layers export to a Keras model."""
import json
import struct
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "gtm_model"


def build_model(input_shape, classes):
    model = tf.keras.Sequential(name="sonic_sentinel_gtm")
    model.add(tf.keras.layers.Input(shape=input_shape, name="input_layer"))
    for index, (filters, pool) in enumerate(((32, True), (64, True), (128, True), (128, True))):
        suffix = "" if index == 0 else f"_{index}"
        model.add(tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv2d{suffix}"))
        model.add(tf.keras.layers.BatchNormalization(name=f"batch_normalization{suffix}"))
        model.add(tf.keras.layers.ReLU(name=f"re_lu{suffix}"))
        if pool:
            model.add(tf.keras.layers.MaxPooling2D(2, name=f"max_pooling2d{suffix}"))
    model.add(tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d"))
    model.add(tf.keras.layers.Dropout(0.3, name="dropout"))
    model.add(tf.keras.layers.Dense(classes, activation="softmax", name="dense"))
    model(np.zeros((1, *input_shape), dtype=np.float32))
    return model


def convert():
    config = json.loads((MODEL_DIR / "model.json").read_text(encoding="utf-8"))
    meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    if config.get("format") != "layers-model":
        raise ValueError("Expected a TensorFlow.js layers-model export")
    topology = config["modelTopology"]["model_config"]["config"]
    input_shape = tuple(topology["build_input_shape"][1:])
    model = build_model(input_shape, len(meta["labels"]))
    shard_path = MODEL_DIR / config["weightsManifest"][0]["paths"][0]
    raw = shard_path.read_bytes()
    offset = 0
    per_layer = {}
    for weight in config["weightsManifest"][0]["weights"]:
        size = int(np.prod(weight["shape"]))
        end = offset + size * 4
        if end > len(raw):
            raise ValueError(f"Truncated GTM weight shard at {weight['name']}")
        values = np.frombuffer(raw[offset:end], dtype="<f4").reshape(weight["shape"]).copy()
        offset = end
        layer_name, variable_name = weight["name"].rsplit("/", 1)
        per_layer.setdefault(layer_name, {})[variable_name] = values
    if offset != len(raw):
        raise ValueError("GTM shard has unreferenced trailing bytes")
    converted_layers = set()
    for layer in model.layers:
        weights = per_layer.get(layer.name)
        if weights is None:
            if layer.weights:
                raise ValueError(f"No exported weights found for layer {layer.name}")
            continue
        values = []
        for variable in layer.weights:
            short = variable.name.rsplit("/", 1)[-1].split(":", 1)[0]
            if short not in weights:
                raise ValueError(f"Missing GTM weight {layer.name}/{short}")
            values.append(weights[short])
        if len(values) != len(layer.get_weights()):
            raise ValueError(f"Unexpected weight count in layer {layer.name}")
        layer.set_weights(values)
        converted_layers.add(layer.name)
    if converted_layers != set(per_layer):
        raise ValueError(f"Unmapped GTM weight layers: {sorted(set(per_layer) - converted_layers)}")
    model.save(MODEL_DIR / "model.keras")
    labels = ["person_asking_for_help" if label == "person_asking_help" else label for label in meta["labels"]]
    (MODEL_DIR / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    reopened = tf.keras.models.load_model(MODEL_DIR / "model.keras", compile=False)
    probe = np.zeros((1, *input_shape), dtype=np.float32)
    if not np.allclose(model(probe, training=False).numpy(), reopened(probe, training=False).numpy(), atol=1e-6):
        raise ValueError("Saved GTM model did not reproduce its pre-save output")
    print(f"Saved {MODEL_DIR / 'model.keras'} with {model.count_params():,} parameters")


if __name__ == "__main__":
    convert()
