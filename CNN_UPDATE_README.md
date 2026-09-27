# CNN model update

The included trained model achieved 86.62% test accuracy and 89.52% validation accuracy.

## Apply to your existing Drive repository

1. Extract this ZIP into the existing `Sonic-sentinal AI` repository folder, replacing the matching code/model files. Keep all relative folders intact. No audio needs to be uploaded again.
2. In Colab, change to that repository folder, then run `%pip install -r requirements-trained-model.txt` if you want to use the included trained artifact. Restart the runtime if packages were already imported.
3. The model is already trained and `python_models/selection.json` selects it. Restart the Flask app to use it. `python_models/yamnet_base/` must remain alongside the classifier.
4. To retrain instead, open `notebooks/colab_transfer_training.ipynb`, verify `REPO_DIR`, then run all cells. Training requires your existing `audio_dataset/raw/<class>/` source audio.

Existing classical model files are not replaced. To roll back, copy `python_models/selection_before_transfer.json` over `python_models/selection.json`.

See `reports/cnn_training_results.md` for measured results and limitations. Background-noise recall is still 62.22%.
