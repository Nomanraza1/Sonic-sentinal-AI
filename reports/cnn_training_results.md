# Trained audio CNN results

Actual local training run on the recovered source audio. The CNN is Google YAMNet (frozen), followed by a standardized RBF SVM trained on pooled CNN embeddings and mel/MFCC statistics.

| Model | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| Previous selected SVM | 79.87% | 76.86% | 76.04% |
| YAMNet + SVM | 89.52% | 86.62% | 85.96% |

Nine candidates were compared using validation macro-F1. The test split was evaluated only for the winner in this experiment. Selection: `svm_combined_c100`. The deployed prediction rule is probability argmax, matching evaluation.

The previous SVM numbers are from its saved Colab report; that older model was not retrained in this run. The new deterministic split has 2,099 training, 450 validation and 450 test source recordings. After segmentation/filtering: 2,167 training, 477 validation and 471 test segments. No exact source hash crosses splits. All ten classes are represented. No audio augmentation was used.

## Test recall by class

| Class | Recall | Segments |
|---|---:|---:|
| machinery_fault | 88.89% | 45 |
| glass_breaking | 93.33% | 45 |
| alarm_siren | 80.00% | 45 |
| vehicle_horn | 80.00% | 45 |
| animal_sound | 91.11% | 45 |
| gunshot | 93.33% | 45 |
| panic_scream | 88.89% | 45 |
| aggression | 86.67% | 45 |
| person_asking_for_help | 96.97% | 66 |
| background_noise | 62.22% | 45 |

Background noise remains the weakest class. These are segment-level results on the project benchmark, which has previously been inspected. They do not establish the same accuracy on new microphones, speakers or environments.

## Verification

- Nine automated tests passed.
- Real test audio passed through the application prediction function using the active CNN model.
- Live extracted features matched the experiment cache; all ten class probabilities were finite and normalized.
- Existing classical models were retained; previous selection is in `python_models/selection_before_transfer.json`.

## Reproduce

Run `python python_models/train_transfer.py --extract --activate`, or use `notebooks/colab_transfer_training.ipynb`. The first run downloads the base model. Interrupted extraction resumes from cached segments. Full candidate results and source manifests are under `data/transfer_experiment/` locally.

Environment: Python 3.13.9, TensorFlow 2.20.0, scikit-learn 1.7.2, NumPy 2.3.5, joblib 1.5.2.

YAMNet reference: https://www.tensorflow.org/tutorials/audio/transfer_learning_audio
