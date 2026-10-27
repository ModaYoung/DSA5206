# Data attribution

The 94 chest radiographs in cache/images were retrieved from the COVID-19 Image Data Collection at the fixed public commit below. They are source research material, not images created for this project. Per-image sources and license descriptions, where provided by the original collection, remain in data/scored_metadata.csv and data/metadata_snapshot.csv.

- Dataset: https://github.com/ieee8023/covid-chestxray-dataset/tree/196143d3dd32a2a43ad51d6891d460aaf690d741
- Paper: Cohen JP et al. Predicting COVID-19 Pneumonia Severity on Chest X-ray With Deep Learning. Cureus 2020;12:e9448. https://doi.org/10.7759/cureus.9448
- Scores: Original CC BY-SA attribution retained in data/severity_scores.csv.
- Image checksums: data/image_manifest.csv.
- Model and feature checksums: data/provenance.json.
- Original models: https://github.com/mlmed/torchxrayvision

The original image, metadata, label and model terms continue to apply. This repository does not relicense or claim ownership of third-party materials. The lighter downloadable experiment ZIP contains cached features and results and excludes the raw radiographs and pretrained weights. No course lecture PDFs are redistributed here.
# Additional pretrained encoders

The extension uses frozen public BioViL, BioViL-T and RAD-DINO image encoders. Source revisions, weight hashes and preprocessing are recorded in `data/extended_feature_provenance.json`. Original model terms continue to apply. We distribute the extracted feature arrays for this coursework, not the pretrained weights or the encoders' pretraining images. BioViL and BioViL-T are loaded through Microsoft's MIT-licensed hi-ml implementation; the RAD-DINO public model card identifies its model license as MIT. This does not change the separate terms of the original radiographs or severity labels.

