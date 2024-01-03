# Students markup

## For developers

Before commit autoformat and lint your code

```sh
isort .
black .
pflake8 .
```

## Setup neural network

1. Use linux (or docker)
2. Install dependencies `pip3 install -r requirements-ml.txt`
3. Initialize submodules `git submodule update --init`

4. Download weights
   1. Through command line (if gdrive denied access do manually)
      ```bash
      # base dirs
      mkdir models
      mkdir models/alignment
      mkdir models/detection
      mkdir models/recognition
      # base
      gdown https://drive.google.com/u/0/uc\?id\=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8 -O /tmp/antelopev2.zip
      unzip -o /tmp/antelopev2.zip -d /tmp/
      cp /tmp/antelopev2/2d106det.onnx models/alignment/
      cp /tmp/antelopev2/scrfd_10g_bnkps.onnx models/detection/
      cp /tmp/antelopev2/glintr100.onnx models/recognition/
      # retina
      gdown https://drive.google.com/u/0/uc?id=1wm-6K688HQEx_H90UdAIuKv-NAsKBu85 -O /tmp/retinaface-R50.zip
      mkdir models/detection/retinaface/
      unzip -o /tmp/retinaface-R50.zip -d models/detection/retinaface
      ```
   2. Manually
      1. TODO