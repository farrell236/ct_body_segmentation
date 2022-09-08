# CT Whole Body Segmentation Model

TODO

### Run Model
```
usage: segmentbody3d.py [-h] [-i INPUT_FN] [-o OUTPUT_FN] [-m MODEL]
                        [-minW MIN_WINDOW] [-maxW MAX_WINDOW] [-iso]
                        [-p PATCH_SIZE] [-s STRIDE] [-v] [-b BATCH_SIZE]

Body Organ Segmentation for CT Images

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FN, --input_fn INPUT_FN
                        Input CT Volume
  -o OUTPUT_FN, --output_fn OUTPUT_FN
                        Output Segmentation
  -m MODEL, --model MODEL
                        Trained Model
  -minW MIN_WINDOW, --min_window MIN_WINDOW
                        Intensity Window minimum value
  -maxW MAX_WINDOW, --max_window MAX_WINDOW
                        Intensity Window maximum value
  -iso, --isotropic     Input Volume is Isotropic
  -p PATCH_SIZE, --patch_size PATCH_SIZE
                        3D Patch Division (patch)
  -s STRIDE, --stride STRIDE
                        3D Patch Division (stride)
  -v, --verbose         Verbose Output
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Inference Batch Size

```

### Requirements

```
Python 3.x.x
```

### Packages:

```
TODO
```

### References

```
TODO
```
