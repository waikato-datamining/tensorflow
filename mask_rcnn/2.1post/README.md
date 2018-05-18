# Mask-RCNN object shape detection

These instructions assume that you have your virtual environment set up 
(`python` is a Python 3 executable).


## Balloon example

This is based on the *balloon* example from the [Mask-RCNN](https://github.com/matterport/Mask_RCNN) project.

This is an example showing the use of Mask RCNN in a real application.
1. We train the model to detect objects only.
2. We use the generated masks to keep objects in color while changing 
   the rest of the image to grayscale.
3. We generate a CSV file with the bounding boxes of the images that 
   were detected. 


### Dataset and pre-trained model

From the [Releases page](https://github.com/matterport/Mask_RCNN/releases) page:
1. Download [mask_rcnn_object.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5). 
2. Download [balloon_dataset.zip](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip)
   and expand the archive. 

Alternatively, you can also download an ImageNet network:

https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5


### Configuration

Copy the `object_example.yaml` configuration template file to `balloon.yaml`.
Change the `NAME` property to `balloon`. The default of 30 epochs and 100 steps
should be sufficient. Make sure the directory specified under `LOGS` is present,
it will contain the log files and the models (after each iteration a model is output).


### Training

Train a new model starting from pre-trained COCO weights (using 
the `.h5` network you downloaded just before):

```bash
python object.py train --dataset=/path/to/balloon/dataset \
  --weights=/some/where/mask_rcnn_balloon.h5 \
  --conf=/path/to/balloon.yaml
```

Resume training a model that you had trained earlier:

```bash
python object.py train --dataset=/path/to/object/dataset \
  --weights=last \
  --conf=/path/to/balloon.yaml
```

Train a new model starting from ImageNet weights (downloads 
the pre-trained network from github):

```bash
python object.py train --dataset=/path/to/object/dataset \
  --weights=imagenet \
  --conf=/path/to/balloon.yaml
```


### Apply color splash using the provided weights

Apply splash effect on an image:

```bash
python object.py splash \
  --weights=/path/to/mask_rcnn/mask_rcnn_object.h5 \
  --image=<file name or URL> \
  --output_dir=/path/to/output_dir \
  --config=/path/to/balloon.yaml
```

Apply splash effect on a video. Requires OpenCV 3.2+:

```bash
python object.py splash \
  --weights=/path/to/mask_rcnn/mask_rcnn_object.h5 \
  --video=<file name or URL> \
  --output_dir=/path/to/output_dir \
  --config=/path/to/balloon.yaml
```

### Generate CSV with bounding box details

Instead of generating a *splash* image, you can also generate a CSV file with 
details of the bounding boxes. Simply use the command *bbox* command. 

```bash
python object.py bbox \
  --weights=/path/to/mask_rcnn/mask_rcnn_object.h5 \
  --image=<file name or URL> \
  --output_dir=/path/to/output_dir \
  --config=/path/to/balloon.yaml
```


## Create your own dataset

### Train and test set

* create a directory for your train and test set
* create sub-directories `train` and `val`
* divide your images among the two sub-directories


### Annotate the images

Use the [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/) HTML annotator:
* download the archive and decompress it (only necessary once)
* open the `via.html` in your browser
* for each directory, `train` and `val` do the following: 
  * load all the images that you want to annotate
  * export the annotations to `via_region_data.json` in that directory  


### Configuration

* Copy the `object_example.yaml` file
* adjust the `NAME` and `LOGS` properties   

### Training

See *balloon* example above.


## Notes on object.py

* When running the code on a multi-GPU machine, you will want to specify the GPU to run on
  using the `--gpu <id>` option. For instance, use `2` to run it on the third GPU.
* Restricting the amount of memory that tensorflow uses on the GPU can be achieved using
  the `tf_gpu_options_per_process_gpu_memory_fraction` parameter in the YAML file (0-1).
  `0.4` works well on a 11GB Nvidia 1080 Ti. 
* When supplying directories for the `--image` or `--video` options, then these directories
  get traversed. Useful when performing batch-processing, as the model only gets loaded 
  and initialized once.


## Updating bounding boxes

If you trained another network that predicted bounding boxes for your object, but produces
ones that are two large, then you can use MASK RCNN to tighten them up using the following
process (most of these steps are described in detail above):

* extract the objects from your images using the identified bounding boxes
* divide them into train and test and annotate them using VIA
* create a configuration YAML file and adjust parameters
* build a new model on these smaller objects

The `update_bbox.py` script can be used to update a CSV that stores the bounding boxes that
need tightening up. The script expects the following column names in the CSV file (order is 
irrelevant, as it uses the column names):
 * x0 -- left
 * y0 -- top
 * x1 -- right
 * y1 -- bottom
 * label_str -- used to identify the bounding boxes to update (via the `--label ...` option) 
 * score -- values from 0 to 1

Here is an example
```csv
file,x0,y0,x1,y1,label_str,score
file1.png,641.2107544,419.1536865,711.9541626,478.366333,myobject,0.955762744
file1.png,1522.893433,234.7555542,1575.258423,286.5966187,myobject,0.926829338
file2.png,629.8745728,260.2509766,688.4630737,320.1975098,myobject,0.601661503
...
```

```bash
python update_bbox.py \
  --image /path/to/original_image.png \
  --bbox /path/to/csv_file/with/annotations.csv  \
  --out /path/to/updated/csv_file/updated.csv \
  --weights /path/to/trained/model.h5 \
  --config /path/to/your/config.yaml \
  --label myobject
```


## Notes on update_bbox.py

* When running the code on a multi-GPU machine, you will want to specify the GPU to run on
  using the `--gpu <id>` option. For instance, use `2` to run it on the third GPU.
* Restricting the amount of memory that tensorflow uses on the GPU can be achieved using
  the `tf_gpu_options_per_process_gpu_memory_fraction` parameter in the YAML file (0-1).
  `0.4` works well on a 11GB Nvidia 1080 Ti. 
* When supplying directories for the `--image` or `--video` options, then these directories
  get traversed. Useful when performing batch-processing, as the model only gets loaded 
  and initialized once. 
* In *directory traversal mode*, the `--bbox` and `--out` options do not need to be supplied.
  The script looks for `-rois.csv` and `.csv` with the same filename as the PNG or JPG files
  that it encounters in that directory for loading the bounding box data from. The updated
  CSV file will have an automatic `-updated.csv` ending.
