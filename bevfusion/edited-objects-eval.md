# Evaluation on edited objects

We can run an evaluation on objects that have been edited with MObI. There are two possible evaluations: either full evaluation (just as normal) or restricted evaluation that only computes the performance of the model on edited objects. Note that when using the restricted evaluation, the mAPs will all be zero because they can not be computed.

We needed to modify the nuscenes-devkit in order to implement this and hence as a first step, install it:
```
pip install -e ../nuscenes-devkit/setup/
```

## How to

For example if you are using `nuscenes-mini`, move the edited samples and the objects list as follows:
```
edited samples -> nuscenes-mini/samples-edited/
edited objects list -> nuscenes-mini/samples-edited/objects.json
```

Then, make sure you can run the normal eval (for example):
```
torchpack dist-run -np 2 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox --eval-options jsonfile_prefix=results
```
This should dump the evaluation results in `./results` for your peruse.

You can then add (and combine) the following eval-options:
- edited_samples_path: use the edited samples instead of the normal samples (both camera and lidar)
- edited_objects_list: restrict the evalution to the objects defined in the list (in format {sample_token: trajectory_id}
For example, to run the restricted evaluation on edited samples, run
```
torchpack dist-run -np 2 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox --eval-options jsonfile_prefix=results edited_samples_path=samples-edited edited_objects_list=samples-edited/objects.json
```

## Visualisation

We've also modified the visualisation script in order to pick samples edited by MObi. Here is how you can run the visualisation script on edited samples:
```
torchpack dist-run -np 2 python tools/visualize.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --checkpoint pretrained/bevfusion-det.pth --mode pred --out-dir viz-mobi --edited-samples-path samples-edited
```