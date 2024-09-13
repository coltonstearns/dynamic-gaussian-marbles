> The below instructions assume a Linux-based operating system with a bash shell and ffmpeg installed.  
<img src="../media/linux-icon.png" alt="drawing" width="80"/>
<img src="../media/ffmpeg-icon.png" alt="drawing" width="120"/>


## Step 1: Create Data Directory for Video
For your video of interest, `xxxxx.mp4`, initialize a data directory containing
the video frames. We recommend the folder `./data/real-world/xxxxx`. To do this, pass in a video 
path and desired output data directory into `00_initialize_directory.sh`:

```aiignore
bash 00_initialize_directory <path_to_mp4> ./data/real-world/<your_video_name>
```

## Step 2: Get Instance Masks from Segment Anything V2
We simply use the latest [SAM2 Demo](https://sam2.metademolab.com/demo) to estimate temporally consistent instance masks.

* In your web browser, follow the [demo link](https://sam2.metademolab.com/demo), arriving at the following page:
![Sam2DemoLandingPage](../media/sam2-landing-page.png)
* Click "change video" in the bottom left, and upload the video at `./data/real-world/<your_video_name>/video.mp4`. 
It is important to upload THIS video, because the SAM2 demo only operates on 24 FPS videos.
* Use SAM2 to segment out each foregound object that exhibits highly distinct motion:
  * Follow the demo's prompting to segment a SINGLE foreground object:
  ![Sam2ClickSegment](../media/sam2-click-segment.png)
  * After reaching a suitable segmentation for this ONE foreground object, click "Good to Go" and then set the 
  foreground and backgrond effects to **Erase**, resulting in a binary mask.
  ![Sam2BinarySegment.png](../media/sam2-binary-segment.png)
  * Finally, click "Next" and then the "Download" Button. This downloads an `mp4` file. ![Sam2Download.png](../media/sam2-download.png)
  * Move the `mp4` file into the `./data/real-world/<your_video_name>/sam2/` directory.
  * Repeat this for EACH foreground object that has distinct motion. 

> Note: due to the current code setup, there must be at least one foreground object present in each frame. If this does not
> happen in your video, then segment out the background as a separate "foreground" class.

> Note: Segmenting more foreground instances is helpful but not necessary. We usually segment one to three obvious foreground
> instances, and then leave the rest for the dynamic background.

After this step, the folder `./data/real-world/<your_video_name>/sam2/` should be populated with an `mp4` video for each unique
foreground instance.

## Step 3: Preprocess Everything Else
Pass the data directory and base conda environment to `preprocess_video.sh`:
```aiignore
bash preprocess_video.sh ./data/real-world/<your_video_name> ~/anaconda3
```
> Note: Running CoTracker typically takes over an hour, depending on the length of the video.
