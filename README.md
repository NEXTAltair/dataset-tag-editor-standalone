# TODO

タガーを切り出してパッケージ化

スコアリング系は別に切り出して､(別の)としあき製のユーザー学習可能なスコアリングモジュールと混ぜ合わせる

# Dataset Tag Editor Standalone

[日本語 Readme](README-JP.md)

This is a WebUI tool to edit training dataset for Text2Image Models.  
This is **a standalone version** of [Dataset Tag Editor](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor), which is an extension for [Stable Diffusion web UI by AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui).  
**Please do not put into `extensions` folder of AUTOMATIC1111's webUI.**

![](pic/ss01.png)

It works well with text captions in comma-separated style (such as the tags generated by DeepDanbooru interrogator).

Caption in the filenames of images can be loaded, but edited captions can only be saved in the form of text files.

## Difference from the Extension Version
(Pros)  
- Avoid bugs caused by specific version of gradio
- Much faster startup and running

(Cons)
- Hijacked CLIP tokenizer is not available (which has SDwebUI specific features like embedding, emphasis, …)
- No more support for old Python < 3.9


## Requirements
All requirements are listed in `requirements.txt`  


**Please install the followings first:**  
- [Python](https://www.python.org/) >= 3.9 (Developed on 3.10.11)  
- [PyTorch](https://pytorch.org/) with CUDA >= 1.10.0 ([onnx-pytorch](https://github.com/fumihwh/onnx-pytorch) requirement; for [TorchDeepDanbooru](https://github.com/AUTOMATIC1111/TorchDeepDanbooru))  
PyTorch version mostly depends on [transformers](https://github.com/huggingface/transformers), please follow [the instruction to install transformers](https://github.com/huggingface/transformers#installation) if install specific version manually.

If you want to use DirectML, please install manually in venv  (install [pytorch-directml](https://pypi.org/project/pytorch-directml/) to enable, **not tested**).

This script will install [ONNX runtime](https://onnxruntime.ai/) **automatically** in `venv` before using wd-taggers by SmilingWolf.

## Installation
### Windows
Just run `install.bat`  

### Linux (or install manually on Windows)
Run following commands on the root directory of this repo.
```sh
python3 -m venv --system-site-packages venv
source ./venv/bin/activate
pip3 install -r requirements.txt
```
(Note: just `.\venv\Scripts\activate` is needed to activate venv on Windows)


## Launch
You can see available command line args with `-h` or `--help` option.

### Windows
Just run `launch_user.bat`

### Linux
```sh
source ./venv/bin/activate
python scripts/launch.py [arguments]
```

### Google Colab

Google Colab users can using it by executing the following command and accessing the generated Gradio Public URL.  
(Probably, I think this is currently only available in the Colab Pro.)

```sh
%cd /content
!git clone https://github.com/toshiaki1729/dataset-tag-editor-standalone.git
%cd /content/dataset-tag-editor-standalone
!pip install -r requirements.txt
!python scripts/launch.py --share
```

## Features
Note. "tag" means each blocks of caption separated by commas.
- Edit and save captions in text file (webUI style) or json file ([kohya-ss sd-scripts metadata](https://github.com/kohya-ss/sd-scripts))
- Edit captions while viewing related images
- Search tags
- Filter images to edit their caption by tags
  - AND/OR logic can be used in each Positive/Negative filters
- Batch replace/remove/append tags
- Batch sort tags
- Batch search and replace
  - [regular expression](https://docs.python.org/3/library/re.html#regular-expression-syntax) can be used
- Use interrogators
  - BLIP, BLIP2, GIT, DeepDanbooru, [Z3D-E621-Convnext](https://huggingface.co/toynya/Z3D-E621-Convnext), SmilingWolf's [WDv1.4 Tagger](https://huggingface.co/SmilingWolf) (v1, v2, v3 and some variants of them)
- You can add Custom Tagger in `userscripts/taggers` (they have to be wrapped by a class derived from `scripts.tagger.Tagger`)
  - Some Aesthetic Score Predictors are implemented in there
- Batch remove image and/or caption files


## Usage
1. Make dataset
    - better to use already cropped images
1. Load them
    - use interrogator if needed
1. Edit their captions
    - filter images you want to edit by tags in "Filter by Tags" tab
    - filter images manually in "Filter by Selection" tab
    - replace/remove tags or append new tags in "Batch Edit Captions" tab
    - edit captions individually in "Edit Caption of Selected Image" tab
      - you also can use interrogator here
    - move/delete files in "Move or Delete Files" tab if needed
1. Click "Save all changes" button


## By the way, how can I edit tags quickly?

Basic workflow is as follows:

1. Filter images
1. Batch edit

Please note that all batch editing will be applyed **only to displayed images (=filtered images)**.

### 1. Which filter is appropriate?
#### I want to edit all at once  
  No filter is required.
#### Some images require editing    
  1. **They should / shouldn't already have same tag(s)**  
    Go to "Filter by Tags" so that the only images to be edited are displayed.
  1. **They have nothing in common**  
    Go to "Filter by Selection" and apply.  
    Images can also be added to the filter by pushing [Enter] key.

### 2. How can I edit as I want?
  ![](pic/ss07.png)  
#### I want to add some new tags  
  1. Go to "Batch Edit Captions" tab
  1. Append tags to "Edit tags" textbox
  1. Push "Apply changes to filtered images" button  
  ![](pic/ss08.png)  
  "foo" and "bar" will be added to all images displayed.

#### I want to replace the tags which are common to displayed images  
  1. Go to "Batch Edit Captions" tab
  1. Replace tags in "Edit tags" textbox
  1. Push "Apply changes to filtered images" button  
  ![](pic/ss09.png)  
  "male focus" and "solo" will be replaced with "foo" and "bar".

#### I want to remove some tags  
  The same as replacing. Just replace the tags with "blank".  
  Also you can use "Remove" tab in "Batch Edit Captions".  
  ![](pic/ss10.png)  

#### I want to add/replace/remove tags more flexibly  
  1. Go to "Batch Edit Captions" tab
  2. Use "Search and Replace" with "Use regex" checked  
  ![](pic/ss11.png)  
  "1boy", "2boys", … will be replaced with "1girl", "2girls", … in each tags of images displayed.  
  A comma will be regarded as the sepalator of two tags.  
  By using regex, you can add/replace/remove tags according to more complex conditions.


## Trouble shooting
### Cannot see any image in dataset and no error is shown
(maybe in >= v0.0.6)  
If you want to load images from other directory than this app, you should register the directory in whitelist in the "Settings" tab, or use temporary image file (as same as the next section).  
Input path in "Path whitelist to show images …" and save settings.  
You can input drive name like "C:\\" (Windows).  
![](pic/ss12.png) 


### Cannot see any image in dataset and saying "All files must contained within the Gradio python app working directory…"
(maybe in <= v0.0.5)  
Set folder to store temporaly image in the "Settings" tab.  
Input path in "Directory to save temporary files" and check "Force using temporary file…" and save settings.  


### So laggy when opening many images or extremely large image
Input non-zero number in "Maximum resolution of ..." in the "Settings" tab to use smaller thumbnail for the image gallery.  
It may not work with dataset with millions of images.  
![](pic/ss13.png) 


### My PyTorch is working without CUDA
- To install PyTorch in your system to share it with other scripts
  1. Follow [the PyTorch installation guide](https://pytorch.org/get-started/locally/) with `-U` (`--upgrade`) option  
  (example) ```pip3 install -U torch torchvision --index-url https://download.pytorch.org/whl/cu118```
  1. Remove `venv` folder
  1. Run `install.bat`
- To install PyTorch only in venv  
  `activate` venv and install PyTorch with `-U` option, or do the following things:  
  1. Open `launch_user.bat` with some text editor
  2. Change the 3rd line to `set COMMANDLINE_ARGS="--force-install-torch cu118"` (you can choose from `cu117`, `cu118`, `cu121` or `cpu`)
  3. Run `launch_user.bat`
  4. (Remove the command line argument)
