## BABEL-QA

<div>
  <img src="../assets/babel-qa_figure.png", width="100%">
</div>

### Obtain AMASS and BABEL data
Generating our dataset requires access to [AMASS](https://amass.is.tue.mpg.de/) and [BABEL](https://babel.is.tue.mpg.de/). Follow each webpage for download instructions. For AMASS, download SMPL+H versions of each sub-dataset, and move each collection into a shared folder.

Once each dataset is downloaded, set environment variables for `amass_root` and `babel_root`.

### Download SMPL+H model
Download the SMPL+H model from [MANO](https://mano.is.tue.mpg.de/) (choose Extended SMPL+H used in AMASS project), and set the `smplh_root` environment variable to the saved location.

### Extract motion concepts
We extract motion concepts from the sequences by parsing frame-level label texts and action categories. Before running the script, set `data_dir` to where you would like the BABEL-QA dataset to be stored.

```bash
python extract_motion_concepts.py \
--babel_root $babel_root \
--data_dir $data_dir
```

### Generate questions
Using the extracted motion concepts, we next construct questions. We've created a file called `split_question_ids.json` that contains which question ids are included in train, val, and test.

```bash
python generate_questions.py \
--data_dir $data_dir \
--data_split_file split_question_ids.json
```

### Process AMASS dataset
We process the AMASS dataset similarly to [HuMoR](https://github.com/davrempe/humor) in order to sample motions to 30 Hz, detect contacts, etc. If you would like to extract other pose representations for your work, you can modify the file to do so.

```bash
cd process_amass_dataset

python process_amass_data.py \
--data_dir $data_dir \
--babel_root $babel_root \
--amass_root $amass_root \
--smplh_root $smplh_root
```


After completing these steps, the dataset directory has the following structure:

```
BABEL-QA
├── questions.json
├── motion_concepts.json
└── motion_sequences
    ├── 1833
    │   ├── joints.npy
    │   └── pose.npy
    └── ...
```

<details>
<summary><h3>Visualizing motion sequences</h3></summary>

```bash
cd process_amass_dataset

python visualize_motion_sequences.py \
--vis_babel_ids_path vis_babel_ids.txt \
--data_dir $data_dir \
--vis_joints
```
</details>