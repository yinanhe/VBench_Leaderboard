import os
# this is .py for store constants 
MODEL_INFO = [
    "Model Name (clickable)",
    "Source",
    "Total Score",
    "Quality Score",
    "Semantic Score",
    "Selected Score",
    ]

MODEL_INFO_TAB_QUALITY = [
    "Model Name (clickable)",
    "Quality Score",
    "Selected Score"
]

MODEL_INFO_TAB_I2V = [
    "Model Name (clickable)",
    "Total Score",
    "I2V Score",
    "Quality Score",
    "Selected Score"
]

TASK_INFO = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency"]

DEFAULT_INFO = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency"
    ]

QUALITY_LIST = [ 
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",]

SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency"
]

QUALITY_TAB = [ 
    "subject consistency",
    "background consistency",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",]

I2V_LIST = [
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
]

I2V_QUALITY_LIST = [
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality",
    "Temporal Flickering"
]

I2V_TAB = [
    "Video-Text Camera Motion",
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality",
    "Temporal Flickering"
]

DIM_WEIGHT = {
"subject consistency":1,
"background consistency":1,
"temporal flickering":1,
"motion smoothness":1,
"aesthetic quality":1,
"imaging quality":1,
"dynamic degree":0.5,
"object class":1,
"multiple objects":1,
"human action":1,
"color":1,
"spatial relationship":1,
"scene":1,
"appearance style":1,
"temporal style":1,
"overall consistency":1
}

DIM_WEIGHT_I2V = {
"Video-Text Camera Motion": 0.1,
"Video-Image Subject Consistency": 1,
"Video-Image Background Consistency": 1,
"Subject Consistency": 1,
"Background Consistency": 1,
"Motion Smoothness": 1,
"Dynamic Degree": 0.5,
"Aesthetic Quality": 1,
"Imaging Quality": 1,
"Temporal Flickering": 1
}

SEMANTIC_WEIGHT = 1
QUALITY_WEIGHT = 4
I2V_WEIGHT = 1.0
I2V_QUALITY_WEIGHT = 1.0

DATA_TITILE_TYPE = ['markdown', 'markdown', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']
I2V_TITILE_TYPE =  ['markdown', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']

SUBMISSION_NAME = "vbench_leaderboard_submission"
SUBMISSION_URL = os.path.join("https://huggingface.co/datasets/Vchitect/", SUBMISSION_NAME)
CSV_DIR = "./vbench_leaderboard_submission/results.csv"
QUALITY_DIR = "./vbench_leaderboard_submission/quality.csv"
I2V_DIR = "./vbench_leaderboard_submission/i2v_results.csv"
LONG_DIR = "./vbench_leaderboard_submission/long.csv"

COLUMN_NAMES = MODEL_INFO + TASK_INFO
COLUMN_NAMES_QUALITY = MODEL_INFO_TAB_QUALITY + QUALITY_TAB
COLUMN_NAMES_I2V = MODEL_INFO_TAB_I2V + I2V_TAB

LEADERBORAD_INTRODUCTION = """# VBench Leaderboard
    
    *"Which Video Generation Model is better?"*  
    🏆 Welcome to the leaderboard of the **VBench**! 🎦 *A Comprehensive Benchmark Suite for Video Generative Models* (**CVPR 2024 Spotlight**)   [![Code](https://img.shields.io/github/stars/Vchitect/VBench.svg?style=social&label=Official)](https://github.com/Vchitect/VBench) 
    <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 10px;">
    <a href='https://arxiv.org/abs/2311.17982'><img src='https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red'></a>
    <a href='https://vchitect.github.io/VBench-project/'><img src='https://img.shields.io/badge/VBench-Website-green?logo=googlechrome&logoColor=green'></a>
    <a href='https://pypi.org/project/vbench/'><img src='https://img.shields.io/pypi/v/vbench'></a>
    <a href='https://www.youtube.com/watch?v=7IhCC8Qqn8Y'><img src='https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red'></a>
    <a href='https://hits.seeyoufarm.com'><img src='https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FVBench&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false'></a>
    </div>
    
    - **Comprehensive Dimensions:** We carefully decompose video generation quality into 16 comprehensive dimensions to reveal individual model's strengths and weaknesses.
    - **Human Alignment:** We conducted extensive experiments and human annotations to validate robustness of VBench.
    - **Valuable Insights:** VBench provides multi-perspective insights useful for the community.  
    
    **Join Leaderboard**: Please see the [instructions](https://github.com/Vchitect/VBench/tree/master?tab=readme-ov-file#trophy-leaderboard) for 3 options to participate. One option is to follow [VBench Usage info](https://github.com/Vchitect/VBench?tab=readme-ov-file#usage), and upload the generated `result.json` file here. After clicking the `Submit here!` button, click the `Refresh` button.

    **Model Information**: What are the details of these Video Generation Models? See [HERE](https://github.com/Vchitect/VBench/tree/master/sampled_videos#what-are-the-details-of-the-video-generation-models)
    
    **Credits**: This leaderboard is updated and maintained by the team of [VBench Contributors](https://github.com/Vchitect/VBench?tab=readme-ov-file#muscle-vbench-contributors).
    """

SUBMIT_INTRODUCTION = """# Submit on VBench Benchmark Introduction

## 🎈
1. Please note that you need to obtain the file `evaluation_results/*.json` by running VBench in Github. You may conduct an [Offline Check](https://github.com/Vchitect/VBench?tab=readme-ov-file#get-final-score-and-submit-to-leaderboard) before uploading.
2. Then, pack these JSON files into a `ZIP` archive, ensuring that the top-level directory of the ZIP contains the individual JSON files. 
3. Finally, upload the ZIP archive below.

⚠️ Uploading generated videos or images of the model is invalid!
⚠️ Submissions that do not correctly fill in the model name and model link may be deleted by the VBench team. The contact information you filled in will not be made public. 
"""

TABLE_INTRODUCTION = """
    """

LEADERBORAD_INFO = """
       VBench, a comprehensive benchmark suite for video generative models. We design a comprehensive and hierarchical Evaluation Dimension Suite to decompose "video generation quality" into multiple well-defined dimensions to facilitate fine-grained and objective evaluation. For each dimension and each content category, we carefully design a Prompt Suite as test cases, and sample Generated Videos from a set of video generation models. For each evaluation dimension, we specifically design an Evaluation Method Suite, which uses carefully crafted method or designated pipeline for automatic objective evaluation. We also conduct Human Preference Annotation for the generated videos for each dimension, and show that VBench evaluation results are well aligned with human perceptions. VBench can provide valuable insights from multiple perspectives.
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""@inproceedings{huang2023vbench,
     title={{VBench}: Comprehensive Benchmark Suite for Video Generative Models},
     author={Huang, Ziqi and He, Yinan and Yu, Jiashuo and Zhang, Fan and Si, Chenyang and Jiang, Yuming and Zhang, Yuanhan and Wu, Tianxing and Jin, Qingyang and Chanpaisit, Nattapol and Wang, Yaohui and Chen, Xinyuan and Wang, Limin and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
     booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
     year={2024}
}"""

QUALITY_CLAIM_TEXT = "We use all the videos on Sora website (https://openai.com/sora) for a preliminary evaluation, including the failure case videos Sora provided."

I2V_CLAIM_TEXT = "Since the open-sourced SVD models do not accept text input during the I2V stage, we are unable to evaluate its `camera motion` in terms of `video-text consistency`. The total score is calculated based on all dimensions except `camera motion`."

LONG_CLAIM_TEXT = "" 

NORMALIZE_DIC = {
  "subject consistency": {"Min": 0.1462, "Max": 1.0},
  "background consistency": {"Min": 0.2615, "Max": 1.0},
  "temporal flickering": {"Min": 0.6293, "Max": 1.0},
  "motion smoothness": {"Min": 0.706, "Max": 0.9975},
  "dynamic degree": {"Min": 0.0, "Max": 1.0},
  "aesthetic quality": {"Min": 0.0, "Max": 1.0},
  "imaging quality": {"Min": 0.0, "Max": 1.0},
  "object class": {"Min": 0.0, "Max": 1.0},
  "multiple objects": {"Min": 0.0, "Max": 1.0},
  "human action": {"Min": 0.0, "Max": 1.0},
  "color": {"Min": 0.0, "Max": 1.0},
  "spatial relationship": {"Min": 0.0, "Max": 1.0},
  "scene": {"Min": 0.0, "Max": 0.8222},
  "appearance style": {"Min": 0.0009, "Max": 0.2855},
  "temporal style": {"Min": 0.0, "Max": 0.364},
  "overall consistency": {"Min": 0.0, "Max": 0.364}
}

NORMALIZE_DIC_I2V = {
    "Video-Text Camera Motion" :{"Min": 0.0, "Max":1.0 },
    "Video-Image Subject Consistency":{"Min": 0.1462, "Max": 1.0},
    "Video-Image Background Consistency":{"Min": 0.2615, "Max":1.0 },
    "Subject Consistency":{"Min": 0.1462, "Max": 1.0},
    "Background Consistency":{"Min": 0.2615, "Max": 1.0 },
    "Motion Smoothness":{"Min": 0.7060, "Max": 0.9975},
    "Dynamic Degree":{"Min": 0.0, "Max": 1.0},
    "Aesthetic Quality":{"Min": 0.0, "Max": 1.0},
    "Imaging Quality":{"Min": 0.0, "Max": 1.0},
    "Temporal Flickering":{"Min":0.6293, "Max": 1.0}
}
