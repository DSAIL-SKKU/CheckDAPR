# CheckDAPR: An MLLM-based DAPR Scoring System for Art Therapy

**CheckDAPR** is a novel DAPR assessment system, which can automatically analyze the drawing test, Draw-A-Person-in-the-Rain (DAPR), a psychological drawing assessment used for identifying stressful experiences and coping behavior. This system use Multimodal Large Language Model(MLLM) to refine object detection results and corrects inaccuracies in sketches by evaluating existing models, and its MLLM also produces a DAPR assessment report that calculates detailed scores.

## Get Started

```
git clone https://github.com/DSAIL-SKKU/CheckDAPR.git
cd CheckDAPR
pip install -r requirements.txt
```

## Example
This example image is based on the [SceneDAPR](https://github.com/DSAIL-SKKU/SceneDAPR) test set.
``` example/ ``` folder contains txt files for the results of stage1 and stage2.

## DAPR assessment task
```
python CheckDAPR.py
```
This code uses the pre-trained object detection model **YOLO-v8** and it employs **Claude 3.5** Sonnet as the MLLM. 

In the API key section of this code (line 7), enter your own API key.
```
# API Key
api_key = "your api key"
```