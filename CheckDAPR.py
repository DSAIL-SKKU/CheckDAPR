import matplotlib.pyplot as plt
import base64
import anthropic
from ultralytics import YOLO
import imghdr

api_key = 'your-api-key'
client = anthropic.Anthropic(api_key=api_key)

def load_and_encode_images(paths):
    encoded_images = []
    for path in paths:
        with open(path, "rb") as image_file:
            image_data = image_file.read()
            img_type = imghdr.what(None, h=image_data)
            if img_type == "jpeg":
                media_type = "image/jpeg"
            elif img_type == "png":
                media_type = "image/png"
            else:
                print(f"Unsupported file format: {path}")
                continue

            encoded_image = base64.b64encode(image_data).decode("utf-8")
            encoded_images.append({
                "type": "image",
                "media_type": media_type,
                "data": encoded_image,
            })
    return encoded_images

def create_prompt_with_images(encoded_images, text):
    content = []
    for image in encoded_images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image["media_type"],
                "data": image["data"],
            },
        
        })
    content.append({
        "type": "text",
        "text": text,
    })
    
    prompts = [{
        "role": "user",
        "content": content,
    }]
    return prompts

def chat_with_images_and_text(image_urls, text_question):
    encoded_images = load_and_encode_images(image_urls)
    prompts = create_prompt_with_images(encoded_images, text_question)
    response = client.messages.create(model="claude-3-5-sonnet-20240620", max_tokens=3000, messages=prompts)
    return response

original_sketch = 'your-image-path' 

# stage 1

## ================================================
## object detection
model_path = './object-detection-model/yolov8-all.pt'

model = YOLO(model_path) 
results = model.predict(source=original_sketch,save=True) 
predict_sketch = results[0].save_dir + '/' + original_sketch.split('/')[-1]
## ================================================

image_sources = [original_sketch, predict_sketch]

stage1_prompt = '''

        You are an artificial intelligence assisting an art therapist.
        
        I will provide two images: one is a original sketch, and the other is the result of object detection. You need to analyze both images.
        
        You need to look for six specific objects in the images.
        The objects are: umbrella, rain, person, puddle, cloud, and lightning.
        Not every object will be present in each image.
        For 'rain', it should detect the presence, if it is present, you have to make the result 1.
        
        * To conduct the analysis, please follow these steps thoughtfully:
            1. You have to understand the sketch's context. Identify how many of the six objects (umbrella, rain, person, puddles, clouds, and lightning) are present in the original sketch; For 'rain', it should detect the presence, if it is present, you have to make the result 1.
            2. Compare the original sketch to the object detection result to assess the accuracy and completeness of the detected objects.
            3. Make final detection results of which objects are present and which are missing or inaccurately detected in the detected image compared with the original image.
        
        
        The output should be formatted as follows:
            [detection results]
            - umbrella: [number of umbrellas] {your description}
            - rain: [presence of rain] {your description}
            - person: [number of people] {your description}
            - puddle: [number of puddles]  {your description}
            - cloud: [number of clouds]  {your description}
            - lightning: [number of lightnings]  {your description}
            
            [contextual information]
            - umbrella: {contextual description}
            - rain: {contextual description}
            - person: {contextual description}
            - puddle: {contextual description}
            - cloud: {contextual description}
            - lightning: {contextual description}

        This structured approach will help in systematically assessing each object's presence and detection accuracy.
        Additionally, provide an opportunity to reconsider and adjust the final count if necessary, please read the overall sketch context.
        
        '''
    
# refined detection results and report
description_content = chat_with_images_and_text(image_sources, stage1_prompt)

result_text = ""
if hasattr(description_content, 'content'):
    for content_block in description_content.content:
        if content_block.type == "text":
            result_text += content_block.text + "\n"
        
print(result_text)
description_content = result_text

# stage 2
stage2_prompt = f'''

            You are an artificial intelligence assisting an art therapist. 

            I will provide two images: one is a original sketch, and the other is the result of object detection. 

            You need to analyze both images and calculate the 'DAPR score' based on the presence and interaction of various elements as detailed below:
            Based on the original sketch and the object detection results, here is the analysis of the six specific objects:
            {description_content}

            * To calculate 'dapr score' based on the analysis description below to determine which elements are correctly identified, which are missing, and which are inaccurately detected, 
                and follow these steps:

            * Stress Score:
                * Rain Presence:
                    * S1: No rain (0 if rain is present, 1 if absent).
                    * S2: Rain present (0 if no rain, 1 if rain is present).
                    * S3: Excess rain (calculate the space occupied by rain compared to the person; positive result indicates heavy rain, scored as 1, else 0).
                * Rain Interaction:
                    * S4: Style of Rain (0 if rain is depicted only as dots, 1 if depicted as other forms like circles, lines, teardrops).
                    * S6: Rain touching (0 if no contact with the person or their gear, 1 if there is contact).
                    * S8: Wind (0 if no wind is present, 1 if wind is present).
                * Puddles:
                    * S9: Puddles (0 if none, 1 for each puddle present).
                    * S10: Standing in puddle(s) (0 if not standing in or no puddles, 1 for each puddle making contact with the person).
                * More Rain Details:
                    * S11: Various rain style(s) (0 if rain is only dots, 1 for each additional form of rain depicted like lines, circles, teardrops).
                * Lightning:
                    * S13: Lightning bolt(s) (0 if none, 1 for each instance of lightning).
                    * S14: Lightning hit(s) (0 if no strike, 1 for each strike hitting the person or their gear).
                * Clouds:
                    * S15: Cloud(s) (0 if none, 1 for each cloud).
                    * S16: Dark Cloud(s) (0 if no dark clouds, 1 for each dark cloud).

            * Resource Score:
                * Protective Measures:
                    * R1: Protection present (0 if absent, 1 if present).
                    * R2: Umbrella present (0 if no umbrella, 1 if present, including folded).
                    * R3: Umbrella held (0 if holding oddly, 1 if correctly).
                * Protection Size:
                    * R5: Adequate size of protection (0 if object width ≤ person's width, 1 if greater).
                    * R6: Clear protection (0 if protection is damaged, 1 if it's intact).
                * Clothing and Appearance:
                    * R11: Whole face (0 if the face is covered by hat, umbrella, or shown in profile, 1 if the full face is visible).
                    * R12: Smile on Face (0 if no smile or expression on the face, 1 if there is a smile).
                * Person Placement and Size:
                    * R13: Centered figure (0 if off-center, 1 if centered).
                    * R14: Size of figure (0 if larger than 6 inches or smaller than 2 inches, 1 if between 2 and 6 inches).
                    * R15: Whole figure (0 if the person is shown from side, back, as a stick figure, head only or partial body, 1 if depicted from head to toe facing forward).

            * Dapr Score Calculation: Subtract the total Stress Score from the total Resource Score.

            * The output should be formatted as follows
                - Stress score: S1+S2+...+S16 = [score]
                - Resource score: R1+R2+...+R15 = [score]
                - total Score: [score]
            '''
            
# DAPR assessment scoring
DAPR_score = chat_with_images_and_text(image_sources, stage2_prompt)

result_text = ""
if hasattr(DAPR_score, 'content'):
    for content_block in DAPR_score.content:
        if content_block.type == "text":
            result_text += content_block.text + "\n"
        
print(result_text)