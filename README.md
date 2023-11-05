# MOSA: Music mOtion and Semantic Annotation dataset

MOSA dataset is a large-scale music dataset containing 742 professional piano and violin solo music performances with 23 musicians (> 30 hours, and > 570 K notes). This dataset features following types of data:
- **High-quality 3-D motion capture data**
- **Audio recordings**
- **Manual semantic annotations**

This is the implementation of the paper:
Huang et al., MOSA: Music Motion with Semantic Annotation Dataset for Multimedia Anaysis and Generation.

**Note:** This paper is under peer-review process, and only a small portion of sample data are released in the `.\sample_data\` folder.
The full dataset will be avaiable in the `.\MOSA-data\` folder when the paper is officially published.

![alt text](https://github.com/yufenhuang/MOSA-Music-mOtion-and-Semantic-Annotation-dataset/blob/main/figure/dataset.png)



<p align="center">
  
https://github.com/yufenhuang/MOSA-Music-mOtion-and-Semantic-Annotation-dataset/assets/42167635/13426b3c-1133-4a4d-96cd-860e6b772b1f

</p>



## 3-D motion capture data
The 3-D motion capture data in this dataset are recorded using Qualisys and Vicon 3-D motion capture system. The original coordinates of 34 body/instrument markers on the x-, y-, and z- axes are provided. The body marker placement
follows the Plug-In Gait Full-body model in Visual-3D's official documention.

![alt text](https://github.com/yufenhuang/MOSA-Music-mOtion-and-Semantic-Annotation-dataset/blob/main/figure/mocap.png)

## Music semantic annotation

In MOSA dataset, we assemble high-quality manual-crafted semantic annotations for music performance. The statistics of annotations in MOSA dataset are shown in the Figure , and the collection of music annotations includes:

- **Note annotations:** pitch name (e.g. C4), midi number, note onset time, note offset time, note duration.
- **Beat/downbeat annotations:** the position of beat and downbeat, beat time, downbeat time.
- **Harmony annotations:** key (e.g. C major), Roman notation for harmony analysis, root degree (I - VII), inversion (root, 1st - 3rd inversion), chord quality (major triad (M), minor triad (m), augmented triad (A), diminished triad (d), dominant seventh (D7), major seventh (M7), minor seventh (m7),  diminished seventh (d7), half-diminished seventh (h7), augmented sixth (A6)),  chord onset time, chord offset time.
- **Expressive annotations:** dynamic marks (ppp - fff, crescendo, diminuendo, accent), tempo variations (accelerado, ritenuto, a tempo), and articulations (legato, staccato).
- **Cadence annotations:** high-level harmonic progression in music, cadence type (authentic (AC), half (HC), deceptive cadence (DC)), cadence onset time, cadence resolve time, cadence offset time.
- **Structural annotations:** phrase boundary, section boundary, motive, section type (e.g. exposition, development, recapitulation).

![alt text](https://github.com/yufenhuang/MOSA-Music-mOtion-and-Semantic-Annotation-dataset/blob/main/figure/annot.png)

## Audio recordings
This dataset provides audio recordings synchronized with 3-D body motion and music semantic annotation

# Usage Guidelines

This project is a part of the research project 'Deep learning models for non-verbal communication: Apply on music semantics' of National Science and Technology Council, Taiwan. The MOSA dataset is released under the Creative Commons Public Domain [CC4](https://creativecommons.org/licenses/by/4.0/) license. Ensure that your use of MOSA dataset does not breach any legislation, notably concerning data protection, defamation or copyright. The MOSA dataset is restricted to be used for the research regarding the body motion, audio performance, and semantics for violin and piano performances in Classical music. The authors make no representations or warranties for any extended use of data. To see detailed information about the dataset, please go to [here](https://github.com/yufenhuang/MOSA-Music-mOtion-and-Semantic-Annotation-dataset/blob/main/MOSA-dataset/dataset.md)

This project provides implement codes for 3 topics:
- **Musician's body motion generation from audio**
- **Time semantics recognition**
- **Expressive semantics recognition**

## Dependencies
- Python 3+
- CUDA 11.8
- Tensorflow 2.9
- Install requirements by running: 
`pip install -r requirement.txt`

## Musician's body motion generation from audio



https://github.com/yufenhuang/MOSA-Music-mOtion-and-Semantic-Annotation-dataset/assets/42167635/9374a3a6-8af0-4bfc-a63b-c2b250906fa2



- **Training from scratch**

  To train a body motion generation model from scratch, please run the following commands.

  The trained model will be saved to the folder `.\motion_generation_model\motion_generation_transformer\`
  
  ```
  python motion_generation_model\data_preprocessing_motion_generation.py
  python motion_generation_model\train_model_motion_generation.py
  ```

- **Inference**

  To make the body motion video, please select an audio file from `.\sample_data\` folder, and run the following commands.

  The generated video will be saved to the folder `.\motion_generation_model\`
  
  ```
  python motion_generation_model\inference_motion_generation.py audiofilename
  ```
  For example,
  ```
  python motion_generation_model\inference_motion_generation.py de1_yv10_t1_audio
  ```

## Time semantics recognition
- **Training from scratch**

   To train a time semantics recognition model from scratch, please run the following commands.
  
   The trained model will be saved to the folder `.\time_semantics_model\time_semantics_transformer\`
    ```
    python time_semantics_model\data_preprocessing_time_semantics.py
    python time_semantics_model\train_model_time_semantics.py
    ```

- **Evaluation**

  To reproduce the results of pre-trained model, please run the following commands.
  ```
  python time_semantics_model\data_preprocessing_time_semantics.py
  python time_semantics_model\evaluation_time_semantics.py
  ```

  ## Expressive semantics recognition
- **Training from scratch**

  To train a expressive semantics recognition model from scratch, please run the following commands.

  The trained model will be saved to the folder `.\expressive_semantics_model\expression_semantics_transformer\`
  ```
  python expressive_semantics_model\data_preprocessing_expressive_semantics.py
  python expressive_semantics_model\train_model_expressive_semantics.py
  ```

- **Evaluation**
    
  To reproduce the results of pre-trained model, please run the following commands.
  ```
  python expressive_semantics_model\data_preprocessing_expressive_semantics.py
  python expressive_semantics_model\evaluation_expressive_semantics.py
  ```
  
 
  





